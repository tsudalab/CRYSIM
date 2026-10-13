import numpy as np
from logzero import logger as log_printer
from pymatgen.core import Structure
from copy import deepcopy

from crysim.src import DATASET_ROOT
from crysim.src.registry import mixing
from crysim.src.optimizer.rand_gen import RandGenDirector

from crysim.src.crysvec.pos import PosMatDiscretizer, decode_ind_sites, perturb_overlapped_atoms
from crysim.src.crysvec.emb import MatProcessor, default_wpc_prec, Constraints, oh_name
from crysim.src.crysvec.utils import crys_embbeder
from crysim.src.spg.spg_utils import analyze_wpc_list_element, convert_pyx_wpc_to_str
from crysim.src.struct_utils import unique_elements, get_crystal_lattice, get_species_list


##### init dataset generator #####


class RandGenMixFilter:
    def __init__(self, dim, system_name, filter_dist, gen_tool, n_struct, 
                 len_lbs, len_ubs, ang_lbs, ang_ubs, seed=0, id_offset_for_rg=0, init_spg_list=None,
                 logger=None, file_root=DATASET_ROOT, load_cache=True, custom_name=mixing):
        self.dim = dim
        self.rg = RandGenDirector(dim=dim, system_name=system_name, gen_tool=gen_tool,
                                  len_lbs=len_lbs, len_ubs=len_ubs, ang_lbs=ang_lbs, ang_ubs=ang_ubs, 
                                  n_struct=n_struct, seed=seed, filtering_fn=self.filtering,
                                  id_offset_for_rg=id_offset_for_rg, filter_dist=filter_dist,
                                  init_spg_list=init_spg_list, logger=logger, custom_name=custom_name,
                                  is_insist_wpclist=True,
                                  file_root=file_root, load_cache=load_cache)
        return

    def filtering(self, pmg_struct):
        ele_coords_dict = {}
        for site in pmg_struct.sites:
            if site.specie.name not in ele_coords_dict:
                ele_coords_dict[site.specie.name] = [site.frac_coords]
            else:
                ele_coords_dict[site.specie.name].append(site.frac_coords)
        return check_first_species_coordinates_by_fn(get_species_list(self.rg.system_name), ele_coords_dict, axis='z')

    def __call__(self, *args, **kwargs):
        return self.rg()


##### Encoder & decoder #####


class PosDiscreteAddEncoder:
    def __init__(self, pos_splits):
        self.coord_repr = PosMatDiscretizer(pos_splits=pos_splits)
        self.pos_splits, self.pos_units, self.pos_prec = self.coord_repr.pos_splits, self.coord_repr.pos_units, self.coord_repr.pos_prec
        assert self.pos_splits.ndim == 1
        return
    
    @property
    def num_var(self):
        return self.coord_repr.num_var
    
    def encode(self, pmg_struct):
        frac_coords = pmg_struct.frac_coords
        pos_vec, _ = self.coord_repr.encode_coords(frac_coords, self.pos_splits, self.pos_units)
        return [pos_vec]
    
    
def distance_to_surface(coord, _axis):
    x, y, z = coord
    dx = min(x, 1 - x) if 'x' in _axis else 999
    dy = min(y, 1 - y) if 'y' in _axis else 999
    dz = min(z, 1 - z) if 'z' in _axis else 999
    return min(dx, dy, dz)
    
    
def distance_to_ground(coord, _axis):
    x, y, z = coord
    dx = x if 'x' in _axis else 999
    dy = y if 'y' in _axis else 999
    dz = z if 'z' in _axis else 999
    return -min(dx, dy, dz)


_default_fn = distance_to_ground


def sort_coordinates_by_fn(coordinates, axis, fn=_default_fn):
    return np.array(sorted(coordinates, key=lambda x: fn(x, _axis=axis)))


def select_idx_uniformly(list_len, num):
    if num == 1:
        return [0]
    int_indices = []
    for i in range(num):
        index = (i * (list_len - 1)) // (num - 1)
        int_indices.append(index)
    return int_indices


def is_wpc_z_zeros(wpc):
    wpc_coords = wpc.split(', ')
    z_ids = range(2, len(wpc_coords), 3)
    z_coords = [wpc_coords[i] for i in z_ids]
    if 'z' not in z_coords and '-z' not in z_coords:
        return True
    else:
        return False


def check_first_species_coordinates_by_fn(species_list, ele_coordinates, axis, fn=_default_fn):
    ele, _ = unique_elements(species_list)
    ele_first_coord = {}
    for e, e_coords in ele_coordinates.items():
        sorted_e_coords_idx = sorted(list(range(len(e_coords))), key=lambda i: fn(e_coords[i], _axis=axis))
        ele_first_coord[e] = e_coords[sorted_e_coords_idx[0]]
    ele_first_list, coord_first_list = list(ele_first_coord.keys()), list(ele_first_coord.values())
    sorted_first_ele_list = sorted(list(range(len(coord_first_list))), key=lambda i: fn(coord_first_list[i], _axis=axis))
    sorted_first_ele_list = [ele_first_list[i] for i in sorted_first_ele_list]
    if not np.all(ele == sorted_first_ele_list):
        return False
    else:
        return True


class PosDiscreteAddIndDecoderByRule:
    def __init__(self, species_list, pos_splits, decoding_fn, sorting_fn, sorting_axis):
        # core algorithm: if there is a fixed site (in 2d, z of fixed site is always 0) decoded for elements
        # ought to be outside, then elements after it should be assigned with all 0s in the z axis
        self.species_list = species_list
        self.elements, self.num_species_atom = unique_elements(species_list=species_list)
        self.coord_repr = PosMatDiscretizer(pos_splits=pos_splits)
        self.pos_splits, self.pos_units, self.pos_prec = self.coord_repr.pos_splits, self.coord_repr.pos_units, self.coord_repr.pos_prec
        
        self.decoding_fn = decoding_fn
        self.sorting_fn, self.sorting_axis = sorting_fn, sorting_axis
        return
    
    def decode(self, pos_vec, wpc_lists):
        usable_wpc_list, ind_sites_num_list = [], []
        for wpc in wpc_lists:
            wpc_element, ind_num = analyze_wpc_list_element(wpc, modify=False)
            usable_wpc_list.append(wpc_element)
            ind_sites_num_list.append(ind_num)
        num_ind_sites = sum(ind_sites_num_list)
        
        # 1. decode frac_coords & sort them by the given rule
        decoded_frac_coords = self.coord_repr.decode_coords(pos_vec, self.pos_splits, self.pos_units, self.decoding_fn, sum(self.num_species_atom))
        decoded_frac_coords = sort_coordinates_by_fn(decoded_frac_coords, self.sorting_axis, fn=self.sorting_fn)
        
        # 2. select ind sites to be filled into WP variables
        modify_z_afterwards = False
        ids_selected = select_idx_uniformly(len(decoded_frac_coords), num_ind_sites)
        
        # 3. fill in WP variables for each element
        atom_list, atom_form, decoded_atom_list = [], [], []
        seq_starting_ids = 0
        used_atoms = []
        for wpc_list_element, num_ind_site_ele in zip(usable_wpc_list, ind_sites_num_list):
            # finding ind site for the current element
            if len(ids_selected) > 0:
                ind_site_ele = ids_selected[seq_starting_ids: seq_starting_ids + num_ind_site_ele]
                ind_site_ele = decoded_frac_coords[ind_site_ele]
            else:
                ind_site_ele = []
            used_atoms.append(ind_site_ele)
            
            # fill in WP variables
            atom_list_ele, atom_form_ele, decoded_atom_list_ele = \
                decode_ind_sites(ind_site_ele, ordered_wpc_list_element=wpc_list_element)
            
            # modify z to guarantee the sequence of elements
            if modify_z_afterwards:
                for i in range(len(atom_list_ele)):
                    atom_list_ele[i][-1] = 0.0
            
            seq_starting_ids += num_ind_site_ele

            atom_list += atom_list_ele
            atom_form += atom_form_ele
            decoded_atom_list += decoded_atom_list_ele
            
            # check z-axis: only modify when there are only fixed atoms
            wp_formula_form = [wp[1] for wp in wpc_list_element if wp[0] == 'f']
            if num_ind_site_ele == 0:
                modify_z_afterwards = True
            elif sum(is_wpc_z_zeros(wpc) for wpc in wp_formula_form) == num_ind_site_ele:
                modify_z_afterwards = True
        
        # 4. check whether generated structure satisfy the sorting function
        ele_dict = {}
        ele_atom_idx = np.cumsum(np.concatenate([[0], self.num_species_atom]))
        for i, ele in enumerate(self.elements):
            ele_dict[ele] = atom_list[ele_atom_idx[i]: ele_atom_idx[i + 1]]
        assert check_first_species_coordinates_by_fn(self.species_list, ele_dict, axis=self.sorting_axis)
        log_printer.info('decoded atoms pass initial structure test')
        
        atom_list = perturb_overlapped_atoms(atom_list, atom_form, decoded_atom_list, self.species_list, self.elements, use_pbc=False)
        return atom_list, used_atoms


##### Embedder #####


class ConstraintsMixing(Constraints):
    def __init__(self, lat_prec, sym_prec, coords_prec):
        super().__init__(lat_prec=lat_prec, sym_prec=sym_prec, coords_prec=coords_prec)
        return
    
    def atom_number_constraints(self, num_species_atom, pos_split):
        assert self.coords_prec == np.prod(pos_split)
        num_constr = 1
        atom_num_mat = self.init(num_constr)
        atom_num_mat[0, -self.coords_prec:] = 1
        return [atom_num_mat, [sum(num_species_atom)], [1]]


class MatCSSPGMixProcessor(MatProcessor):
    def __init__(self, dim, species_list, splits, len_lbs, len_ubs, ang_lbs, ang_ubs, len_splits, ang_splits, decode_num, decoding_fn, 
                 include_spg_list, wpc_prec=default_wpc_prec, sorting_fn=_default_fn):
        self.sorting_fn, self.sorting_axis = sorting_fn, 'z'
        super().__init__(emb_type='latdis+oh+struct', dim=dim, species_list=species_list, splits=splits, 
                         len_lbs=len_lbs, len_ubs=len_ubs, ang_lbs=ang_lbs, ang_ubs=ang_ubs, len_splits=len_splits, ang_splits=ang_splits,
                         decoding_fn=decoding_fn, decode_num=decode_num, wpc_prec=wpc_prec, include_spg_list=include_spg_list)
        return
    
    def _load_pos_emb(self):
        self.pos_enc = PosDiscreteAddEncoder(pos_splits=self.splits)
        self.pos_dec = PosDiscreteAddIndDecoderByRule(species_list=self.species_list, 
                                                      pos_splits=self.splits, decoding_fn=self.decoding_mat_fn, 
                                                      sorting_fn=self.sorting_fn, sorting_axis=self.sorting_axis)
        self.coords_prec = self.pos_enc.pos_prec
        return

    def decode(self, struct_vec):
        spg_list, wpc_idx_list, wpc_list_list, loss_list, ids = self.symmetry_decode(struct_vec)
        struct_vec = [struct_vec] * len(spg_list) if (struct_vec.ndim == 1 or len(struct_vec) == 1) else struct_vec
        struct_list, used_atoms_list = [], []
        pass_list = []
        for i in range(len(ids)):
            spg, wpc_list, solved_value, idx = spg_list[i], wpc_list_list[i], loss_list[i], ids[i]
            wpc_list = convert_pyx_wpc_to_str(spg=spg, wpc=wpc_list, dim=self.dim)
            log_printer.info(f"Generating {i}-th struct with {spg} SG, {wpc_list} WPC with {solved_value} MAE")
            lat = self.lat_dec.decode(self.get_lat_vec(struct_vec[idx]), crys_sys=get_crystal_lattice(spg, self.dim))
            pos, used_atoms = self.pos_dec.decode(self.get_pos_vec(struct_vec[idx]), wpc_lists=wpc_list)
            pmg_struct = Structure(lattice=lat, species=self.species_list, coords=pos, coords_are_cartesian=False)
            struct_list.append(pmg_struct)
            used_atoms_list.append(used_atoms)
            
            # Another filtering is necessary due to the position shift for preventing overlaps
            species_list = [s.name for s in pmg_struct.species]
            ele_coords_dict = {}
            for site in pmg_struct.sites:
                if site.specie.name not in ele_coords_dict:
                    ele_coords_dict[site.specie.name] = [site.frac_coords]
                else:
                    ele_coords_dict[site.specie.name].append(site.frac_coords)
            is_passed = check_first_species_coordinates_by_fn(species_list, ele_coords_dict, axis=self.sorting_axis)
            log_printer.info(f"Generating {i}-th struct with {spg} SG, {wpc_list} WPC, {is_passed} pass relative position check")
            pass_list.append(is_passed)
            
        if sum(pass_list) == 0:
            self.spg, self.wpc_idx, self.wpc, self.used_atoms, self.ids = spg_list, wpc_idx_list, wpc_list_list, used_atoms_list, ids
            self.cs = [get_crystal_lattice(spg, self.dim) for spg in spg_list]
            log_printer.info('No structs passed the check, output all')
            return struct_list
        else:
            self.spg, self.wpc_idx, self.wpc, self.used_atoms, self.ids = \
                [spg for i, spg in enumerate(spg_list) if pass_list[i]], \
                [wpc_idx for i, wpc_idx in enumerate(wpc_idx_list) if pass_list[i]], \
                [wpc for i, wpc in enumerate(wpc_list_list) if pass_list[i]], \
                [used_atoms for i, used_atoms in enumerate(used_atoms_list) if pass_list[i]], \
                [idx for i, idx in enumerate(ids) if pass_list[i]]
            self.cs = [get_crystal_lattice(spg, self.dim) for spg in self.spg]
            passed_struct_list = [pmg_s for i, pmg_s in enumerate(struct_list) if pass_list[i]]
            log_printer.info(f'Output {len(passed_struct_list)} structs that passed the check')
            return passed_struct_list

    def get_constraints(self):
        constr = ConstraintsMixing(lat_prec=self.lat_enc.num_var, sym_prec=self.sym_emb.num_var, coords_prec=self.pos_enc.num_var)
        lat_num = constr.lattice_number_constraints(self.len_splits, self.ang_splits)
        sym_num = constr.csspg_constraints(self.sym_emb.cs_prec, self.sym_emb.spg_prec, self.sym_emb.wpc_prec)
        atom_num = constr.atom_number_constraints(self.num_species_atom, self.splits)
        return [np.concatenate([lat_num[0], sym_num[0], atom_num[0]], axis=0), 
                lat_num[1] + sym_num[1] + atom_num[1], lat_num[2] + sym_num[2] + atom_num[2]]

crys_embbeder[f'latdis+oh+struct-{mixing}-z'] = MatCSSPGMixProcessor
