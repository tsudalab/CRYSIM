import numpy as np
from copy import deepcopy
from logzero import logger as log_printer

from crysim.src import DEFAULT_NUMPY_DTYPE
from crysim.src.struct_utils import frac_pbc, perturb_frac_atom, unique_elements
from crysim.src.spg.spg_utils import analyze_wpc_list_element, StandardSPGDecoder


num_coords = 3


def fraccoords2bits(sites, units, splits):
    """ units: list of units in x, y, z axes 
        sites.shape == (n, 3); np.array(units).shape == np.array(splits).shape == (3,) """
    sites = frac_pbc(np.array(sites))
    sites = np.around(sites / np.array(units)).astype(int)  # broadcast
    for i in range(3):
        sites[:, i] = np.where(sites[:, i] >= splits[i], 0, sites[:, i])
    return sites


def bits2fraccoords(pos_bits, num_atom, units):
    default_pos = (0.3, 0.2, 0.6)
    sites = pos_bits * np.array(units)  # broadcast
    if len(sites) < num_atom:
        add_site_list = []
        log_printer.warning('Use default position')
        for _ in range(num_atom - len(sites)):
            add_site_list.append([default_pos[0] + perturb_frac_atom(), 
                                  default_pos[1] + perturb_frac_atom(), 
                                  default_pos[2] + perturb_frac_atom()])
        add_site_list = np.vstack(add_site_list)
        sites = np.concatenate([sites, add_site_list], axis=0)
    return sites


class PosFrac:
    def __init__(self, **kwargs):
        num_species_atom = kwargs.get('num_species_atom')
        self.pos_splits = [[num_coords] for _ in num_species_atom]
        self.pos_units = [[0] for _ in num_species_atom]
        self.pos_prec = [num_atom * num_coords for num_atom in num_species_atom]
        return
    
    @property
    def num_var(self):
        return sum(self.pos_prec)
    
    def encode_coords(self, frac_coords, splits, units):
        return frac_coords.flatten(), None
    
    def decode_coords(self, pos_vec, splits, units, decoding_fn, num_atom):
        return pos_vec.reshape(-1, num_coords).astype(DEFAULT_NUMPY_DTYPE)


class PosSiteDiscretizer:
    def __init__(self, **kwargs):
        num_species_atom, pos_splits = kwargs.get('num_species_atom'), kwargs.get('pos_splits')
        self.pos_splits = np.array(pos_splits)
        assert self.pos_splits.ndim == 2
        assert len(num_species_atom) == len(self.pos_splits)
        self.pos_units = 1 / self.pos_splits
        self.pos_prec = [num_atom * sum(pos_split) for num_atom, pos_split in zip(num_species_atom, self.pos_splits)]
        return
    
    @property
    def num_var(self):
        return sum(self.pos_prec)
    
    def encode_coords(self, frac_coords, splits, units):
        pos_vec = np.zeros(shape=(len(frac_coords), sum(splits)))
        if len(frac_coords) > 0:
            pos_bits = fraccoords2bits(frac_coords, units=units, splits=splits)
            for i, bit in enumerate(pos_bits):
                pos_vec[i, bit[0]] = 1
                pos_vec[i, splits[0] + bit[1]] = 1
                pos_vec[i, splits[0] + splits[1] + bit[2]] = 1
        else:
            pos_bits = None
        return pos_vec.flatten(), pos_bits
    
    def decode_coords(self, pos_vec, splits, units, decoding_fn, num_atom):        
        pos_vec = pos_vec.reshape(num_atom, sum(splits))
        cum_splits = np.cumsum(splits)[:-1]
        x_vec, y_vec, z_vec = np.split(pos_vec, cum_splits, axis=1)
        x_bits, y_bits, z_bits = np.argmax(x_vec, axis=1), np.argmax(y_vec, axis=1), np.argmax(z_vec, axis=1)
        pos_bits = np.vstack([x_bits, y_bits, z_bits]).T
        decoded_frac_coords = bits2fraccoords(pos_bits, num_atom=num_atom, units=units).astype(DEFAULT_NUMPY_DTYPE)
        return decoded_frac_coords


class PosMatDiscretizer:
    def __init__(self, **kwargs):
        pos_splits = kwargs.get('pos_splits')
        self.pos_splits = np.array(pos_splits)
        assert self.pos_splits.ndim in [1, 2]
        if self.pos_splits.ndim == 2 and self.pos_splits.shape[0] == 1:
            self.pos_splits = self.pos_splits.flatten()
        self.pos_units = 1 / self.pos_splits
        if self.pos_splits.ndim == 1:  # when atoms of different species can be mixed in one vector
            self.pos_prec = [np.prod(self.pos_splits)]
        else:
            self.pos_prec = [np.prod(pos_split) for pos_split in self.pos_splits]
        return
    
    @property
    def num_var(self):
        return sum(self.pos_prec)
    
    def encode_coords(self, frac_coords, splits, units):        
        pos_vec = np.zeros(shape=(splits[0], splits[1], splits[2]))
        if len(frac_coords) > 0:
            pos_bits = fraccoords2bits(frac_coords, units=units, splits=splits)
            for bit in pos_bits:
                pos_vec[bit[0], bit[1], bit[2]] = 1
        else:
            pos_bits = None
        return pos_vec.flatten(), pos_bits
    
    def decode_coords(self, pos_vec, splits, units, decoding_fn, num_atom):           
        pos_vec = pos_vec.reshape(splits[0], splits[1], splits[2])
        pos_bits = decoding_fn(pos_vec)
        pos_bits = np.array([[pos_bits[0][i], pos_bits[1][i], pos_bits[2][i]] for i in range(len(pos_bits[0]))])
        decoded_frac_coords = bits2fraccoords(pos_bits, num_atom=num_atom, units=units).astype(DEFAULT_NUMPY_DTYPE)
        return decoded_frac_coords


##### Normal Encoder #####


class PosConcatEncoder:
    def __init__(self, coord_representor, species_list, pos_splits_list):
        self.elements, self.num_species_atom = unique_elements(species_list=species_list)
        assert len(self.elements) == len(pos_splits_list)
        self.coord_repr = coord_representor(num_species_atom=self.num_species_atom, pos_splits=pos_splits_list)
        self.pos_splits, self.pos_units, self.pos_prec = self.coord_repr.pos_splits, self.coord_repr.pos_units, self.coord_repr.pos_prec
        return
    
    @property
    def num_var(self):
        return self.coord_repr.num_var
    
    def encode(self, pmg_struct):
        frac_coords_dict = {e: [] for e in self.elements}
        for site in pmg_struct.sites:
            frac_coords_dict[site.species_string].append(site.frac_coords)
        pos_vec = []
        for i, e in enumerate(self.elements):
            frac_coords_element = np.vstack(frac_coords_dict[e])
            splits, units = self.pos_splits[i], self.pos_units[i]
            pos_vec_element, _ = self.coord_repr.encode_coords(frac_coords_element, splits, units)
            pos_vec.append(pos_vec_element)
        return [np.concatenate(pos_vec)]


def organize_ind_free_sites_list(elements, num_species_atom, used_atoms, wp_gen):
    """ saperate used_atoms into one list for each element """
    if isinstance(used_atoms, dict):
        return [used_atoms[e] for e in elements]
    elif isinstance(used_atoms[0], list) or \
            (isinstance(used_atoms[0], np.ndarray) and used_atoms[0].ndim == 2):  # list of list of used_atoms for each elements respectively
        return used_atoms
    else:  # list of used_atoms
        used_atoms_list = [[] for _ in num_species_atom]
        curr_k, curr_cum_num_atom = 0, 0
        cum_num_atom = np.cumsum(num_species_atom)
        wp_gen_flat = []
        for wp_ele in wp_gen:
            wp_gen_flat += wp_ele
        for wp, site in zip(wp_gen_flat, used_atoms):
            mtp, letter, dof = wp
            used_atoms_list[curr_k].append(site)
            curr_cum_num_atom += mtp
            if curr_cum_num_atom == cum_num_atom[curr_k]:
                curr_k += 1
        return used_atoms_list


class PosSiteEncoder:
    def __init__(self, coord_representor, species_list, pos_splits_list):
        self.elements, self.num_species_atom = unique_elements(species_list=species_list)
        assert len(self.elements) == len(pos_splits_list)
        self.coord_repr = coord_representor(num_species_atom=self.num_species_atom, pos_splits=pos_splits_list)
        self.pos_splits, self.pos_units, self.pos_prec = self.coord_repr.pos_splits, self.coord_repr.pos_units, self.coord_repr.pos_prec
        return
    
    @property
    def num_var(self):
        return self.coord_repr.num_var
    
    def encode(self, used_atoms, wp_gen):  # a list of all ind sites, not a list of lists of ind sites for elements
        ind_sites_list = organize_ind_free_sites_list(self.elements, self.num_species_atom, used_atoms, wp_gen)
        pos_vec = []
        for i, (wpc, sites) in enumerate(zip(wp_gen, ind_sites_list)):
            splits, units = self.pos_splits[i], self.pos_units[i]
            frac_coords_element = []
            for wp, site in zip(wpc, sites):  # ignore static sites
                mtp, letter, dof = wp
                if dof > 0:
                    frac_coords_element.append(site)
            if len(frac_coords_element) > 0:
                frac_coords_element = np.vstack(frac_coords_element)
                pos_vec_element, _ = self.coord_repr.encode_coords(frac_coords_element, splits, units)
            else:
                pos_vec_element = []
            supp_atom_num = self.num_species_atom[i] - len(frac_coords_element)  # supplement -1 after encoding
            pos_vec_element = np.concatenate([pos_vec_element, [-1] * supp_atom_num * sum(splits)])
            pos_vec.append(pos_vec_element)
        return [np.concatenate(pos_vec)]


##### Normal Decoder #####


class PosConcatAllDecoder:
    def __init__(self, coord_representor, species_list, pos_splits_list, decoding_fn):
        self.elements, self.num_species_atom = unique_elements(species_list=species_list)
        assert len(self.elements) == len(pos_splits_list)
        self.coord_repr = coord_representor(num_species_atom=self.num_species_atom, pos_splits=pos_splits_list)
        self.pos_splits, self.pos_units, self.pos_prec = self.coord_repr.pos_splits, self.coord_repr.pos_units, self.coord_repr.pos_prec
        self.decoding_fn = decoding_fn
        return
    
    @property
    def num_var(self):
        return self.coord_repr.num_var
    
    def decode(self, pos_vec):
        pos_prec_cum = np.concatenate([[0], np.cumsum(self.pos_prec)])
        pos_vec_list = [pos_vec[pos_prec_cum[i]: pos_prec_cum[i + 1]] for i in range(len(self.elements))]
        frac_coords_list = []
        for i in range(len(self.elements)):
            vec, splits, num_atom, units = pos_vec_list[i], self.pos_splits[i], self.num_species_atom[i], self.pos_units[i]
            decoded_frac_coords = self.coord_repr.decode_coords(vec, splits, units, self.decoding_fn, num_atom)
            frac_coords_list.append(decoded_frac_coords)
        return np.vstack(frac_coords_list), frac_coords_list


# always use the first several ind sites and then add fixed sites;
# therefore no ind site is skipped when current WP is special
def decode_ind_sites(decoded_frac_coords, ordered_wpc_list_element):
    decoder = StandardSPGDecoder()
    used_atom_idx = 0
    for wp in ordered_wpc_list_element:
        if wp[0] == 's':
            site = wp[1]
            decoder.append_fixed_atom(site)
        elif wp[0] == 'f':
            wp_form, wp_f = wp[1], wp[2]
            atom = decoded_frac_coords[used_atom_idx]
            decoder.decode_free_atom(wp_formula=wp_f, wp_formula_f=wp_form, atom=atom)
            used_atom_idx += 1  # discard unused atoms in decoded_frac_coords
        else:
            raise NotImplementedError
    atom_list, atom_form, decoded_atom_list = \
        decoder.atom_list, decoder.atom_form, decoder.decoded_atom_list
    return atom_list, atom_form, decoded_atom_list


def perturb_overlapped_atoms(atom_list, atom_form, decoded_atom_list, species_list, elements, use_pbc):
    # elements, num_species_atom = src.struct_utils.unique_elements(species_list)
    atom_list_0 = atom_list.copy()
    # try:
    decoder = StandardSPGDecoder()
    atom_pd = decoder.build_atom_pd(atom_list, atom_form, decoded_atom_list, species_list, use_pbc=use_pbc)
    fixed_test = np.unique(atom_form)
    if not ((len(fixed_test) == 1) and (fixed_test[0] == 'fix')):  # only containing fixed sites
        atom_list = decoder.shift_repeated_atoms(elements, atom_pd, use_pbc=use_pbc)
    # except Exception:
    #     log_printer.error("Perturbation failed. Return the original atom list")
    #     atom_list = atom_list_0
    #     from crysim.src import PROJECT_ROOT
    #     from pathlib import Path
    #     import re
    #     import pickle
    #     pattern = re.compile(r"debug-struct-(\d+).pkl")
    #     highest_num = -1
    #     for file_path in Path(PROJECT_ROOT).iterdir():
    #         if file_path.is_file():
    #             match = pattern.search(file_path.name)
    #             if match:
    #                 num = int(match.group(1))
    #                 if num > highest_num:
    #                     highest_num = num
    #     next_num = highest_num + 1
    #     with open(f"{PROJECT_ROOT}/debug-struct-{next_num}.pkl", 'wb') as f:
    #         pickle.dump((atom_list, atom_form, decoded_atom_list, species_list, elements, use_pbc), f)
    return atom_list


class PosConcatIndDecoder:
    def __init__(self, coord_representor, species_list, pos_splits_list, decoding_fn, **kwargs):
        self.species_list = species_list
        self.elements, self.num_species_atom = unique_elements(species_list=species_list)
        assert len(self.elements) == len(pos_splits_list)
        self.coord_repr = coord_representor(num_species_atom=self.num_species_atom, pos_splits=pos_splits_list)
        self.pos_splits, self.pos_units, self.pos_prec = self.coord_repr.pos_splits, self.coord_repr.pos_units, self.coord_repr.pos_prec
        self.modify_wp_formula = kwargs['modify_wp_formula']
        self.decoding_fn = decoding_fn
        return
    
    @property
    def num_var(self):
        return self.coord_repr.num_var
    
    def decode(self, pos_vec, wpc_lists):
        usable_wpc_list, ind_sites_num_list = [], []
        for wpc in wpc_lists:
            wpc_element, ind_num = analyze_wpc_list_element(wpc, modify=self.modify_wp_formula)
            usable_wpc_list.append(wpc_element)
            ind_sites_num_list.append(ind_num)
        
        # 1. split pos vec into elements
        pos_prec_cum = np.concatenate([[0], np.cumsum(self.pos_prec)])
        pos_vec_list = [pos_vec[pos_prec_cum[i]: pos_prec_cum[i + 1]] for i in range(len(self.elements))]
        
        # 2. fill in WP variables for each element
        used_atoms = []
        species_list, atom_list, atom_form, decoded_atom_list = [], [], [], []
        for i, e in enumerate(self.elements):
            # decode ind coords
            vec, splits, num_atom, units = pos_vec_list[i], self.pos_splits[i], self.num_species_atom[i], self.pos_units[i]
            decoded_frac_coords = self.coord_repr.decode_coords(vec, splits, units, self.decoding_fn, num_atom)

            # fill in WP variables
            usable_wpc_list_element = usable_wpc_list[i]
            ind_sites_num = ind_sites_num_list[i]
            used_atoms.append(decoded_frac_coords[:ind_sites_num])
            
            atom_list_ele, atom_form_ele, decoded_atom_list_ele = \
                decode_ind_sites(decoded_frac_coords, ordered_wpc_list_element=usable_wpc_list_element)
            
            atom_list += atom_list_ele
            atom_form += atom_form_ele
            decoded_atom_list += decoded_atom_list_ele
            species_list += [e] * len(atom_list_ele)
            
        assert species_list == self.species_list
        atom_list = perturb_overlapped_atoms(atom_list, atom_form, decoded_atom_list, species_list, self.elements, use_pbc=True)
        return atom_list, used_atoms


from crysim.src.registry import OptimizerName
from crysim.src.crysvec.utils import crys_embbeder
from functools import partial
oh_name = OptimizerName.onehot()[0]
crys_embbeder['coords'][f'{oh_name}-basic'] = {
    'enc': partial(PosConcatEncoder, coord_representor=PosMatDiscretizer), 
    'dec': partial(PosConcatAllDecoder, coord_representor=PosMatDiscretizer)
}
crys_embbeder['coords'][f'{oh_name}-basic-site'] = {
    'enc': partial(PosSiteEncoder, coord_representor=PosSiteDiscretizer), 
    'dec': partial(PosConcatAllDecoder, coord_representor=PosSiteDiscretizer)
}
crys_embbeder['coords'][f'fc'] = {
    'enc': partial(PosSiteEncoder, coord_representor=PosFrac), 
    'dec': partial(PosConcatAllDecoder, coord_representor=PosFrac)
}

crys_embbeder['coords'][f'{oh_name}-spg'] = {
    'enc': partial(PosConcatEncoder, coord_representor=PosMatDiscretizer), 
    'dec': partial(PosConcatIndDecoder, coord_representor=PosMatDiscretizer)
}
crys_embbeder['coords'][f'{oh_name}-spg-site'] = {
    'enc': partial(PosSiteEncoder, coord_representor=PosSiteDiscretizer), 
    'dec': partial(PosConcatIndDecoder, coord_representor=PosSiteDiscretizer)
}
crys_embbeder['coords'][f'fc-spg'] = {
    'enc': partial(PosSiteEncoder, coord_representor=PosFrac), 
    'dec': partial(PosConcatIndDecoder, coord_representor=PosFrac)
}
