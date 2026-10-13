import numpy as np
from pymatgen.core import Structure
from logzero import logger as log_printer

from crysim.src.crysvec.lat import *
from crysim.src.crysvec.sym_oh import *
from crysim.src.crysvec.pos import *
from crysim.src.registry import coords_emb, sym_emb, enc_obj
from crysim.src.spg.spg_utils import sg_list, convert_pyx_wpc_to_str
from crysim.src.spg.get_wyckoff_position import get_all_wyckoff_combination
from crysim.src.struct_utils import unique_elements, default_ang_lb, default_ang_ub, get_crystal_lattice
try:
    from d_crysym.src.crysvec.sym_repr import *
except ImportError:
    pass


emb_type_dict = {
    'latdis+oh-csspg+struct': {'lat': crys_embbeder['lat'].get('oh-cs'), 'sym': crys_embbeder['sym'].get('oh-csspg'), 'coords': crys_embbeder['coords'].get('oh-spg')},
    'sitedis+oh-csspg+used_atoms': {'lat': crys_embbeder['lat'].get('oh-cs'), 'sym': crys_embbeder['sym'].get('oh-csspg'), 'coords': crys_embbeder['coords'].get('oh-spg-site')},
    'sitecont+oh-csspg+used_atoms': {'lat': crys_embbeder['lat'].get('lp-cs'), 'sym': crys_embbeder['sym'].get('oh-csspg'), 'coords': crys_embbeder['coords'].get('fc-spg')},
    
    'latdis+oh-spg+struct': {'lat': crys_embbeder['lat'].get('oh-cs'), 'sym': crys_embbeder['sym'].get('oh-spg'), 'coords': crys_embbeder['coords'].get('oh-spg')},
    'sitedis+oh-spg+used_atoms': {'lat': crys_embbeder['lat'].get('oh-cs'), 'sym': crys_embbeder['sym'].get('oh-spg'), 'coords': crys_embbeder['coords'].get('oh-spg-site')},
    'sitecont+oh-spg+used_atoms': {'lat': crys_embbeder['lat'].get('lp-cs'), 'sym': crys_embbeder['sym'].get('oh-spg'), 'coords': crys_embbeder['coords'].get('fc-spg')},
    
    'latdis+oh-repr+struct': {'lat': crys_embbeder['lat'].get('oh-cs'), 'sym': crys_embbeder['sym'].get('oh-repr'), 'coords': crys_embbeder['coords'].get('oh-spg')},
    'sitedis+oh-repr+used_atoms': {'lat': crys_embbeder['lat'].get('oh-cs'), 'sym': crys_embbeder['sym'].get('oh-repr'), 'coords': crys_embbeder['coords'].get('oh-spg-site')},
    'sitecont+oh-repr+used_atoms': {'lat': crys_embbeder['lat'].get('lp-cs'), 'sym': crys_embbeder['sym'].get('oh-repr'), 'coords': crys_embbeder['coords'].get('fc-spg')},
}


default_ang_lbs = np.array([default_ang_lb, default_ang_lb, default_ang_lb])
default_ang_ubs = np.array([default_ang_ub, default_ang_ub, default_ang_ub])
default_latlen_resol_ratio = 10
default_latang_resol = np.array([40, 40, 40]).astype(int)
default_wpc_prec = 300


def sym_decode_oh(sym_vec, sym_emb, wyckoffs_dict, required_decode_num):
    spg, wpc0 = sym_emb.decode(sym_vec)
    is_count_wpc_idx = False if len(spg) > 1 and len(wpc0) > 1 else True
    if is_count_wpc_idx:
        for s, w in zip(spg, wpc0):
            wpc_list_spg = wyckoffs_dict[s]
            num_wpcs = len(wpc_list_spg)
            decode_num = int(min(required_decode_num, num_wpcs))
            spg_list, loss_list, ids = [s] * decode_num, [0] * decode_num, list(range(decode_num))
            wpc_idx_list, wpc_list_list = [], []
            for i in range(decode_num):
                wpc = w + i
                if wpc >= num_wpcs:
                    wpc = wpc - num_wpcs
                wpc_idx_list.append(wpc)
                wpc_list_list.append(wpc_list_spg[wpc])
    else:
        decoded_num = len(wpc0)
        ids = list(range(decoded_num))
        if decoded_num > required_decode_num:
            select_id = np.random.choice(range(0, decoded_num), required_decode_num, replace=False)
            # select_id = np.arange(0, self.decode_num)  # solutions have been sorted with values from small to large
            spg_list = [spg[i] for i in select_id]
            wpc_idx_list = [wpc0[i] for i in select_id]
            wpc_list_list = [wyckoffs_dict[spg[i]][wpc0[i]] for i in select_id]
            ids = [ids[i] for i in select_id]
        else:
            spg_list = spg
            wpc_idx_list = wpc0
            wpc_list_list = [wyckoffs_dict[spg[i]][wpc0[i]] for i in ids]
        loss_list = [0] * int(min(required_decode_num, decoded_num))
    return spg_list, wpc_idx_list, wpc_list_list, loss_list, ids


class MatProcessor:
    """
    support binary encoding & continuous encoding (sitecont, crystal parameters themselves without processing)
    """
    def __init__(self, emb_type, dim, species_list, splits, 
                 len_lbs, len_ubs, ang_lbs, ang_ubs, len_splits, ang_splits,
                 decoding_fn, decode_num, wpc_prec, include_spg_list):
        self.emb_type = emb_type
        self.coords_type, self.sym_type, self.enc_obj = emb_type.split('+')
        
        self.dim = dim
        self.species_list = species_list
        self.elements, self.num_species_atom = unique_elements(species_list)
        
        self.splits, self.len_lbs, self.len_ubs = np.array(splits), np.array(len_lbs), np.array(len_ubs)
        self.ang_lbs = np.array([ang_lbs[0]]) if dim == 2 and len(ang_lbs) == 3 else np.array(ang_lbs)
        self.ang_ubs = np.array([ang_ubs[0]]) if dim == 2 and len(ang_ubs) == 3 else np.array(ang_ubs)
        if len_splits is None:
            if self.splits.ndim == 2:
                self.len_splits = default_latlen_resol_ratio * np.max(self.splits, axis=0)
            else:
                self.len_splits = default_latlen_resol_ratio * self.splits
        else:
            self.len_splits = len_splits
        if ang_splits is None:
            self.ang_splits = np.array([default_latang_resol[0]]) if dim == 2 else default_latang_resol
        else:
            self.ang_splits = np.array([ang_splits]) if dim == 2 else ang_splits

        self.decoding_vec_fn, self.decoding_mat_fn = decoding_fn
        self.decode_num = decode_num
        self.wpc_prec = default_wpc_prec if wpc_prec is None else wpc_prec
        self.include_spg_list = include_spg_list
        
        self._load_emb_type_dict()
        self._load_lat_emb()
        self._load_sym_emb()
        self._load_pos_emb()
        self.load_precs()
        self.load_empty_results()
        return
    
    def _load_emb_type_dict(self):
        self.emb_type_dict = emb_type_dict
        return

    def _load_lat_emb(self):
        lat_enc, lat_dec = self.emb_type_dict[self.emb_type]['lat']['enc'], self.emb_type_dict[self.emb_type]['lat']['dec']
        self.lat_enc = lat_enc(dim=self.dim, len_ubs=self.len_ubs, len_lbs=self.len_lbs, len_splits=self.len_splits, 
                               ang_ubs=self.ang_ubs, ang_lbs=self.ang_lbs, ang_splits=self.ang_splits)
        self.lat_dec = lat_dec(dim=self.dim, len_ubs=self.len_ubs, len_lbs=self.len_lbs, len_splits=self.len_splits, 
                               ang_ubs=self.ang_ubs, ang_lbs=self.ang_lbs, ang_splits=self.ang_splits, 
                               decoding_fn=self.decoding_vec_fn, default_lat=(None, None))
        self.lat_len_prec, self.lat_ang_prec = self.lat_enc.lat_len_prec, self.lat_enc.lat_ang_prec
        return
    
    def _load_sym_emb(self):
        sym_emb = self.emb_type_dict[self.emb_type]['sym']
        enc_spg = sg_list(dim=self.dim) if self.include_spg_list is None else list(set(sg_list(dim=self.dim)).intersection(self.include_spg_list))
        self.wyckoffs_dict, _ = get_all_wyckoff_combination(sg_list=enc_spg, atom_num=self.num_species_atom, mode='pyx', dim=self.dim)
        if 'repr' in self.sym_type:
            from d_crysym.src.crysvec.sym_repr import get_comp_repr_dict
            self.spg_repr_dict, self.wp_repr_dict = get_comp_repr_dict(SPGReprIdx(self.dim)(), WPReprIdx(self.dim)(), 
                                                                       self.species_list, self.dim, include_spg_list=self.include_spg_list)
            self.sym_emb = sym_emb(dim=self.dim, decoding_fn=self.decoding_vec_fn, wpc_prec=self.wpc_prec, 
                                   wyckoffs_dict=self.wyckoffs_dict, spg_repr_dict=self.spg_repr_dict, default_wpc=None)
        else:
            self.sym_emb = sym_emb(dim=self.dim, decoding_fn=self.decoding_vec_fn, wpc_prec=self.wpc_prec, 
                                   wyckoffs_dict=self.wyckoffs_dict, default_spg=None, default_wpc=None)
        if 'cs' in self.sym_type:
            self.cs_prec = self.sym_emb.cs_prec
        if 'spg' in self.sym_type or 'repr' in self.sym_type:
            self.spg_prec, self.wpc_prec = self.sym_emb.spg_prec, self.sym_emb.wpc_prec
        return
    
    def _load_pos_emb(self):
        pos_enc, pos_dec = self.emb_type_dict[self.emb_type]['coords']['enc'], self.emb_type_dict[self.emb_type]['coords']['dec']
        self.pos_enc = pos_enc(species_list=self.species_list, pos_splits_list=self.splits)
        self.pos_dec = pos_dec(species_list=self.species_list, pos_splits_list=self.splits, 
                               decoding_fn=self.decoding_mat_fn, modify_wp_formula=False)
        self.coords_prec = self.pos_enc.pos_prec
        return
    
    def load_precs(self):
        self.lat_prec, self.pos_prec = self.lat_enc.num_var, self.pos_enc.num_var
        self.sym_prec = self.sym_emb.num_var
        self.num_var = self.lat_prec + self.sym_prec + self.pos_prec
        return
    
    def load_empty_results(self):
        self.cs, self.spg, self.wpc_idx, self.wpc, self.used_atoms, self.ids = None, None, None, None, None, None
        return

    def symmetry_decode(self, struct_vec):
        sym_vec = self.get_sym_vec(struct_vec)
        # oh is essentially for qubo, which only generally gives one solution instead of batch solving
        return sym_decode_oh(sym_vec, self.sym_emb, self.wyckoffs_dict, required_decode_num=self.decode_num)
    
    def get_lat_vec(self, struct_vec):
        return struct_vec[..., :self.lat_prec]
    
    def get_sym_vec(self, struct_vec):
        return struct_vec[..., self.lat_prec: self.lat_prec + self.sym_prec]
    
    def get_pos_vec(self, struct_vec):
        return struct_vec[..., -self.pos_prec:]
    
    def encode(self, struct):
        struct, used_atoms, spg, wp_gen = struct
        if self.enc_obj == 'used_atoms':
            struct_vec = self.lat_enc.encode(struct) + self.sym_emb.encode([spg, wp_gen]) + self.pos_enc.encode(used_atoms, self.wyckoffs_dict[spg][wp_gen])
        else:
            struct_vec = self.lat_enc.encode(struct) + self.sym_emb.encode([spg, wp_gen]) + self.pos_enc.encode(struct)
        return np.concatenate(struct_vec)
    
    def decode(self, struct_vec):
        spg_list, wpc_idx_list, wpc_list_list, loss_list, ids = self.symmetry_decode(struct_vec)
        struct_vec = [struct_vec] * len(spg_list) if (struct_vec.ndim == 1 or len(struct_vec) == 1) else struct_vec
        struct_list, used_atoms_list = [], []
        for i in range(len(ids)):
            spg, wpc_list, solved_value, idx = spg_list[i], wpc_list_list[i], loss_list[i], ids[i]
            log_printer.info(f"Generating {i}-th struct with {spg} SG, {wpc_list} WPC with {solved_value} MAE")
            wpc_list = convert_pyx_wpc_to_str(spg=spg, wpc=wpc_list, dim=self.dim)
            lat = self.lat_dec.decode(self.get_lat_vec(struct_vec[idx]), crys_sys=get_crystal_lattice(spg, self.dim))
            pos, used_atoms = self.pos_dec.decode(self.get_pos_vec(struct_vec[idx]), wpc_lists=wpc_list)
            pmg_struct = Structure(lattice=lat, species=self.species_list, coords=pos, coords_are_cartesian=False)
            struct_list.append(pmg_struct)
            used_atoms_list.append(used_atoms)
        self.spg, self.wpc_idx, self.wpc, self.used_atoms, self.ids = spg_list, wpc_idx_list, wpc_list_list, used_atoms_list, ids
        self.cs = [get_crystal_lattice(spg, self.dim) for spg in spg_list]
        return struct_list

    def get_constraints(self):
        constr = Constraints(lat_prec=self.lat_enc.num_var, sym_prec=self.sym_emb.num_var, coords_prec=self.pos_enc.num_var)
        lat_num = constr.lattice_number_constraints(self.len_splits, self.ang_splits)
        if self.sym_type == 'oh-csspg':
            sym_num = constr.csspg_constraints(self.sym_emb.cs_prec, self.sym_emb.spg_prec, self.sym_emb.wpc_prec)
        elif self.sym_type == 'oh-spg':
            sym_num = constr.spg_constraints(self.sym_emb.spg_prec, self.sym_emb.wpc_prec)
        elif self.sym_type == 'oh-repr':
            sym_num = constr.oh_repr_constraints(self.sym_emb.spg_prec, self.sym_emb.wpc_prec)
        else:
            sym_num = [[], [], []]
        if self.coords_type == 'latdis':
            atom_num = constr.atom_number_constraints(self.num_species_atom, self.splits)
        elif self.coords_type == 'sitedis':
            atom_num = constr.site_number_constraints(self.num_species_atom, self.splits)
        else:
            raise NotImplementedError
        return [np.concatenate([lat_num[0], sym_num[0], atom_num[0]], axis=0), 
                lat_num[1] + sym_num[1] + atom_num[1], lat_num[2] + sym_num[2] + atom_num[2]]


class Constraints:
    def __init__(self, lat_prec, sym_prec, coords_prec):
        self.lat_prec, self.sym_prec, self.coords_prec = lat_prec, sym_prec, coords_prec
        self.linear_constr_len = lat_prec + sym_prec + coords_prec
        return
    
    def init(self, num_constr):
        return np.zeros(shape=(num_constr, self.linear_constr_len))
    
    def lattice_number_constraints(self, len_splits, ang_splits):
        assert self.lat_prec == sum(len_splits) + sum(ang_splits)
        num_constr = len(len_splits) + len(ang_splits)
        lat_mat = self.init(num_constr)
        lat_splits = np.concatenate([[0], np.cumsum(np.concatenate([len_splits, ang_splits]))])
        for i in range(num_constr):
            lat_mat[i, lat_splits[i]: lat_splits[i + 1]] = 1
        return [lat_mat, [1] * num_constr, [1] * num_constr]
    
    def atom_number_constraints(self, num_species_atom, pos_splits):
        assert len(num_species_atom) == len(pos_splits)
        coords_prec_list = [np.prod(pos_split) for pos_split in pos_splits]
        assert self.coords_prec == np.sum(coords_prec_list)
        num_constr = len(num_species_atom)
        atom_num_mat = self.init(num_constr)
        atom_num_splits = np.concatenate([[0], np.cumsum(coords_prec_list)])[::-1]
        for i in range(num_constr - 1):
            atom_num_mat[i, -atom_num_splits[i]: -atom_num_splits[i + 1]] = 1
        atom_num_mat[num_constr - 1, -atom_num_splits[num_constr - 1]:] = 1
        return [atom_num_mat, num_species_atom, [1] * num_constr]
    
    def site_number_constraints(self, num_species_atom, pos_splits):
        assert len(num_species_atom) == len(pos_splits)
        coords_prec_list = []
        for p, n in zip(pos_splits, num_species_atom):
            coords_prec_list += list(p) * n
        assert self.coords_prec == np.sum(coords_prec_list)
        num_constr = len(coords_prec_list)
        site_num_mat = self.init(num_constr)
        site_num_splits = np.concatenate([[0], np.cumsum(coords_prec_list)])[::-1]
        for i in range(num_constr - 1):
            site_num_mat[i, -site_num_splits[i]: -site_num_splits[i + 1]] = 1
        site_num_mat[num_constr - 1, -site_num_splits[num_constr - 1]:] = 1
        return [site_num_mat, [1] * num_constr, [1] * num_constr]
    
    def crys_sys_constraints(self, crys_sys_prec):
        assert self.sym_prec == crys_sys_prec
        num_constr = 1
        crys_sys_mat = self.init(num_constr)
        crys_sys_mat[0, self.lat_prec: self.lat_prec + self.sym_prec] = 1
        return [crys_sys_mat, [1] * num_constr, [1] * num_constr]
    
    def spg_constraints(self, spg_prec, wpc_prec):
        assert self.sym_prec == spg_prec + wpc_prec
        num_constr = 2
        spg_mat = self.init(num_constr)
        spg_mat[0, self.lat_prec: self.lat_prec + spg_prec] = 1
        spg_mat[1, self.lat_prec + spg_prec: self.lat_prec + self.sym_prec] = 1
        return [spg_mat, [1] * num_constr, [1] * num_constr]
    
    def csspg_constraints(self, crys_sys_prec, spg_prec, wp_prec):
        assert self.sym_prec == crys_sys_prec + spg_prec + wp_prec
        num_constr = 3
        csspg_mat = self.init(num_constr)
        csspg_mat[0, self.lat_prec: self.lat_prec + crys_sys_prec] = 1
        csspg_mat[1, self.lat_prec + crys_sys_prec: self.lat_prec + crys_sys_prec + spg_prec] = 1
        csspg_mat[2, self.lat_prec + crys_sys_prec + spg_prec: self.lat_prec + self.sym_prec] = 1
        return [csspg_mat, [1] * num_constr, [1] * num_constr]
    
    def oh_repr_constraints(self, spg_prec, wpc_prec):
        assert self.sym_prec == spg_prec + wpc_prec
        num_constr = 1
        spg_mat = self.init(num_constr)
        spg_mat[0, self.lat_prec + spg_prec: self.lat_prec + self.sym_prec] = 1
        return [spg_mat, [1] * num_constr, [1] * num_constr]
