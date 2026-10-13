from functools import partial
import numpy as np
from pymatgen.core import Structure
from crysim.src.struct_utils import get_crystal_lattice, get_cs_list, get_spg_for_cs, feat2oh_map, oh2feat_map, SPG, crystal_system_list
from crysim.src.spg.spg_utils import get_spg_list


def encode_cs(crys_sys, crys_sys_list):
    return feat2oh_map(crys_sys_list)[crys_sys]


def decode_cs(crys_sys_bit, crys_sys_list):
    return np.array(crys_sys_list)[crys_sys_bit]


def encode_spg(spg, sg_list):
    return feat2oh_map(sg_list)[spg]


def decode_spg(spg_oh, sg_list):
    return np.array(sg_list)[spg_oh]


def encode_wp_for_spg(wp, spg, wyckoffs_dict, wp_precision):
    wp_list_for_spg = wyckoffs_dict[spg]
    wpc_bit = int(np.around(wp_precision * wp / len(wp_list_for_spg)))
    if wpc_bit == wp_precision:
        wpc_bit -= 1
    return wpc_bit


def decode_wpc_for_spg(wp_oh, spg, wyckoffs_dict, wp_precision):
    if not hasattr(spg, "__len__") and not hasattr(wp_oh, "__len__"):
        spg, wp_oh = [spg], [wp_oh]
    wpc_idx = []
    for s, w in zip(spg, wp_oh):
        wp_list_for_spg = wyckoffs_dict[s]
        wpc_idx.append(int(np.around(w / wp_precision * len(wp_list_for_spg))))
    return np.array(wpc_idx)


def encode_spg_for_cs(spg, cs, spg_list, spg_precision, dim):
    spg_list_for_cs = get_spg_for_cs(cs, spg_list, dim=dim)
    spg_oh_for_cs = feat2oh_map(spg_list_for_cs)[spg]
    return int(np.around(spg_precision * spg_oh_for_cs / len(spg_list_for_cs)))


def decode_spg_for_cs(spg_oh, cs, spg_list, spg_precision, dim):
    if not hasattr(spg_oh, "__len__") and not hasattr(cs, "__len__"):
        spg_oh, cs = [spg_oh], [cs]
    out_spg = []
    for c, s in zip(cs, spg_oh):
        spg_list_for_cs = get_spg_for_cs(c, spg_list, dim=dim)
        spg_oh_for_cs = int(np.around(s / spg_precision * len(spg_list_for_cs)))
        try:
            out_spg.append(np.array(spg_list_for_cs)[spg_oh_for_cs])
        except KeyError:
            if spg_oh_for_cs >= len(spg_list_for_cs):
                out_spg.append(np.array(spg_list_for_cs)[len(spg_list_for_cs) - 1])
            else:
                raise Exception("Unexpected spg")
    return np.array(out_spg)


######### encoder & decoder #########


class SymVec:
    def __init__(self, dim, num_var, decoding_fn):
        self.dim = dim
        self.num_var = num_var
        self.decoding_fn = decoding_fn
        return

    def derive_symmetry(self, struct):
        raise NotImplementedError

    def encode(self, *args):
        raise NotImplementedError

    def _find_symmetry_bit(self, sym_vec):
        return self.decoding_fn(sym_vec)

    def decode(self, *args):
        raise NotImplementedError


class CrysSysVec(SymVec):
    def __init__(self, dim, decoding_fn, cs_list, default_crys_sys):
        # spg_list = get_spg_list2(num_species_atom=num_species_atom, dim=dim)
        # self.cs_list = get_cs_list(spg_list, dim=self.dim)
        self.cs_list = cs_list
        super().__init__(dim=dim, num_var=len(self.cs_list), decoding_fn=decoding_fn)
        self.default_crys_sys = default_crys_sys
        return
    
    def derive_symmetry(self, struct):
        if isinstance(struct, Structure):
            _, spg_num = SPG.struct_spg_estimate(struct)
            cs = SPG.struct_lattice_estimate(struct, spg_num)
        else:
            spg_num, _ = struct
            cs = get_crystal_lattice(spg_num, dim=self.dim)
        return cs

    def encode(self, struct):
        cs = self.derive_symmetry(struct)
        crys_sys_vec = np.zeros(shape=(self.num_var,))
        oh_bit = encode_cs(cs, crys_sys_list=self.cs_list)
        crys_sys_vec[oh_bit] = 1
        return [crys_sys_vec]

    def decode(self, sym_vec):
        # try:
        crys_sys_bit = self._find_symmetry_bit(sym_vec)
        crys_sys = decode_cs(crys_sys_bit, crys_sys_list=self.cs_list)
        # except IndexError:
        #     print('Use default crystal system')
        #     crys_sys = [self.default_crys_sys] * batch_size
        return crys_sys


class SPGOH(SymVec):
    def __init__(self, dim, decoding_fn, **kwargs):
        self.wyckoffs_dict = kwargs['wyckoffs_dict']
        self.spg_list = get_spg_list(self.wyckoffs_dict)
        self.spg_prec = len(self.spg_list)
        super().__init__(dim=dim, num_var=self.spg_prec, decoding_fn=decoding_fn)
        self.default_spg = kwargs['default_spg']
        return
    
    def derive_symmetry(self, struct):
        if isinstance(struct, Structure):
            _, spg_num = SPG.struct_spg_estimate(struct)
        else:
            spg_num, _ = struct
        return spg_num

    def encode(self, struct):
        spg_num = self.derive_symmetry(struct)
        spg_vec = np.zeros(shape=(self.spg_prec,))
        spg_oh = encode_spg(spg_num, self.spg_list)
        spg_vec[spg_oh] = 1
        return [spg_vec]

    def decode(self, spg_vec):
        # try:
        spg = self._find_symmetry_bit(spg_vec)
        spg = decode_spg(spg, self.spg_list)
        # except IndexError:
        #     print('Use default space group & Wyckoff position')
        #     spg = self.default_spg
        return spg


class WPCOH(SymVec):
    def __init__(self, dim, decoding_fn, **kwargs):
        self.wyckoffs_dict = kwargs['wyckoffs_dict']
        self.wpc_prec = min(kwargs['wpc_prec'], max(len(wpc) for wpc in self.wyckoffs_dict.values()))
        super().__init__(dim=dim, num_var=self.wpc_prec, decoding_fn=decoding_fn)
        self.default_wpc = kwargs['default_wpc']
        return
    
    def derive_symmetry(self, struct):
        if isinstance(struct, Structure):
            wpc = None
        else:
            _, wpc = struct
        return wpc

    def encode(self, spg_num, struct):
        wpc_idx = self.derive_symmetry(struct)
        wpc_vec = np.zeros(shape=(self.wpc_prec,))
        if wpc_idx is None:
            print('Did not receive Wyckoff position number for encoding, randomly select one')
            wpc_oh = np.random.choice(list(range(self.wpc_prec)))
        else:
            wpc_oh = encode_wp_for_spg(wpc_idx, spg_num, wyckoffs_dict=self.wyckoffs_dict, wp_precision=self.wpc_prec)
        wpc_vec[wpc_oh] = 1
        return [wpc_vec]

    def decode(self, spg_num, wpc_vec):
        # try:
        wpc = self._find_symmetry_bit(wpc_vec)
        wpc = decode_wpc_for_spg(wpc, spg_num, self.wyckoffs_dict, self.wpc_prec)
        # except IndexError:
        #     print('Use default space group & Wyckoff position')
        #     wpc = self.default_wpc
        return wpc


class SPGVec:
    def __init__(self, spg_repr, wpc_repr, dim, decoding_fn, **kwargs):
        # wpc_prec, wyckoffs_dict, default_spg, default_wpc
        self.spg_repr = spg_repr(dim=dim, decoding_fn=decoding_fn, **kwargs)
        self.wpc_repr = wpc_repr(dim=dim, decoding_fn=decoding_fn, **kwargs)
        self.spg_prec, self.wpc_prec = self.spg_repr.spg_prec, self.wpc_repr.wpc_prec
        self.dim = dim
        self.num_var = self.spg_prec + self.wpc_prec
        self.decoding_fn = decoding_fn
        return

    def encode(self, struct):
        spg_num = self.spg_repr.derive_symmetry(struct)
        spg_vec = self.spg_repr.encode(struct)
        wpc_vec = self.wpc_repr.encode(spg_num, struct)
        return spg_vec + wpc_vec

    def decode(self, spg_wpc_vec):
        spg_vec, wpc_vec = spg_wpc_vec[..., :self.spg_repr.spg_prec], spg_wpc_vec[..., -self.wpc_repr.wpc_prec:]
        spg_num = self.spg_repr.decode(spg_vec)
        wpc = self.wpc_repr.decode(spg_num, wpc_vec)
        if isinstance(wpc, tuple):  # feasible_spg, wpc, loss, ids
            return wpc
        return spg_num, wpc


def calculate_spg_precision(spg_list, dim):
    return np.max([len(get_spg_for_cs(cs, spg_list, dim=dim)) for cs in crystal_system_list(dim=dim)])


class CSSPGVec(SymVec):
    def __init__(self, dim, decoding_fn, wpc_prec, wyckoffs_dict, default_spg, default_wpc):
        self.wyckoffs_dict = wyckoffs_dict
        self.spg_list = get_spg_list(wyckoffs_dict)
        self.cs_list = get_cs_list(self.spg_list, dim=dim)
        self.cs_prec = len(self.cs_list)
        self.spg_prec = calculate_spg_precision(self.spg_list, dim=dim)
        self.wpc_prec = min(wpc_prec, max(len(wpc) for wpc in self.wyckoffs_dict.values()))
        super().__init__(dim=dim, num_var=self.cs_prec + self.spg_prec + self.wpc_prec, decoding_fn=decoding_fn)
        self.default_spg = default_spg
        self.default_wpc = default_wpc
        self.default_cs = None if default_spg is None else get_crystal_lattice(default_spg, dim=dim)
        return
    
    def derive_symmetry(self, struct):
        if isinstance(struct, Structure):
            _, spg_num = SPG.struct_spg_estimate(struct)
            cs = SPG.struct_lattice_estimate(struct, spg_num)
            wpc = None
        else:
            spg_num, wpc = struct
            cs = get_crystal_lattice(spg_num, dim=self.dim)
        return cs, spg_num, wpc

    def encode(self, struct):
        cs, spg_num, wpc = self.derive_symmetry(struct)
        cs_vec, spg_vec, wpc_vec = \
            np.zeros(shape=(self.cs_prec,)), np.zeros(shape=(self.spg_prec,)), np.zeros(shape=(self.wpc_prec,))
        if wpc is None:
            print('Did not receive Wyckoff position number for encoding, randomly select one')
            wpc_oh = np.random.choice(list(range(self.wpc_prec)))
        else:
            wpc_oh = encode_wp_for_spg(wpc, spg_num, self.wyckoffs_dict, self.wpc_prec)
        cs_oh = encode_cs(cs, crys_sys_list=self.cs_list)
        cs_vec[cs_oh] = 1
        spg_oh = encode_spg_for_cs(spg_num, cs, self.spg_list, self.spg_prec, dim=self.dim)
        spg_vec[spg_oh] = 1
        wpc_vec[wpc_oh] = 1
        return [cs_vec, spg_vec, wpc_vec]

    def decode(self, cs_spg_wpc_vec):
        cs_vec, spg_vec, wpc_vec = cs_spg_wpc_vec[..., :self.cs_prec], \
            cs_spg_wpc_vec[..., self.cs_prec: self.cs_prec + self.spg_prec], \
            cs_spg_wpc_vec[..., -self.wpc_prec:]
        # try:
        cs, spg, wpc = self._find_symmetry_bit(cs_vec), \
            self._find_symmetry_bit(spg_vec), self._find_symmetry_bit(wpc_vec)
        cs = decode_cs(cs, crys_sys_list=self.cs_list)
        spg = decode_spg_for_cs(spg, cs, self.spg_list, self.spg_prec, dim=self.dim)
        wpc = decode_wpc_for_spg(wpc, spg, self.wyckoffs_dict, self.wpc_prec)
        # except IndexError:
        #     print('Use default space group & Wyckoff position')
        #     spg, wpc = self.default_spg, self.default_wpc
        return spg, wpc


from crysim.src.registry import OptimizerName
from crysim.src.crysvec.utils import crys_embbeder
oh_name = OptimizerName.onehot()[0]
crys_embbeder['sym'][f'{oh_name}-cs'] = CrysSysVec
crys_embbeder['sym'][f'{oh_name}-spg'] = partial(SPGVec, spg_repr=SPGOH, wpc_repr=WPCOH)
crys_embbeder['sym'][f'{oh_name}-csspg'] = CSSPGVec
