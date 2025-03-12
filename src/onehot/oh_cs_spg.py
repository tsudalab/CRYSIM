import numpy as np
from src.onehot.oh_cs import OneHotVecProcessor
from src.onehot.oh_spg import OneHotProcessorSPG, encode_wp, output_spg_and_wp
from src.struct_utils import SPG, crystal_system_list, get_spg_for_cs, get_bravais_lattice, \
    feat2oh_map, oh2feat_map


def encode_cs_based_on_spg(crys_sys, cs_list):
    # this is a new cs list excluding empty cs, in which none spg can be fulfilled,
    # generated by calculate_cs_list.
    return feat2oh_map(cs_list)[crys_sys]


def decode_cs_based_on_spg(crys_sys_bit, cs_list):
    return cs_list[crys_sys_bit]


def encode_spg_for_cs(spg, cs, spg_list, spg_precision):
    spg_list_for_cs = get_spg_for_cs(cs, spg_list)
    spg_oh_for_cs = feat2oh_map(spg_list_for_cs)[spg]
    return int(np.around(spg_precision * spg_oh_for_cs / len(spg_list_for_cs)))


def decode_spg_for_cs(spg_oh, cs, spg_list, spg_precision):
    spg_list_for_cs = get_spg_for_cs(cs, spg_list)
    spg_oh_for_cs = int(np.around(spg_oh / spg_precision * len(spg_list_for_cs)))
    try:
        out_spg = oh2feat_map(spg_list_for_cs)[spg_oh_for_cs]
    except KeyError:
        if spg_oh_for_cs >= len(spg_list_for_cs):
            out_spg = oh2feat_map(spg_list_for_cs)[len(spg_list_for_cs) - 1]
        else:
            raise Exception("Unexpected spg")
    return out_spg


def calculate_spg_precision(spg_list):
    return np.max([len(get_spg_for_cs(cs, spg_list)) for cs in crystal_system_list])


class OneHotVecProcessorCSSPG(OneHotVecProcessor):
    def __init__(self, cs_precision, spg_precision, wp_precision, wycokffs_dict,
                 wyckoffs_max, spg_list, cs_list, default_spg, default_wp, logger):
        super().__init__(num_var=cs_precision + spg_precision + wp_precision, logger=logger)
        self.cs_precision = cs_precision
        self.spg_precision = spg_precision
        self.wp_precision = wp_precision
        self.wyckoffs_dict = wycokffs_dict
        self.wyckoffs_max = wyckoffs_max
        self.default_spg = default_spg
        self.default_wp = default_wp
        self.default_cs = None if default_spg is None else get_bravais_lattice(default_spg)
        self.spg_list = spg_list
        self.cs_list = cs_list
        return

    def _init_symmetry_container(self):
        cs_vec = np.zeros(shape=(self.cs_precision,))
        spg_vec = np.zeros(shape=(self.spg_precision,))
        wp_vec = np.zeros(shape=(self.wp_precision,))
        return cs_vec, spg_vec, wp_vec

    def _detect_symmetry(self, struct):
        _, spg_n = SPG.struct_spg_estimate(struct)
        cs = SPG.struct_bravais_estimate(struct, spg_n)
        return cs, spg_n

    def _derive_symmetry(self, struct):
        # print(f"Dealing with {len(struct)} given information")
        if len(struct) == 3:
            struct, spg_num, wp = struct
            # cs = SPG.struct_bravais_estimate(struct, spg_num)
            cs = get_bravais_lattice(spg_num)
        elif len(struct) == 4:
            struct, cs, spg_num, wp = struct
            # _, spg_num = SPG.struct_spg_estimate(struct)
        elif len(struct) == 5:
            struct, cs, spg_num, wp, _ = struct
        else:
            raise NotImplementedError
        return cs, spg_num, wp

    def _symmetry_encode(self, struct_symmetry):
        cs_vec, spg_vec, wp_vec = self._init_symmetry_container()
        if not len(struct_symmetry) == 3:
            print('Did not receive Wyckoff position number for encoding, randomly select one')
            cs, spg = struct_symmetry
            wp_oh = np.random.choice(list(range(self.wp_precision)))
        else:
            cs, spg, wp = struct_symmetry
            wp_oh = encode_wp(self.wp_precision, wp, self.wyckoffs_max)
        cs_oh = encode_cs_based_on_spg(cs, cs_list=self.cs_list)
        cs_vec[cs_oh] = 1
        spg_oh = encode_spg_for_cs(spg, cs, self.spg_list, self.spg_precision)
        spg_vec[spg_oh] = 1
        wp_vec[wp_oh] = 1
        return [cs_vec, spg_vec, wp_vec]

    def symmetry_encode(self, struct):
        return self._symmetry_encode(self.detect_symmetry(struct))

    def symmetry_decode(self, cs_spg_wp_vec, get_list, to_decode_wp):
        cs_vec, spg_vec, wp_vec = cs_spg_wp_vec[:self.cs_precision], \
            cs_spg_wp_vec[self.cs_precision: self.cs_precision + self.spg_precision], \
            cs_spg_wp_vec[-self.wp_precision:]
        try:
            cs, spg, wp = np.where(cs_vec)[0][0], \
                np.where(spg_vec)[0][0], np.where(wp_vec)[0][0]  # NOTE: choose the first ones
            cs = decode_cs_based_on_spg(cs, cs_list=self.cs_list)
            spg = decode_spg_for_cs(spg, cs, self.spg_list, self.spg_precision)
        except IndexError:
            print('Use default space group & Wyckoff position')
            spg, wp = self.default_spg, self.default_wp
            wp = encode_wp(self.wp_precision, wp, self.wyckoffs_max)
        # spg = shift_spg(spg, wyckoffs_dict=self.wyckoffs_dict, logger=self.logger)
        return output_spg_and_wp(sym_processor=self, spg=spg, wp=wp, get_list=get_list, to_decode_wp=to_decode_wp)

    def symmetry_and_cs_decode(self, sym_vec, get_list, to_decode_wp):
        spg, wp = self.symmetry_decode(sym_vec, get_list, to_decode_wp)
        cs = get_bravais_lattice(spg)
        return cs, spg, wp


class OneHotProcessorCSSPG(OneHotProcessorSPG):
    def __init__(self, species_list, splits, lat_ub, default_spg, default_wp,
                 wyckoffs_dict, wyckoffs_max, spg_list, cs_list, logger):
        self.cs_list = cs_list
        super().__init__(species_list, splits, lat_ub, default_spg, default_wp,
                         wyckoffs_dict, wyckoffs_max, spg_list, logger)
        return

    def _load_symmetry(self):
        self.spg_precision = calculate_spg_precision(self.spg_list)
        self.crys_sys_precision = len(self.cs_list)
        self.wp_precision = 300
        self.num_var = self.lattice_precision + \
                       self.crys_sys_precision + \
                       self.spg_precision + self.wp_precision + \
                       self.num_elements * self.space_precision
        self.spg_processor = OneHotVecProcessorCSSPG(cs_precision=self.crys_sys_precision,
                                                     spg_precision=self.spg_precision,
                                                     wp_precision=self.wp_precision,
                                                     wycokffs_dict=self.wyckoffs_dict,
                                                     wyckoffs_max=self.wyckoffs_max,
                                                     spg_list=self.spg_list,
                                                     cs_list=self.cs_list,
                                                     default_spg=self.default_spg,
                                                     default_wp=self.default_wp,
                                                     logger=self.logger)
        return

    def cs_spg_wp_decode(self, struct_vec, get_list=False):
        sym_vec = self.fetch_symmetry_vec_from_struct_vec(struct_vec)
        cs, spg, wp = self.spg_processor.symmetry_and_cs_decode(sym_vec, get_list, to_decode_wp=True)
        return cs, spg, wp


class OneHotCSSPGConstraintsBuilder:
    def __init__(self, cs_precision, spg_precision, wp_precision):
        self.cs_precision = cs_precision
        self.spg_precision = spg_precision
        self.wp_precision = wp_precision
        return

    def cs_spg_wp_constraints(self):
        cs_spg_wp_mat = np.zeros((3, self.cs_precision + self.spg_precision + self.wp_precision))
        cs_spg_wp_mat[0, 0: self.cs_precision] = 1
        cs_spg_wp_mat[1, self.cs_precision: self.cs_precision + self.spg_precision] = 1
        cs_spg_wp_mat[2, -self.wp_precision:] = 1
        return [cs_spg_wp_mat, [1] * 3, [1] * 3]

    def __call__(self, *args, **kwargs):
        return {'linear': self.cs_spg_wp_constraints(), 'quadratic': None}
