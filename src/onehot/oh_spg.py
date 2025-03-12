import numpy as np
import copy
from pymatgen.core import Structure

from src.onehot.oh_basic import OneHotProcessor4Element
from src.onehot.oh_basic import _default_pos
from src.onehot.oh_cs import OneHotProcessorCrysSys, OneHotVecProcessor
from src.spg.spg_utils import WPElementProcess, StandardSPGDecoder, get_wp_list2
from src.struct_utils import get_bravais_lattice, perturb_frac_atom, SPG, feat2oh_map, oh2feat_map


def encode_wp(wp_precision, wp, wyckoffs_max):
    # return int(np.around(wp_precision * wp / wyckoffs_max))  # numerical error
    return int(np.floor(wp_precision * wp / wyckoffs_max))


def decode_wp(wp_precision, wp_oh, wyckoffs_max):
    return int(np.around(wyckoffs_max * wp_oh / wp_precision))


def encode_spg(spg, sg_list):
    try:
        return feat2oh_map(sg_list)[spg]
    except KeyError:
        left, right = 0, 0
        spg_l = spg
        spg_r = spg
        while spg_r not in sg_list:
            if spg_r == 231:
                spg_r = 1
            spg_r += 1
            right += 1
        while spg_l not in sg_list:
            if spg_l == 0:
                spg_l = 231
            spg_l -= 1
            left -= 1
        if right > -left:
            return feat2oh_map(sg_list)[spg_l]
        else:
            return feat2oh_map(sg_list)[spg_r]
    # return spg - 2  # [2, 231)


def decode_spg(spg_oh, sg_list):
    return oh2feat_map(sg_list)[spg_oh]
    # return spg_oh + 2


def output_spg_and_wp(sym_processor, spg, wp, get_list, to_decode_wp):
    if sym_processor.logger is not None:
        sym_processor.logger.record_anything1(operation='Decoded Space Group', result=spg)
        sym_processor.logger.record_anything2(operation='Decoded Wyckoff Positions', result=wp)
    if not get_list:
        if not to_decode_wp:
            wp_prop = wp
        else:
            decoded_wp = decode_wp(sym_processor.wp_precision, wp, sym_processor.wyckoffs_max)
            if sym_processor.logger is not None:
                sym_processor.logger.record_anything3(operation='Real Decoded Wyckoff Positions', result=decoded_wp)
            wp_prop = decoded_wp
    else:
        decoded_wp = decode_wp(sym_processor.wp_precision, wp, sym_processor.wyckoffs_max)
        if sym_processor.logger is not None:
            sym_processor.logger.record_anything3(operation='Real Decoded Wyckoff Positions', result=decoded_wp)
        wp_list = get_wp_list2(sym_processor.wyckoffs_dict, sym_processor.wyckoffs_max, spg, decoded_wp)
        wp_prop = wp_list
    return spg, wp_prop


class OneHotVecProcessorSPG(OneHotVecProcessor):
    def __init__(self, spg_precision, wp_precision, wycokffs_dict, wyckoffs_max, spg_list,
                 default_spg, default_wp, logger):
        super().__init__(num_var=spg_precision + wp_precision, logger=logger)
        self.spg_precision = spg_precision
        self.wp_precision = wp_precision
        self.wyckoffs_dict = wycokffs_dict
        self.wyckoffs_max = wyckoffs_max
        self.default_spg = default_spg
        self.default_wp = default_wp
        self.spg_list = spg_list
        return

    def _init_symmetry_container(self):
        spg_vec = np.zeros(shape=(self.spg_precision,))
        wp_vec = np.zeros(shape=(self.wp_precision,))
        return spg_vec, wp_vec

    def _detect_symmetry(self, struct):
        _, spg_num = SPG.struct_spg_estimate(struct)
        return spg_num

    def _derive_symmetry(self, struct):
        if len(struct) == 3:
            struct, spg_num, wp = struct
            # _, spg_num = SPG.struct_spg_estimate(struct)
        elif len(struct) == 4:
            struct, spg_num, wp, _ = struct
        else:
            raise NotImplementedError
        return spg_num, wp

    def _symmetry_encode(self, struct_symmetry):
        spg_vec, wp_vec = self._init_symmetry_container()
        if not isinstance(struct_symmetry, tuple):
            print('Did not receive Wyckoff position number for encoding, randomly select one')
            spg = struct_symmetry
            wp_oh = np.random.choice(list(range(self.wp_precision)))
        else:
            spg, wp = struct_symmetry
            wp_oh = encode_wp(self.wp_precision, wp, self.wyckoffs_max)
        spg_oh = encode_spg(spg, self.spg_list)
        spg_vec[spg_oh] = 1
        wp_vec[wp_oh] = 1
        return [spg_vec, wp_vec]

    def symmetry_encode(self, struct):
        return self._symmetry_encode(self.detect_symmetry(struct))

    def symmetry_decode(self, spg_wp_vec, get_list, to_decode_wp):
        spg_vec, wp_vec = spg_wp_vec[:self.spg_precision], spg_wp_vec[-self.wp_precision:]
        try:
            spg, wp = np.where(spg_vec)[0][0], np.where(wp_vec)[0][0]  # note: choose the first ones
            spg = decode_spg(spg, self.spg_list)
        except IndexError:
            print('Use default space group & Wyckoff position')
            spg, wp = self.default_spg, self.default_wp
            wp = encode_wp(self.wp_precision, wp, self.wyckoffs_max)
        # spg = shift_spg(spg, wyckoffs_dict=self.wyckoffs_dict, logger=self.logger)
        return output_spg_and_wp(sym_processor=self, spg=spg, wp=wp, get_list=get_list, to_decode_wp=to_decode_wp)


class OneHotProcessorSPG4Element(OneHotProcessor4Element):
    def __init__(self, splits, num_atom, wp_list_element, decoder_obj):
        super().__init__(splits=splits, num_atom=num_atom)
        self.wp_list_element = wp_list_element
        self.decoder_obj = decoder_obj
        return

    def _wp_process(self):
        self.wpe = WPElementProcess(self.wp_list_element)
        self.wpe.load_variables()
        self.wpe.load_formulas()
        return self.wpe.var, self.wpe.fix

    def _sample_from_sites(self, centers=None):
        centers = [0.25, 0.75] if centers is None else centers
        atom_vec = np.reshape(self.element_sites,
                              newshape=(self.splits_a, self.splits_b, self.splits_c))
        atoms = []
        center_ids = [int(len(atom_vec.flatten()) * center) for center in centers]
        left_or_right_flag = -1
        walk_steps = 1
        while len(atoms) < self.num_atom:
            for i, c in enumerate(center_ids):
                atom_bit = np.unravel_index(c, (self.splits_a, self.splits_b, self.splits_c))
                if (atom_vec[atom_bit] == 1) & (len(atoms) < self.num_atom):
                    atom = atom_bit
                    atom = (atom[0] * self.unit_a, atom[1] * self.unit_b, atom[2] * self.unit_c)
                    atoms.append(atom)
                else:
                    pass
                center_ids[i] += left_or_right_flag * walk_steps
            left_or_right_flag *= -1
            walk_steps += 1
        return atoms

    def decode_sites(self):
        wp_formula, wp_site = self._wp_process()
        wp_formula_form = self.wpe.var_form
        # num_avail_bits = len(np.where(self.element_sites.flatten())[0])
        # if num_avail_bits > self.num_atom:
        #     print("Sample around on specific centers")
        #     atoms = self._sample_from_sites()
        # else:
        atoms = self._decode_sites()
        wp_formula, wp_formula_form = copy.deepcopy(wp_formula), copy.deepcopy(wp_formula_form)

        for i in range(len(wp_formula)):  # note: choose the first ones
            wp_pairs, wp_pairs_form = wp_formula[i], wp_formula_form[i]
            try:
                atom = atoms[i]
            except IndexError:
                print('Use default position')
                atom = (_default_pos[0] + perturb_frac_atom(),
                        _default_pos[1] + perturb_frac_atom(),
                        _default_pos[2] + perturb_frac_atom())
            self.decoder_obj.decode_free_atom(wp_formula=wp_pairs,
                                              wp_formula_f=wp_pairs_form,
                                              atom=atom)  # discard unused atoms
        for site in wp_site:
            self.decoder_obj.append_fixed_atom(site)

        atom_list, atom_form, decoded_atom_list = \
            self.decoder_obj.atom_list, self.decoder_obj.atom_form, self.decoder_obj.decoded_atom_list
        self.decoder_obj.clear()
        return atom_list, atom_form, decoded_atom_list


class OneHotProcessorSPG(OneHotProcessorCrysSys):  # space group is based on lattice system
    def __init__(self, species_list, splits, lat_ub, default_spg, default_wp,
                 wyckoffs_dict, wyckoffs_max, spg_list, logger):
        default_crys_sys = None if default_spg is None else get_bravais_lattice(default_spg)
        self.decoder = StandardSPGDecoder()
        self.default_spg = default_spg
        self.default_wp = default_wp
        self.spg_list = spg_list
        self.wyckoffs_dict, self.wyckoffs_max = wyckoffs_dict, wyckoffs_max
        super().__init__(species_list=species_list, splits=splits, lat_ub=lat_ub,
                         default_crys_sys=default_crys_sys, logger=logger)
        return

    def _load_symmetry(self):
        self.spg_precision = len(self.spg_list)
        self.wp_precision = 300
        self.num_var = self.lattice_precision + \
                       self.spg_precision + self.wp_precision + \
                       self.num_elements * self.space_precision
        self.spg_processor = OneHotVecProcessorSPG(spg_precision=self.spg_precision,
                                                   wp_precision=self.wp_precision,
                                                   wycokffs_dict=self.wyckoffs_dict,
                                                   wyckoffs_max=self.wyckoffs_max,
                                                   spg_list=self.spg_list,
                                                   default_spg=self.default_spg,
                                                   default_wp=self.default_wp,
                                                   logger=self.logger)
        return

    def one_hot_encode(self, struct):
        struct[0] = self._one_hot_encode_basic(struct[0])
        struct_vec = []
        struct_vec += self._lat_encode(struct[0])
        struct_vec += self.spg_processor.symmetry_encode(struct)
        struct_vec += self._pos_encode(struct[0])
        return np.concatenate(struct_vec)

    def space_group_decode(self, struct_vec, get_list=False, to_decode_wp=True):
        sym_vec = self.fetch_symmetry_vec_from_struct_vec(struct_vec)
        spg, wp = self.spg_processor.symmetry_decode(sym_vec, get_list, to_decode_wp=to_decode_wp)
        return spg, wp

    def _crystal_system_decode(self, struct_vec):
        crys_sys = get_bravais_lattice(self.spg)
        return crys_sys

    def init_position_container_decode(self):
        ohp_ele = {}
        for i, ele in enumerate(self.elements):
            wp_list_element = self.wp_list[i]
            ohp_ele[ele] = OneHotProcessorSPG4Element(self.splits,
                                                      wp_list_element=wp_list_element,
                                                      decoder_obj=self.decoder,
                                                      num_atom=self.num_species_atom[i])
        return ohp_ele

    def _one_hot_decode_ele(self, struct_vec):
        atom_vec = np.split(struct_vec[-self.num_elements * self.space_precision:], self.num_elements)
        ohp_ele = self.init_position_container_decode()
        # the encoded vector does not contain element species info
        # therefore, it is necessary to obtain the info based on this class
        species_list = []
        atom_list, atom_form, decoded_atom_list = [], [], []
        for i, species in enumerate(self.elements):
            obj = ohp_ele[species]
            obj.set_sites(element_sites=atom_vec[i])
            atom = obj.decode_sites()
            atom_list += atom[0]
            atom_form += atom[1]
            decoded_atom_list += atom[2]
            species_list += [species] * len(atom[0])
        return (atom_list, atom_form, decoded_atom_list), species_list

    def one_hot_decode_build_struct(self, struct_vec, atom_list, atom_form,
                                    decoded_atom_list, species_list):
        lat = self._one_hot_decode_lat(struct_vec)
        atom_pd = self.decoder.build_atom_pd(atom_list, atom_form, decoded_atom_list, species_list)
        fixed_test = np.unique(atom_form)
        if not ((len(fixed_test) == 1) and (fixed_test[0] == 'fix')):  # only containing fixed sites
            atom_list = self.decoder.shift_repeated_atoms(self.elements, atom_pd)

        pmg_struct = Structure(lattice=lat, species=species_list, coords=atom_list,
                               coords_are_cartesian=False)
        return pmg_struct

    def one_hot_decode(self, struct_vec):
        self.spg, self.wp_list = self.space_group_decode(struct_vec, get_list=True)
        atoms, species_list = self._one_hot_decode_ele(struct_vec)
        atom_list, atom_form, decoded_atom_list = atoms
        pmg_struct = self.one_hot_decode_build_struct(struct_vec, atom_list, atom_form,
                                                      decoded_atom_list, species_list)
        return pmg_struct

    def decode_more_wp(self, struct_vec, decode_num):
        self.spg, wp0 = self.space_group_decode(struct_vec, get_list=False)
        num_wps = len(self.wyckoffs_dict[self.spg])
        wp0 = int(num_wps * wp0 / self.wyckoffs_max)
        struct_list = []
        decode_num = int(np.min([decode_num, num_wps]))
        for i in range(decode_num):
            wp = wp0 + i
            if wp >= num_wps:
                wp = wp - num_wps
            self.wp_list = self.wyckoffs_dict[self.spg][wp]
            atoms, species_list = self._one_hot_decode_ele(struct_vec)
            atom_list, atom_form, decoded_atom_list = atoms
            pmg_struct = self.one_hot_decode_build_struct(struct_vec, atom_list, atom_form,
                                                          decoded_atom_list, species_list)
            struct_list.append(pmg_struct)
        return struct_list


class OneHotSPGConstraintsBuilder:
    def __init__(self, spg_precision, wp_precision):
        self.spg_precision = spg_precision
        self.wp_precision = wp_precision
        return

    def spg_wp_constraints(self):
        spg_wp_mat = np.zeros((2, self.spg_precision + self.wp_precision))
        spg_wp_mat[0, 0:self.spg_precision] = 1
        spg_wp_mat[1, -self.wp_precision:] = 1
        return [spg_wp_mat, [1] * 2, [1] * 2]

    def __call__(self, *args, **kwargs):
        return {'linear': self.spg_wp_constraints(), 'quadratic': None}
