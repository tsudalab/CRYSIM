import numpy as np
from src.optimizer.rand_gen import default_ang_mean
from src.onehot.oh_basic import _num_lat_len, _num_lat_ang
from src.onehot.oh_basic import OneHotLatticeScaleProcessor
from src.struct_utils import crystal_system_list, crystal_system_dict, feat2oh_map, SPG

_num_crys_sys = len(crystal_system_list)
_default_crys_sys = np.random.choice(crystal_system_list)


def encode_cs(crys_sys):
    return feat2oh_map(crystal_system_list)[crys_sys]


def decode_cs(crys_sys_bit):
    return crystal_system_list[crys_sys_bit]


class OneHotVecProcessor:
    def __init__(self, num_var, logger):
        self.num_var = num_var
        self.logger = logger
        return

    def _init_symmetry_container(self):
        raise NotImplementedError

    def _symmetry_encode(self, struct_symmetry):
        raise NotImplementedError

    def _detect_symmetry(self, struct):
        raise NotImplementedError

    def _derive_symmetry(self, struct):
        raise NotImplementedError

    def detect_symmetry(self, struct):
        if not isinstance(struct, list):
            return self._detect_symmetry(struct)
        else:
            return self._derive_symmetry(struct)

    def symmetry_encode(self, struct):
        raise NotImplementedError

    def symmetry_decode(self, *args):
        raise NotImplementedError


class OneHotVecProcessorCrysSys(OneHotVecProcessor):
    def __init__(self, crys_sys_precision, default_crys_sys, logger):
        super().__init__(num_var=crys_sys_precision, logger=logger)
        self.crys_sys_precision = crys_sys_precision
        self.default_crys_sys = default_crys_sys
        return

    def _init_symmetry_container(self):
        crys_sys_vec = np.zeros(shape=(self.crys_sys_precision,))
        return crys_sys_vec

    def _symmetry_encode(self, struct_symmetry):
        crys_sys_vec = self._init_symmetry_container()
        oh_bit = encode_cs(struct_symmetry)
        crys_sys_vec[oh_bit] = 1
        return [crys_sys_vec]

    def _detect_symmetry(self, struct):
        _, spg_n = SPG.struct_spg_estimate(struct)
        return SPG.struct_bravais_estimate(struct, spg_n)

    def _derive_symmetry(self, struct):
        if len(struct) == 2:
            struct, cs = struct
            return cs
        else:
            raise NotImplementedError

    def symmetry_encode(self, struct):
        return self._symmetry_encode(self.detect_symmetry(struct))

    def symmetry_decode(self, sym_vec):
        try:
            crys_sys_bit = np.where(sym_vec)[0][0]  # note: choose the first ones
            crys_sys = decode_cs(crys_sys_bit)
        except IndexError:
            print('Use default crystal system')
            crys_sys = self.default_crys_sys
        if self.logger is not None:
            self.logger.record_anything1(operation='Decoded Crystal System', result=crys_sys)
        return crys_sys


class OneHotProcessorCrysSys(OneHotLatticeScaleProcessor):
    def __init__(self, species_list, splits, lat_ub, default_crys_sys, logger):
        self.default_crys_sys = default_crys_sys
        super().__init__(species_list, splits, lat_ub, logger=logger)
        self._load_symmetry()
        return

    def _load_symmetry(self):
        self.crys_sys_precision = _num_crys_sys
        self.num_var = self.lattice_precision + \
                       self.crys_sys_precision + \
                       self.num_elements * self.space_precision
        self.crys_sys_processor = OneHotVecProcessorCrysSys(crys_sys_precision=self.crys_sys_precision,
                                                            default_crys_sys=self.default_crys_sys,
                                                            logger=self.logger)
        return

    def one_hot_encode(self, struct):
        struct = self._one_hot_encode_basic(struct)
        struct_vec = []
        struct_vec += self._lat_encode(struct)
        struct_vec += self.crys_sys_processor.symmetry_encode(struct)
        struct_vec += self._pos_encode(struct)
        return np.concatenate(struct_vec)

    def fetch_symmetry_vec_from_struct_vec(self, struct_vec):
        if len(struct_vec.shape) == 1:
            return struct_vec[self.lattice_precision: -self.num_elements * self.space_precision]
        else:
            return struct_vec[:, self.lattice_precision: -self.num_elements * self.space_precision]

    def _crystal_system_decode(self, struct_vec):
        sym_vec = self.fetch_symmetry_vec_from_struct_vec(struct_vec)
        return self.crys_sys_processor.symmetry_decode(sym_vec)

    def _lattice_length_decode(self, lat_len_vec):
        lat_len_list = np.split(lat_len_vec, _num_lat_len)
        lat = []
        for i, lat_len_bit_vec in enumerate(lat_len_list):
            try:
                lat_len_bit = np.where(lat_len_bit_vec)[0]
                lat_len_bit = lat_len_bit[0]  # note: choose the first ones
                lat_len = self._single_lattice_length_decode(lat_len_bit)
            except IndexError:
                print('Use default lattice length')
                lat_len = self.lat_mean
            if i == 0:
                lat_len_a = lat_len
            if i == 1:  # b
                if crystal_system_dict()[self.crys_sys]['b'] == 'a':
                    lat_len = lat_len_a
            if i == 2:  # c
                if crystal_system_dict()[self.crys_sys]['c'] == 'a':
                    lat_len = lat_len_a
            lat.append(lat_len)
        return lat

    def _lattice_angle_decode(self, lat_ang_vec):
        lat_ang_list = np.split(lat_ang_vec, _num_lat_ang)
        lat = []
        angle_name = ['alpha', 'beta', 'gamma']
        for i, lat_ang in enumerate(lat_ang_list):
            preassigned_ang = crystal_system_dict()[self.crys_sys][angle_name[i]]
            if preassigned_ang is not None:
                lat_ang = preassigned_ang
            else:
                try:
                    lat_ang_bit = np.where(lat_ang)[0]
                    lat_ang_bit = lat_ang_bit[0]  # note: choose the first ones
                    lat_ang = self._single_lattice_angle_decode(lat_ang_bit)
                except IndexError:
                    print('Use default lattice angle')
                    lat_ang = default_ang_mean
            lat.append(lat_ang)
        return lat

    def _one_hot_decode_lat_gen(self, struct_vec):
        lat_len_vec = struct_vec[0: self.lat_len_precision]
        lat_ang_vec = struct_vec[self.lat_len_precision: self.lattice_precision]
        self.crys_sys = self._crystal_system_decode(struct_vec)
        lat_len = self._lattice_length_decode(lat_len_vec)
        lat_ang = self._lattice_angle_decode(lat_ang_vec)
        return lat_len, lat_ang


class OneHotCrysSysConstraintsBuilder:
    def __init__(self, crys_sys_precision):
        self.crys_sys_precision = crys_sys_precision
        return

    def crystal_system_constraints(self):
        crys_sys_mat = np.ones((1, self.crys_sys_precision))
        return [crys_sys_mat, [1], [1]]

    def __call__(self, *args, **kwargs):
        return {'linear': self.crystal_system_constraints(), 'quadratic': None}
