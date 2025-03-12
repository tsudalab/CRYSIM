import numpy as np
from pymatgen.core import Structure, PeriodicSite
from pyxtal.lattice import para2matrix

from src.struct_utils import array_to_pymatgen_struct, frac_pbc, unique_elements
from src.struct_utils import perturb_frac_atom
from src.optimizer.rand_gen import LatticeLengthController, \
    default_ang_mean, default_ang_lb, default_ang_ub

_num_lat_len = 3
_num_lat_ang = 3
_lat_precs = 10
_angle_precs = 90

_default_pos = (0.3, 0.2, 0.6)

""" In the basic OneHot class, make sure that when decoding, we always choose the 
    first bit in each parameter domain to calculate the decoded values, so that 
    despite there are more than required bits in this domain, the decoding process
    can still be conducted successfully.
    In that case, we can run the optimization without constraints. 
    
    For example, the first 100 bits is for the first lattice param. If using constraints,
    there will be only one bit to be one in the 100 bits. However, in implementation, 
    we always choose the first bit that is one to decode, actually not counting how many
    ones there are in the 100 bits. 
    
    We will mark the related code lines as 'choose the first ones' to emphasize this
    consideration in implementation. """


def one_hot_num_var(precs, num_species, lat_precs=_lat_precs, angle_precs=_angle_precs):
    return precs * precs * precs * num_species + \
        lat_precs * precs * _num_lat_len + angle_precs * _num_lat_ang


def single_lat_param_encode(value, param_lower_bound, unit, split):
    encoded_value = value - param_lower_bound
    encoded_value = encoded_value if encoded_value >= 0 else 0
    return int(np.min([np.floor(encoded_value / unit), split - 1]))


def single_frac_site_encode(site, units, splits):
    """ units: list of units in x, y, z axes """
    site = frac_pbc(site)
    site = np.array([int(np.around(site[0] / units[0])),
                     int(np.around(site[1] / units[1])),
                     int(np.around(site[2] / units[2]))])
    site = np.array([np.where(site[0] >= splits[0], 0, site[0]),
                     np.where(site[1] >= splits[1], 0, site[1]),
                     np.where(site[2] >= splits[2], 0, site[2])])
    return site


class OneHotProcessor4Element:
    """ Dealing with atoms of the same element species in the crystal.
        Only deal with atomic positions, excluding lattice parameters. """

    def __init__(self, splits, num_atom):
        self.splits = splits
        self.num_atom = num_atom
        self._load_container()
        return

    def _load_container(self):
        self._atomic_positions_ingredients()
        self.element_sites = np.zeros(shape=(self.splits_a, self.splits_b, self.splits_c))
        return

    def _atomic_positions_ingredients(self):
        # atomic positions ingredients
        self.splits_a, self.splits_b, self.splits_c = self.splits
        self.unit_a, self.unit_b, self.unit_c = (
            1.0 / self.splits_a, 1.0 / self.splits_b, 1.0 / self.splits_c)
        return

    def _single_site_encode(self, site):
        if isinstance(site, PeriodicSite):
            site = site.frac_coords  # pymatgen site
        return single_frac_site_encode(site, units=[self.unit_a, self.unit_b, self.unit_c],
                                       splits=[self.splits_a, self.splits_b, self.splits_c])

    def encode_site(self, site):
        bit = self._single_site_encode(site)
        self.element_sites[bit[0], bit[1], bit[2]] = 1
        return

    def output(self):
        return self.element_sites.flatten()

    def set_sites(self, element_sites):
        self.element_sites = element_sites
        return

    def _decode_sites(self):
        atom_vec = np.reshape(self.element_sites,
                              newshape=(self.splits_a, self.splits_b, self.splits_c))
        atom_bit = np.where(atom_vec)
        atoms = []
        # for i in range(len(atom_bit[0])):
        for i in range(self.num_atom):  # note: choose the first ones
            try:
                atom = (atom_bit[0][i], atom_bit[1][i], atom_bit[2][i])
                atom = (atom[0] * self.unit_a, atom[1] * self.unit_b, atom[2] * self.unit_c)
            except IndexError:
                atom = self._empty_atoms()
            if atom is not None:
                atoms.append(atom)
        return atoms

    def _empty_atoms(self):
        print('Use default position')
        atom = (_default_pos[0] + perturb_frac_atom(),
                _default_pos[1] + perturb_frac_atom(),
                _default_pos[2] + perturb_frac_atom())
        return atom

    def decode_sites(self):
        return self._decode_sites()


class OneHotProcessor:
    def __init__(self, species_list, splits, lat_ub, logger):
        self.species_list = species_list
        self.elements, self.num_species_atom = unique_elements(species_list=species_list)
        # self.elements = pd.unique(pd.Series(species_list)).tolist()
        self.num_elements = len(self.elements)
        # the usage of "elements" and "species" is consistent with the one in pymatgen.core.Structure

        self.splits = splits
        self.lat_ub = lat_ub
        self.logger = logger
        llc = LatticeLengthController(num_atom=len(self.species_list), logger=self.logger)
        self.lat_lb = llc.min()
        self.lat_mean = llc.mean()

        self._lattice_parameters_ingredients()
        self._load_properties()
        return

    def _lattice_parameters_ingredients(self):
        # lattice parameters ingredients
        self.lat_precs = _lat_precs
        self.splits_len = int(np.max(self.splits) * self.lat_precs)
        self.unit_len = self.lat_ub / self.splits_len

        self.splits_ang = _angle_precs
        self.unit_ang = 180 / self.splits_ang
        return

    def _load_properties(self):
        # descriptive variables for outside usage
        self.splits_a, self.splits_b, self.splits_c = self.splits
        self.unit_a, self.unit_b, self.unit_c = (
            1.0 / self.splits_a, 1.0 / self.splits_b, 1.0 / self.splits_c)
        self.space_precision = self.splits_a * self.splits_b * self.splits_c
        self.lat_len_precision = _num_lat_len * self.splits_len
        self.lat_ang_precision = _num_lat_ang * self.splits_ang
        self.lattice_precision = self.lat_len_precision + self.lat_ang_precision
        self.num_var = self.lattice_precision + self.num_elements * self.space_precision
        return

    def init_position_container(self):
        ohp_ele = {ele: OneHotProcessor4Element(self.splits, num_atom) for ele, num_atom
                   in zip(self.elements, self.num_species_atom)}
        return ohp_ele

    def init_lattice_container(self):
        lat_len_vec = np.zeros(shape=(_num_lat_len * self.splits_len))
        lat_ang_vec = np.zeros(shape=(_num_lat_ang * self.splits_ang))
        return lat_len_vec, lat_ang_vec

    def _single_lattice_length_encode(self, lat_len):
        return single_lat_param_encode(lat_len, param_lower_bound=0,
                                       unit=self.unit_len, split=self.splits_len)

    def _single_lattice_angle_encode(self, lat_ang):
        return single_lat_param_encode(lat_ang, param_lower_bound=0,
                                       unit=self.unit_ang, split=self.splits_ang)

    def _lat_encode(self, struct):
        lat_len_vec, lat_ang_vec = self.init_lattice_container()
        for i, lat_len in enumerate(struct.lattice.lengths):  # a, b, c
            bit = self._single_lattice_length_encode(lat_len)
            lat_len_vec[i * self.splits_len + bit] = 1
        for i, lat_ang in enumerate(struct.lattice.angles):  # alpha, beta, gamma
            bit = self._single_lattice_angle_encode(lat_ang)
            lat_ang_vec[i * self.splits_ang + bit] = 1
        return [lat_len_vec, lat_ang_vec]

    def _pos_encode(self, struct):
        ohp_ele = self.init_position_container()
        pos_vec = []
        for site in struct.sites:
            element = site.species.elements[0].name
            ohp_ele[element].encode_site(site)
        for spec in self.elements:
            pos_vec.append(ohp_ele[spec].output())
        return pos_vec

    def _one_hot_encode_basic(self, struct):
        struct = array_to_pymatgen_struct(lattice=struct.lattice, pos=struct.pos,
                                          species_list=struct.species, cart=struct.cart) \
            if not isinstance(struct, Structure) else struct
        return struct

    def one_hot_encode(self, struct):
        struct = self._one_hot_encode_basic(struct)
        struct_vec = []
        struct_vec += self._lat_encode(struct)
        struct_vec += self._pos_encode(struct)
        return np.concatenate(struct_vec)

    def _single_lattice_length_decode(self, lat_len_bit):
        return float(lat_len_bit * self.unit_len)

    def _single_lattice_angle_decode(self, lat_ang_bit):
        return float(lat_ang_bit * self.unit_ang)

    def _lattice_length_decode(self, lat_len_vec):
        lat_len_list = np.split(lat_len_vec, _num_lat_len)
        lat = []
        for lat_len in lat_len_list:  # note: choose the first ones
            try:
                lat_len_bit = np.where(lat_len)[0]
                lat_len_bit = lat_len_bit[0]
                lat_len = self._single_lattice_length_decode(lat_len_bit)
            except IndexError:
                print('Use default lattice length')
                lat_len = self.lat_mean
            lat.append(lat_len)
        return lat

    def _lattice_angle_decode(self, lat_ang_vec):
        lat_ang_list = np.split(lat_ang_vec, _num_lat_ang)
        lat = []
        for lat_ang in lat_ang_list:  # note: choose the first ones
            try:
                lat_ang_bit = np.where(lat_ang)[0]
                lat_ang_bit = lat_ang_bit[0]
                lat_ang = self._single_lattice_angle_decode(lat_ang_bit)
            except IndexError:
                print('Use default lattice angle')
                lat_ang = default_ang_mean
            lat.append(lat_ang)
        return lat

    def _one_hot_decode_lat_gen(self, struct_vec):
        lat_len_vec = struct_vec[0: self.lat_len_precision]
        lat_ang_vec = struct_vec[self.lat_len_precision: self.lattice_precision]
        lat_len = self._lattice_length_decode(lat_len_vec)
        lat_ang = self._lattice_angle_decode(lat_ang_vec)
        return lat_len, lat_ang

    def _one_hot_decode_lat(self, struct_vec):
        lat_len, lat_ang = self._one_hot_decode_lat_gen(struct_vec)
        lat = para2matrix(cell_para=lat_len + lat_ang, radians=False, format='lower')
        default_lat = 8
        if np.isnan(np.sum(lat)):  # para2matrix can lead to nan
            lat[np.isnan(lat)] = default_lat
        lat = np.where(lat > 1e-4, lat, 0)
        return lat

    def _one_hot_decode_ele(self, struct_vec):
        atom_vec = np.split(struct_vec[-self.num_elements * self.space_precision:], self.num_elements)
        ohp_ele = self.init_position_container()
        # the encoded vector does not contain element species info
        # therefore, it is necessary to obtain the info based on this class
        atoms, species_list = [], []
        for i, species in enumerate(self.elements):
            obj = ohp_ele[species]
            obj.set_sites(element_sites=atom_vec[i])
            atom = obj.decode_sites()
            atoms += atom
            species_list += [species] * len(atom)
        return atoms, species_list

    def one_hot_decode(self, struct_vec):
        lat = self._one_hot_decode_lat(struct_vec)
        atoms, species_list = self._one_hot_decode_ele(struct_vec)
        pmg_struct = Structure(lattice=lat, species=species_list, coords=atoms,
                               coords_are_cartesian=False)
        return pmg_struct


class OneHotLatticeScaleProcessor(OneHotProcessor):
    def __init__(self, species_list, splits, lat_ub, logger,
                 ang_ub=default_ang_ub, ang_lb=default_ang_lb):
        self.ang_ub = ang_ub
        self.ang_lb = ang_lb
        super().__init__(species_list, splits, lat_ub, logger=logger)
        return

    def _lattice_parameters_ingredients(self):
        # lattice parameters ingredients
        self.lat_precs = _lat_precs
        self.splits_len = int(np.max(self.splits) * self.lat_precs)
        self.splits_len = int(self.splits_len * (self.lat_ub - self.lat_lb) / self.lat_ub)
        self.unit_len = (self.lat_ub - self.lat_lb) / self.splits_len

        self.splits_ang = _angle_precs
        self.splits_ang = int(self.splits_ang * (self.ang_ub - self.ang_lb) / 180)
        self.unit_ang = (self.ang_ub - self.ang_lb) / self.splits_ang
        return

    def _single_lattice_length_encode(self, lat_len):
        return single_lat_param_encode(lat_len, param_lower_bound=self.lat_lb,
                                       unit=self.unit_len, split=self.splits_len)

    def _single_lattice_angle_encode(self, lat_ang):
        return single_lat_param_encode(lat_ang, param_lower_bound=self.ang_lb,
                                       unit=self.unit_ang, split=self.splits_ang)

    def _single_lattice_length_decode(self, lat_len_bit):
        return float(lat_len_bit * self.unit_len) + self.lat_lb

    def _single_lattice_angle_decode(self, lat_ang_bit):
        return float(lat_ang_bit * self.unit_ang) + self.ang_lb


class OneHotProcessorExtractor:
    def __init__(self, one_hot_obj):
        assert isinstance(one_hot_obj, OneHotProcessor)
        self.species_list = one_hot_obj.species_list
        self.num_elements = one_hot_obj.num_elements
        self.num_species_atom = one_hot_obj.num_species_atom

        self.unit_a, self.unit_b, self.unit_c, self.splits_a, self.splits_b, self.splits_c = \
            one_hot_obj.unit_a, one_hot_obj.unit_b, one_hot_obj.unit_c, \
                one_hot_obj.splits_a, one_hot_obj.splits_b, one_hot_obj.splits_c
        self.unit_len, self.unit_ang = one_hot_obj.unit_len, one_hot_obj.unit_ang
        self.splits_len = one_hot_obj.splits_len
        self.splits_ang = one_hot_obj.splits_ang

        self.space_precision, self.lat_len_precision, self.lat_ang_precision, \
            self.lattice_precision, self.num_var = one_hot_obj.space_precision, \
            one_hot_obj.lat_len_precision, one_hot_obj.lat_ang_precision, \
            one_hot_obj.lattice_precision, one_hot_obj.num_var

        if hasattr(one_hot_obj, 'lat_lb'):
            self.lat_lb = one_hot_obj.lat_lb
        if hasattr(one_hot_obj, 'ang_ub'):
            self.ang_ub = one_hot_obj.ang_ub
        if hasattr(one_hot_obj, 'ang_lb'):
            self.ang_lb = one_hot_obj.ang_lb
        return

    def __call__(self, *args, **kwargs):
        """ Output the built constraints. """
        return {}


class OneHotAtomNumberConstraintsBuilder(OneHotProcessorExtractor):
    def __init__(self, one_hot_obj):
        super().__init__(one_hot_obj)
        return

    def atom_number_constraints(self):
        atom_num_mat = np.zeros((self.num_elements, self.num_elements * self.space_precision))
        # Each self.space_precision bits represent one element species,
        # and we require that the number of 1(s) for one element species should
        # be equal to the number of atoms in the chemical formula.
        for i in range(self.num_elements):
            atom_num_mat[i, i * self.space_precision: (i + 1) * self.space_precision] = 1
        return [atom_num_mat, self.num_species_atom, [1] * self.num_elements]

    def __call__(self, *args, **kwargs):
        return {'linear': self.atom_number_constraints(),
                'quadratic': None}


class OneHotAtomDistConstraintsBuilder(OneHotProcessorExtractor):
    def __init__(self, one_hot_obj, dist_min):
        super().__init__(one_hot_obj)
        # The units here is frac coord-based.
        # Use assumed lattice parameter to switch it into Cartesian space.
        assumed_lat_dim = 10
        self.unit_a, self.unit_b, self.unit_c = \
            self.unit_a * assumed_lat_dim, self.unit_b * assumed_lat_dim, self.unit_c * assumed_lat_dim
        self.bits_min_a, self.bits_min_b, self.bits_min_c = \
            self._calculate_min_bit(dist_min, self.unit_a), \
                self._calculate_min_bit(dist_min, self.unit_b), \
                self._calculate_min_bit(dist_min, self.unit_c)
        return

    @staticmethod
    def _calculate_min_bit(dist_min, unit):
        bits_min = int(dist_min / unit) - 1
        return bits_min if bits_min > 0 else 0

    def _neighborhood_detecting(self, array_3d, x, y, z):
        constraint = x * self.splits_b * self.splits_c + y * self.splits_c + z
        neighbor_box = np.array(np.meshgrid(
            [constraint],
            circulate_slice(self.splits_a, x - self.bits_min_a, x + self.bits_min_a + 1),
            circulate_slice(self.splits_b, y - self.bits_min_b, y + self.bits_min_b + 1),
            circulate_slice(self.splits_c, z - self.bits_min_c, z + self.bits_min_c + 1)
        )).T.reshape(-1, 4)
        array_3d[neighbor_box[:, 0], neighbor_box[:, 1],
        neighbor_box[:, 2], neighbor_box[:, 3]] = 1
        # array_3d[constraint, x, y, z] = 0  # center
        return array_3d

    def atom_matrix_constraints(self):
        atom_mat = np.zeros((self.space_precision,
                             self.splits_a, self.splits_b, self.splits_c))
        for x in range(self.splits_a):
            for y in range(self.splits_b):
                for z in range(self.splits_c):
                    atom_mat = self._neighborhood_detecting(atom_mat, x, y, z)
        atom_mat = atom_mat.reshape(self.space_precision, -1)
        atom_mat = np.tile(atom_mat, (self.num_elements, self.num_elements))
        ctr_row, ctr_col = np.diag_indices_from(atom_mat)  # assigning centers as zeros
        atom_mat[ctr_row, ctr_col] = 0
        return [atom_mat, [0] * self.num_elements * self.space_precision,
                [1] * self.num_elements * self.space_precision]

    def __call__(self, *args, **kwargs):
        return {'linear': None,
                'quadratic': self.atom_matrix_constraints()}


def circulate_slice(length, start, end):
    array = np.arange(length)
    if start < 0:
        return np.concatenate([array[start:], array[0:end]])
    elif end > length:
        return np.concatenate([array[start:], array[0:end - length]])
    else:
        return array[start: end]


class OneHotLatticeConstraintsBuilder(OneHotProcessorExtractor):
    def __init__(self, one_hot_obj):
        super().__init__(one_hot_obj)
        return

    def _lattice_number_constraints(self):
        lat_mat = np.zeros((_num_lat_len + _num_lat_ang, self.lattice_precision))
        # Each self.split_len, self.splits_ang bits represent one lat para,
        # and we only allow there is one element to be 1.
        for i in range(_num_lat_len):
            lat_mat[i, i * self.splits_len: (i + 1) * self.splits_len] = 1
        for i in range(_num_lat_ang):
            lat_mat[i + _num_lat_len, self.lat_len_precision + i * self.splits_ang:
                                      self.lat_len_precision + (i + 1) * self.splits_ang] = 1
        return lat_mat

    def lattice_number_constraints(self):
        lat_mat = self._lattice_number_constraints()
        return [lat_mat, [1] * (_num_lat_len + _num_lat_ang), [1] * (_num_lat_len + _num_lat_ang)]

    def __call__(self, *args, **kwargs):
        return {'linear': self.lattice_number_constraints(),
                'quadratic': None}
