import re
import numpy as np
import pandas as pd
import spglib
from pymatgen.core import Structure, Lattice
from configparser import ConfigParser
from pyxtal.lattice import matrix2para


""" -------------------------------------- """
""" ---- data structure & log related ---- """
""" -------------------------------------- """


def get_species_list(sys_name):
    alphabets = re.findall(r'[A-Za-z]+', sys_name)
    numbers = [int(num) for num in re.findall(r'\d+', sys_name)]
    species_list = []
    for alpha, number in zip(alphabets, numbers):
        species_list += [alpha] * number
    return species_list


def unique_elements(species_list):
    elements_counting = pd.Series(species_list).value_counts(sort=False)  # pd.unique does not sort
    elements = list(elements_counting.index)
    num_species_atom = elements_counting.to_list()
    return elements, num_species_atom


def get_system_size(sys_name):
    system_size = re.findall(r'\d+', sys_name)
    system_size = sum([eval(i) for i in system_size])
    return system_size


class Struct:
    def __init__(self, lattice, pos, species, cart, pbc=None):
        if pbc is None:
            pbc = [True, True, True]

        self.lattice = lattice
        self.pos = pos
        self._sp = species
        self.cart = cart
        self.pbc = pbc
        return

    @property
    def lattice_param(self):
        return matrix2para(self.lattice, radians=False)

    @property
    def species(self):
        species = []
        if isinstance(self._sp, tuple):  # (['Na', 'Cl'], [4, 4])
            specie, specie_number = self._sp
            for j in range(len(specie)):
                species += [specie[j]] * int(specie_number[j])
        else:
            species = self._sp  # ['Na', 'Na', 'Na', 'Na', 'Cl', 'Cl', 'Cl', 'Cl']
        return species

    @property
    def system_size(self):
        return self.pos.shape[0]

    @property
    def space_group(self):
        struct = array_to_pymatgen_struct(lattice=self.lattice, pos=self.pos,
                                          species_list=self.species, cart=self.cart)
        _, spg_num = SPG.struct_spg_estimate(struct)
        return spg_num

    @property
    def crystal_system(self):
        return get_bravais_lattice(self.space_group)


def array_to_pmg_for_training_and_recording(struct):
    if not isinstance(struct, Structure):
        struct = array_to_pymatgen_struct(struct.lattice, struct.pos,
                                          struct.species, cart=False)
    return struct


class StructReader:
    def __init__(self, struct_file_path, cart=False):
        self.poscar_list = []
        self.ele_list = []
        self.ele_num_list = []
        self.cart = cart
        with open(struct_file_path) as f:
            content = f.readlines()
        id_list = [i for i, line in enumerate(content) if "ID_" in line]

        for i in range(len(id_list) - 1):
            latpara = self._collect_mat(content, id_list[i] + 2, id_list[i] + 5)
            dire = self._collect_mat(content, id_list[i] + 8, id_list[i + 1])
            if cart:
                dire = dire @ latpara
            self.poscar_list.append([latpara, dire])
            self.ele_list.append(re.findall(r'[\w]+', content[id_list[i] + 5]))
            self.ele_num_list.append(list(map(int, re.findall(r'[\d]+', content[id_list[i] + 6]))))

        latpara = self._collect_mat(content, id_list[-1] + 2, id_list[-1] + 5)
        dire = self._collect_mat(content, id_list[-1] + 8, len(content))
        if cart:
            dire = dire @ latpara
        self.poscar_list.append((latpara, dire))
        self.ele_list.append(re.findall(r'[\w]+', content[id_list[-1] + 5]))
        self.ele_num_list.append(list(map(int, re.findall(r'[\d]+', content[id_list[-1] + 6]))))
        return

    @staticmethod
    def _extract(line):
        """ Extract the values and ignore the str/nan. """
        content = re.findall(r'[\d.-]+', line)
        num = np.float64(content)
        return num

    def _collect_mat(self, content, i, j):
        return np.array([self._extract(content[idx]) for idx in range(i, j)])

    # def __call__(self, *args, **kwargs):
    #     return self.poscar_list, self.ele_list, self.ele_num_list

    def to_pymatgen(self):
        struct_list = []
        for i in range(len(self.poscar_list)):
            struct = array_to_pymatgen_struct(lattice=self.poscar_list[i][0],
                                              pos=self.poscar_list[i][1],
                                              species_list=(self.ele_list[i], self.ele_num_list[i]),
                                              cart=self.cart)
            struct_list.append(struct)
        return struct_list

    def to_struct(self):
        struct_list = []
        for i in range(len(self.poscar_list)):
            struct = Struct(lattice=self.poscar_list[i][0],
                            pos=self.poscar_list[i][1],
                            species=(self.ele_list[i], self.ele_num_list[i]),
                            cart=self.cart)
            struct_list.append(struct)
        return struct_list


def array_to_pymatgen_struct(lattice, pos, species_list, cart):
    lattice = Lattice(matrix=lattice, pbc=(True, True, True))
    species = []
    if isinstance(species_list, tuple):  # (['Na', 'Cl'], [4, 4])
        species_list, element_numbers = species_list
        for j in range(len(species_list)):
            species += [species_list[j]] * int(element_numbers[j])
    else:
        species = species_list  # ['Na', 'Na', 'Na', 'Na', 'Cl', 'Cl', 'Cl', 'Cl']
    struct = Structure(lattice=lattice, species=species, coords=pos,
                       coords_are_cartesian=cart)
    return struct


def pymatgen_struct_to_array(struct):
    lattice = struct.lattice.matrix
    sites = [np.expand_dims(atom.frac_coords, axis=0) for atom in struct.sites]
    sites = np.concatenate(sites, axis=0)
    species_list = [atom.name for atom in struct.species]
    struct = Struct(lattice, sites, species_list, cart=False)  # Structure.frac_coord
    return struct


def get_curr_struct_num(init_pos_path, struct_size=None):
    try:
        with open(init_pos_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return 0
    if struct_size is None:
        id_list = []
        for i, line in enumerate(lines):
            if "ID_" in line:
                id_list.append(i)
            if len(id_list) == 2:
                break
        struct_size = id_list[1] - (id_list[0] + 8)
    num_line = len(lines)
    struct_file_len = 8 + struct_size
    return int(num_line / struct_file_len)


def save_pymatgen_struct(struc, cid, id_offset, fpath):
    # ---------- poscar format
    pos = struc.to(fmt='poscar')
    pos = pos.split('\n')
    blank_indx = pos.index('')  # cut unnecessary parts
    pos = pos[:blank_indx]
    pos[0] = 'ID_{}'.format(cid + id_offset)  # replace with ID
    lines = [line + '\n' for line in pos]

    # ---------- append POSCAR
    with open(fpath, 'a+') as f:
        for line in lines:
            f.write(line)
    return


def frac_pbc(site):
    flag = 1 if isinstance(site, tuple) else 0
    if flag:
        site = np.array(site)
    while (site >= 1).any() or (site < 0).any():
        site = np.where(site < 0, site + 1, site)  # pbc
        site = np.where(site >= 1, site - 1, site)  # pbc
    if flag:
        site = tuple(site)
    return site


def atoms_overlapping(atoms_array):
    atoms_array = np.round(atoms_array, 4)
    atoms_array = frac_pbc(atoms_array)
    unique_atoms = np.unique(atoms_array, axis=0)
    if len(atoms_array) != len(unique_atoms):
        return True
    else:
        return False


def perturb_frac_atom():
    value = np.random.normal(0.1, 0.08)
    sign = np.random.choice([1, -1])
    return value * sign


def compare_struct(pmg_struct0, pmg_struct1):
    species = pd.unique(pd.Series(pmg_struct0.species)).tolist()

    def extract_struct(pmg_struct):
        species_coord = {spec: [] for spec in species}
        for site in pmg_struct:
            fc = site.frac_coords
            fc = np.where(fc < 0, fc + 1, fc)
            fc = np.where(fc > 1, fc - 1, fc)
            fc = fc @ pmg_struct.lattice.matrix
            species_coord[site.specie].append(fc)
        for key, values in species_coord.items():
            values.sort(key=lambda x: (x[0], x[1], x[2]))
            species_coord[key] = np.concatenate(values).reshape(-1, 3)
        species_coord = np.concatenate([species_coord[key]
                                        for key in species_coord.keys()], axis=0)
        return species_coord

    struct0 = extract_struct(pmg_struct0)
    struct1 = extract_struct(pmg_struct1)
    delta = struct0 - struct1
    return delta


""" ---------------------------------- """
""" ---- lattice symmetry related ---- """
""" ---------------------------------- """


class LatticeInfo:
    def __init__(self, lat_para):
        self.a_vec, self.b_vec, self.c_vec = lat_para[0, :], lat_para[1, :], lat_para[2, :]
        return

    @staticmethod
    def _vec2norm(vec):
        return np.linalg.norm(vec, ord=2)

    @staticmethod
    def _vec2angle(vec1, vec2):
        angle = np.arccos(np.dot(vec1, vec2) /
                          (np.linalg.norm(vec1, ord=2) * np.linalg.norm(vec2, ord=2) + 1e-8)
                          )
        return angle / np.pi * 180

    def __call__(self, *args, **kwargs):
        _a, _b, _c = self._vec2norm(self.a_vec), self._vec2norm(self.b_vec), self._vec2norm(self.c_vec)
        _alpha, _beta, _gamma = (self._vec2angle(self.b_vec, self.c_vec),
                                 self._vec2angle(self.a_vec, self.c_vec),
                                 self._vec2angle(self.a_vec, self.b_vec))
        return _a, _b, _c, _alpha, _beta, _gamma

    @classmethod
    def analyze_lattice(cls, struct):
        lat = struct.lattice if not isinstance(struct, Structure) else struct.lattice.matrix
        return cls(lat)()


def oh2feat_map(vec_alphabet):
    return {i: ls for i, ls in enumerate(vec_alphabet)}


def feat2oh_map(vec_alphabet):
    return {ls: i for i, ls in enumerate(vec_alphabet)}


crystal_system_list = ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal',
                       'trigonal', 'hexagonal', 'cubic']


def crystal_system_dict():
    ls_dict = {}
    lat_para_name = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
    for ls_name in crystal_system_list:
        ls_dict[ls_name] = {name: None for name in lat_para_name}
    ls_dict['monoclinic']['alpha'], ls_dict['monoclinic']['beta'] = 90, 90
    ls_dict['orthorhombic']['alpha'], ls_dict['orthorhombic']['beta'], \
        ls_dict['orthorhombic']['gamma'] = 90, 90, 90
    ls_dict['tetragonal']['alpha'], ls_dict['tetragonal']['beta'], \
        ls_dict['tetragonal']['gamma'], ls_dict['tetragonal']['b'] = 90, 90, 90, 'a'
    ls_dict['hexagonal']['alpha'], ls_dict['hexagonal']['beta'], \
        ls_dict['hexagonal']['gamma'], ls_dict['hexagonal']['b'] = 90, 90, 120, 'a'
    ls_dict['trigonal']['alpha'], ls_dict['trigonal']['beta'], \
        ls_dict['trigonal']['gamma'], ls_dict['trigonal']['b'] = 90, 90, 120, 'a'
    ls_dict['cubic']['alpha'], ls_dict['cubic']['beta'], \
        ls_dict['cubic']['gamma'], ls_dict['cubic']['b'], ls_dict['cubic']['c'] = 90, 90, 90, 'a', 'a'
    return ls_dict


spg_dict = {'triclinic': [1, 3],
            'monoclinic': [3, 16],
            'orthorhombic': [16, 75],
            'tetragonal': [75, 143],
            'trigonal': [143, 168],
            'hexagonal': [168, 195],
            'cubic': [195, 231]}


def get_bravais_lattice(space_group_number):
    for cs in spg_dict.keys():
        if (space_group_number >= spg_dict[cs][0]) and (space_group_number < spg_dict[cs][1]):
            return cs
        # if space_group_number in list(range(spg_dict[cs][0], spg_dict[cs][1])):
        #     return cs
    raise Exception("Wrong spg")


def get_spg_for_cs(cs, spg_list):
    if spg_list is None:
        return list(range(spg_dict[cs][0], spg_dict[cs][1]))
    else:
        spg_list = np.array(spg_list)
        return spg_list[(spg_list >= spg_dict[cs][0]) & (spg_list < spg_dict[cs][1])]


def get_cs_list(spg_list):
    cs_list = []
    for cs in crystal_system_list:
        if len(get_spg_for_cs(cs, spg_list)) != 0:
            cs_list.append(cs)
    return cs_list


class SPG:
    def __call__(self, lattice, sites, species):
        spg = spglib.get_spacegroup(cell=(lattice, sites, species), symprec=0.003)
        try:
            space_group_number = int(re.findall(r'\((\d+)\)', spg)[0])
        except TypeError:
            space_group_number = 0
        return spg, space_group_number

    @staticmethod
    def _load_struct(pmg_struct):
        if not isinstance(pmg_struct, Structure):
            pmg_struct = array_to_pymatgen_struct(pmg_struct.lattice, pmg_struct.pos,
                                                  pmg_struct.species, pmg_struct.cart)
        lattice = pmg_struct.lattice.matrix
        sites = [np.expand_dims(atom.frac_coords, axis=0) for atom in pmg_struct.sites]
        sites = np.concatenate(sites, axis=0)
        element = [atom.name for atom in pmg_struct.species]
        species = pd.Series(element).astype('category').cat.codes.values
        return lattice, sites, species

    @staticmethod
    def _load_lat_param(pmg_struct):
        if not isinstance(pmg_struct, Structure):
            pmg_struct = array_to_pymatgen_struct(pmg_struct.lattice, pmg_struct.pos,
                                                  pmg_struct.species, pmg_struct.cart)
        return list(pmg_struct.lattice.abc) + list(pmg_struct.lattice.angles)

    @classmethod
    def struct_spg_estimate(cls, pmg_struct):
        lattice, sites, species = cls._load_struct(pmg_struct)
        return cls()(lattice, sites, species)

    @classmethod
    def struct_bravais_estimate(cls, pmg_struct, spg):
        a, b, c, alpha, beta, gamma = cls._load_lat_param(pmg_struct)
        a, b, c, alpha, beta, gamma = \
            np.round(a, 3), np.round(b, 3), np.round(c, 3), \
            np.round(alpha, 3), np.round(beta, 3), np.round(gamma, 3)
        if alpha != beta:
            return 'triclinic'
        else:  # alpha == beta
            if alpha != 90:
                return 'triclinic'
            else:  # alpha == beta == 90
                if gamma == 120:
                    if np.round(a, 3) != np.round(b, 3):
                        return 'monoclinic'
                    else:  # gamma == 120, a == b
                        if spg is None:
                            return 'hexagonal'
                        else:
                            if (spg >= spg_dict['trigonal'][0]) & (spg < spg_dict['trigonal'][1]):
                                return 'trigonal'
                            elif (spg >= spg_dict['hexagonal'][0]) & (spg < spg_dict['hexagonal'][1]):
                                return 'hexagonal'
                            else:
                                raise Exception("The given spg is out of the scale "
                                                "of trigonal / hexagonal.")
                else:
                    if gamma != 90:
                        return 'monoclinic'
                    else:  # gamma == 90
                        if np.round(a, 3) != np.round(b, 3):
                            return 'orthorhombic'
                        else:  # a == b
                            if np.round(a, 3) != np.round(c, 3):
                                return 'tetragonal'
                            else:  # a == c
                                return 'cubic'


def get_best_spg(struct_list, e_list):
    spg_list = [SPG.struct_spg_estimate(struct)[1] for struct in struct_list]
    return spg_list[np.argmin(np.array(e_list))]


def get_best_crys_sys(struct_list, e_list):
    crys_sys_list = [SPG.struct_bravais_estimate(struct, None) for struct in struct_list]
    return crys_sys_list[np.argmin(np.array(e_list))]


def get_best_spg2(spg_list, wp_list, e_list):
    e_min_idx = np.argmin(np.array(e_list))
    return spg_list[e_min_idx], wp_list[e_min_idx]


def get_best_crys_sys2(spg_list, e_list):
    e_min_idx = np.argmin(np.array(e_list))
    spg_min = spg_list[e_min_idx]
    return get_bravais_lattice(spg_min)


""" ------------------------------------------- """
""" ---- pair-wise distance matrix related ---- """
""" ------------------------------------------- """


# def dist_pair(coord):
#     # not used, for unable to consider pbc
#     """
#     Calculate the pair-wise distance matrix.
#     :param coord: (np.array, (num_a, 3)) coordinates of atoms
#     :return: (np.array, (num_a, num_a)) pair-wise distance matrix
#     """
#     num_atom = coord.shape[0]
#     coord_m_1 = np.tile(coord, (1, num_atom))  # (num_a, 3 * num_a)
#     coord_m_2 = coord.reshape(1, num_atom * 3).repeat(num_atom, axis=0)
#     r2 = coord_m_1 - coord_m_2
#     mat = np.ones((num_atom, num_atom))
#     for i in range(num_atom):
#         coord_r = r2[:, i * 3:(i + 1) * 3]
#         mat[:, i] = np.sqrt(np.sum(np.square(coord_r), axis=1))
#     return mat


def min_pair_dist_mat(pmg_struct):
    try:
        mat = pmg_struct.distance_matrix
    except np.linalg.LinAlgError:
        return False
    mat += np.eye(N=mat.shape[0], dtype=float) * 999.0
    return mat


class DistFilter:
    def __init__(self, atom_num_list, dist=1.5, cryspy_dict=False):
        if atom_num_list is None:
            self._load_atom_num()
        else:
            self.atom_num_list = atom_num_list
        self.num_species = len(self.atom_num_list)
        if cryspy_dict:
            self._load_cryspy_dist()
        else:
            self._load_uniform_dist(dist=dist)
        ids = np.expand_dims(np.arange(0, self.num_species), axis=0)
        ids_1 = np.repeat(ids, self.num_species, axis=1).flatten()
        ids_2 = np.repeat(ids, self.num_species, axis=0).flatten()
        self.filter = []
        block_list = []
        if self.num_species == 1:
            self.filter = np.ones((self.atom_num_list[0], self.atom_num_list[0])) * self.min_dist_list[0][0]
        else:
            for i in range(self.num_species ** 2):
                block = np.ones((self.atom_num_list[ids_1[i]], self.atom_num_list[ids_2[i]]))
                block *= self.min_dist_list[ids_1[i]][ids_2[i]]
                block_list.append(block)
                if (i != 0) & ((i + 1) % self.num_species == 0):
                    self.filter.append(np.concatenate(block_list, axis=1))
                    block_list = []
            self.filter = np.concatenate(self.filter, axis=0)
        return

    def _load_atom_num(self):
        config = ConfigParser()
        config.read('./cryspy.in')
        self.atom_num_list = np.array(config['structure']['nat'].split(' ')).astype(int)
        return

    def _load_cryspy_dist(self):
        config = ConfigParser()
        config.read('./cryspy.in')
        min_dist_list = []
        for option in config.options('structure'):
            if 'mindist' in option:
                min_dist_list.append(config['structure'][option].split(' '))
        self.min_dist_list = np.array(min_dist_list).astype(float)
        return

    def _load_uniform_dist(self, dist=1.5):
        self.min_dist_list = np.ones((self.num_species, self.num_species)) * dist
        return

    def __call__(self, pmg_struct, test_lattice=True, *args, **kwargs):
        """ Make sure that the coordinates are listed
            in the same sequence as atom_num_list does. """
        mat = min_pair_dist_mat(pmg_struct)
        if isinstance(mat, bool):
            return False
        if not np.all(mat > self.filter):
            return False
        if test_lattice:
            lat = pmg_struct.lattice
            if not (lat.alpha < 130) & (lat.beta < 130) & (lat.gamma < 130):
                return False
            if not (lat.a > 3) & (lat.b > 3) & (lat.c > 3):
                return False
        return True


if __name__ == '__main__':
    # from pymatgen.io.vasp import Poscar
    # poscar = Poscar.from_file('./NaCl_POS_init')

    import random

    _a = [2, 3, 4, 5]
    random.seed(4)
    aa = random.choices(_a)

    conf = ConfigParser()
    conf.read('./cryspy.in')

    df = DistFilter([8, 8])
    dd = df.filter

    elem = ['Na', 'Na', 'Cl', 'Cl', 'Cl']
    ele_num = np.unique(elem, return_counts=True)

    ssr = StructReader('Na8Cl8_POS_seed0').to_pymatgen()[0]
