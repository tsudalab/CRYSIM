import re
import pickle
# from itertools import product
import numpy as np
import pandas as pd
import spglib
from pymatgen.core import Structure, Lattice, Element, SymmOp
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from configparser import ConfigParser
from pyxtal.lattice import matrix2para
from logzero import logger as log_printer

from crysim.src import PROJECT_ROOT, DEFAULT_NUMPY_DTYPE


_symprec = 0.003
_mid_scale_lb = 0.8
_mid_scale_ub = 1.2

_vacuum_thickness = 15
_hard_min_mid = 1e-4


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


def get_system_name(elements, num_species_atom):
    system_name = []
    for i in range(len(elements)):
        system_name.append(elements[i])
        system_name.append(str(num_species_atom[i]))
    system_name = "".join(system_name)
    return system_name


def get_system_size(sys_name):
    system_size = re.findall(r'\d+', sys_name)
    system_size = sum([eval(i) for i in system_size])
    return system_size


def map_species_to_int(species_list):
    arr = np.array(species_list)
    _, idx, inverse = np.unique(arr, return_index=True, return_inverse=True)
    rank = np.argsort(np.argsort(idx))  # Re-map the inverse indices based on the first appearance
    result = rank[inverse] + 1
    return result


class Struct:
    def __init__(self, lattice, pos, species, cart, dim, 
                 potential_energy=None, forces=None, stress=None):
        self.lattice = lattice
        self.pos = pos
        self._sp = species
        self.cart = cart
        self.dim = dim
        self.pbc = dim2pbc(self.dim)
        
        # properties
        self.potential_energy = potential_energy
        self.forces = forces
        self.stress = stress
        return
    
    def get_positions(self):
        return self.pos

    @property
    def lattice_param(self):
        return matrix2para(self.lattice, radians=False)
    
    def get_cell(self):
        return self.lattice

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
        struct = array_to_pymatgen_struct(lattice=self.lattice, pos=self.pos, pbc=self.pbc,
                                          species_list=self.species, cart=self.cart)
        _, spg_num = SPG.struct_spg_estimate(struct)
        return spg_num

    @property
    def crystal_system(self):
        return get_crystal_lattice(self.space_group, dim=self.dim)
    
    def get_potential_energy(self):
        return self.potential_energy
    
    def get_forces(self):
        return self.forces
    
    def get_stress(self):
        return self.stress
    
    def structure(self):
        # to pymatgen.core.Structure
        return array_to_pymatgen_struct(lattice=self.get_cell(), 
                                        species_list=self.species, 
                                        pos=self.get_positions(),
                                        cart=self.cart, 
                                        pbc=self.pbc)
    
    def atoms(self):
        # to ase.Atoms
        pmg_struct = self.structure()
        return AseAtomsAdaptor.get_atoms(pmg_struct)


def array_to_pmg_for_training_and_recording(struct):
    if not isinstance(struct, Structure):
        struct = array_to_pymatgen_struct(lattice=struct.lattice, pos=struct.pos,
                                          species_list=struct.species, cart=False, pbc=struct.pbc)
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
        num = float(content)
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
                                              cart=self.cart, pbc=[1, 1, 1])
            struct_list.append(struct)
        return struct_list

    def to_struct(self):
        struct_list = []
        for i in range(len(self.poscar_list)):
            struct = Struct(lattice=self.poscar_list[i][0],
                            pos=self.poscar_list[i][1],
                            species=(self.ele_list[i], self.ele_num_list[i]),
                            cart=self.cart, dim=3)
            struct_list.append(struct)
        return struct_list


def array_to_pymatgen_struct(lattice, pos, species_list, cart, pbc):
    lattice = Lattice(matrix=lattice, pbc=pbc)
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
    dim = sum(struct.lattice.pbc)
    sites = [np.expand_dims(atom.frac_coords, axis=0) for atom in struct.sites]
    sites = np.concatenate(sites, axis=0)
    species_list = [atom.name for atom in struct.species]
    struct = Struct(lattice, sites, species_list, cart=False, dim=dim)  # Structure.frac_coord
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


def get_savable_structures(struct):
    if isinstance(struct, Atoms):
        struct = AseAtomsAdaptor.get_structure(struct)
    return Structure(lattice=struct.lattice, species=struct.species,
                     coords=struct.frac_coords, coords_are_cartesian=False)


def frac_pbc(site):
    flag = 1 if isinstance(site, tuple) else 0
    if flag:
        site = np.array(site)
    site = site.round(5).astype(DEFAULT_NUMPY_DTYPE)
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
    return (value * sign).astype(DEFAULT_NUMPY_DTYPE)


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


# https://matsci.org/t/putting-lattice-vectors-in-lower-triangle-form/47172
def pmg_struct_to_standard_form(pmg_struct):
    atoms = AseAtomsAdaptor.get_atoms(pmg_struct)
    cell = atoms.cell
    # transform the cell to standard form (lower triangular form)
    std = cell.standard_form()
    # pick off the transformation matrix
    rotation_matrix = std[1]
    # Back to pymatgen
    # define a symmetry operation with the rotation matrix
    myop = SymmOp.from_rotation_and_translation(rotation_matrix)
    #apply the rotation to the pymatgen structure
    pmg_struct.apply_operation(myop)
    return pmg_struct


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


def crystal_system_list(dim):
    if dim == 3:
        return ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal',
                'trigonal', 'hexagonal', 'cubic']
    elif dim == 2:
        init_2d = ['triclinic', 'monoclinic-inclined', 'monoclinic-orthogonal', 'orthorhombic', 'tetragonal',
                   'trigonal', 'hexagonal']
        init_2d.remove('triclinic')
        init_2d.remove('monoclinic-orthogonal')  # remove lattice in which alpha or beta is not 90 degree
        return init_2d
    else:
        raise NotImplementedError


def crystal_system_dict(dim):
    lat_type = {
        'triclinic': {'a': None, 'b': None, 'c': None, 'alpha': None, 'beta': None, 'gamma': None},
        'orthorhombic': {'a': None, 'b': None, 'c': None, 'alpha': 90, 'beta': 90, 'gamma': 90},
        'tetragonal': {'a': None, 'b': 'a', 'c': None, 'alpha': 90, 'beta': 90, 'gamma': 90},
        'trigonal': {'a': None, 'b': 'a', 'c': None, 'alpha': 90, 'beta': 90, 'gamma': 120},
        'hexagonal': {'a': None, 'b': 'a', 'c': None, 'alpha': 90, 'beta': 90, 'gamma': 120},
    }
    if dim == 3:
        lat_type.update(
            {'monoclinic': {'a': None, 'b': None, 'c': None, 'alpha': 90, 'beta': None, 'gamma': 90},
             'cubic': {'a': None, 'b': 'a', 'c': 'a', 'alpha': 90, 'beta': 90, 'gamma': 90},
            })
        return lat_type
    elif dim == 2:  # https://en.wikipedia.org/wiki/Layer_group
        lat_type.update(
            {'monoclinic-inclined': {'a': None, 'b': None, 'c': None, 'alpha': 90, 'beta': 90, 'gamma': None},
             'monoclinic-orthogonal': {'a': None, 'b': None, 'c': None, 'alpha': None, 'beta': 90, 'gamma': 90},
            })
        return lat_type
    else:
        raise NotImplementedError

    # ls_dict = {}
    # lat_para_name = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
    # for ls_name in crystal_system_list(dim):
    #     ls_dict[ls_name] = {name: None for name in lat_para_name}
    #
    # if 'monoclinic' in ls_dict.keys():
    #     ls_dict['monoclinic']['alpha'], ls_dict['monoclinic']['beta'] = 90, 90
    #     ls_dict['cubic']['alpha'], ls_dict['cubic']['beta'], \
    #         ls_dict['cubic']['gamma'], ls_dict['cubic']['b'], ls_dict['cubic']['c'] = 90, 90, 90, 'a', 'a'
    #
    # if 'monoclinic-inclined' in ls_dict.keys():
    #     ls_dict['monoclinic-inclined']['alpha'], ls_dict['monoclinic-inclined']['beta'] = 90, 90
    #     if 'monoclinic-orthogonal' in ls_dict.keys():
    #         ls_dict['monoclinic-orthogonal']['gamma'], ls_dict['monoclinic-orthogonal']['beta'] = 90, 90
    #
    # ls_dict['orthorhombic']['alpha'], ls_dict['orthorhombic']['beta'], \
    #     ls_dict['orthorhombic']['gamma'] = 90, 90, 90
    # ls_dict['tetragonal']['alpha'], ls_dict['tetragonal']['beta'], \
    #     ls_dict['tetragonal']['gamma'], ls_dict['tetragonal']['b'] = 90, 90, 90, 'a'
    # ls_dict['hexagonal']['alpha'], ls_dict['hexagonal']['beta'], \
    #     ls_dict['hexagonal']['gamma'], ls_dict['hexagonal']['b'] = 90, 90, 120, 'a'
    # ls_dict['trigonal']['alpha'], ls_dict['trigonal']['beta'], \
    #     ls_dict['trigonal']['gamma'], ls_dict['trigonal']['b'] = 90, 90, 120, 'a'
    #
    # return ls_dict


def spg_dict(dim):
    if dim == 3:
        return {'triclinic': [1, 3],
                'monoclinic': [3, 16],
                'orthorhombic': [16, 75],
                'tetragonal': [75, 143],
                'trigonal': [143, 168],
                'hexagonal': [168, 195],
                'cubic': [195, 231]}
    elif dim == 2:  # https://en.wikipedia.org/wiki/Layer_group
        return {'triclinic': [1, 3],
                'monoclinic-inclined': [3, 8],
                'monoclinic-orthogonal': [8, 19],
                'orthorhombic': [19, 49],
                'tetragonal': [49, 65],
                'trigonal': [65, 73],
                'hexagonal': [73, 81]}
    else:
        raise NotImplementedError


def get_crystal_lattice(space_group_number, dim):
    _spg_dict = spg_dict(dim)
    for cs in _spg_dict.keys():
        if (space_group_number >= _spg_dict[cs][0]) and (space_group_number < _spg_dict[cs][1]):
            return cs
        # if space_group_number in list(range(spg_dict[cs][0], spg_dict[cs][1])):
        #     return cs
    raise Exception("Wrong spg")


def get_spg_for_cs(cs, spg_list, dim):
    _spg_dict = spg_dict(dim)
    if spg_list is None:
        return np.array(list(range(_spg_dict[cs][0], _spg_dict[cs][1])))
    else:
        spg_list = np.array(spg_list)
        return spg_list[(spg_list >= _spg_dict[cs][0]) & (spg_list < _spg_dict[cs][1])]


def get_cs_list(spg_list, dim):
    _crystal_system_list = crystal_system_list(dim)
    cs_list = []
    for cs in _crystal_system_list:
        if len(get_spg_for_cs(cs, spg_list, dim)) != 0:
            cs_list.append(cs)
    return cs_list


# for 3d crystal since spglib does not support 2d
class SPG:
    def __init__(self, symprec=_symprec):
        self.symprec = symprec
        return

    def __call__(self, lattice, sites, species):
        spg = spglib.get_spacegroup(cell=(lattice, sites, species), symprec=self.symprec)
        try:
            space_group_number = int(re.findall(r'\((\d+)\)', spg)[0])
        except TypeError:
            space_group_number = 0
        return spg, space_group_number

    @staticmethod
    def _load_struct(pmg_struct):
        if not isinstance(pmg_struct, Structure):
            pmg_struct = array_to_pymatgen_struct(lattice=pmg_struct.lattice, pos=pmg_struct.pos,
                                                  species_list=pmg_struct.species, cart=pmg_struct.cart,
                                                  pbc=pmg_struct.pbc)
        lattice = pmg_struct.lattice.matrix
        sites = [np.expand_dims(atom.frac_coords, axis=0) for atom in pmg_struct.sites]
        sites = np.concatenate(sites, axis=0)
        element = [atom.name for atom in pmg_struct.species]
        species = pd.Series(element).astype('category').cat.codes.values
        return lattice, sites, species

    @staticmethod
    def _load_lat_param(pmg_struct):
        if not isinstance(pmg_struct, Structure):
            pmg_struct = array_to_pymatgen_struct(lattice=pmg_struct.lattice, pos=pmg_struct.pos,
                                                  species_list=pmg_struct.species, cart=pmg_struct.cart,
                                                  pbc=pmg_struct.pbc)
        return list(pmg_struct.lattice.abc) + list(pmg_struct.lattice.angles)

    @classmethod
    def struct_spg_estimate(cls, pmg_struct, prec=_symprec):
        spg_e = cls(symprec=prec)
        lattice, sites, species = spg_e._load_struct(pmg_struct)
        return spg_e(lattice, sites, species)

    @classmethod
    def struct_lattice_estimate(cls, pmg_struct, spg):
        a, b, c, alpha, beta, gamma = cls._load_lat_param(pmg_struct)
        a, b, c, alpha, beta, gamma = \
            np.round(a, 3), np.round(b, 3), np.round(c, 3), \
            np.round(alpha, 3), np.round(beta, 3), np.round(gamma, 3)
        _spg_dict = spg_dict(dim=3)
        if a == b and a == c and alpha == beta and alpha == gamma:
            if alpha == 90:
                return 'cubic'
            else:
                return 'trigonal'  # rhombohedral
        if a == b:
            if alpha == beta and alpha == gamma and alpha == 90:
                return 'tetragonal'
            elif alpha == beta and alpha == 90 and gamma == 120:
                if spg is None:
                    return 'hexagonal'
                else:
                    if (spg >= _spg_dict['trigonal'][0]) & (spg < _spg_dict['trigonal'][1]):
                        return 'trigonal'
                    elif (spg >= _spg_dict['hexagonal'][0]) & (spg < _spg_dict['hexagonal'][1]):
                        return 'hexagonal'
                    else:
                        raise Exception("The given spg is out of the scale "
                                        "of trigonal / hexagonal.")
        if alpha == gamma and alpha == 90:
            if alpha == beta:
                return 'orthorhombic'
            else:
                return 'monoclinic'
        else:
            return 'triclinic'


def get_best_spg(struct_list, e_list):
    spg_list = [SPG.struct_spg_estimate(struct)[1] for struct in struct_list]
    return spg_list[np.argmin(np.array(e_list))]


def get_best_crys_sys(struct_list, e_list):
    crys_sys_list = [SPG.struct_lattice_estimate(struct, None) for struct in struct_list]
    return crys_sys_list[np.argmin(np.array(e_list))]


def get_best_spg2(spg_list, wp_list, e_list):
    e_min_idx = np.argmin(np.array(e_list))
    return spg_list[e_min_idx], wp_list[e_min_idx]


def get_best_crys_sys2(spg_list, e_list, dim):
    e_min_idx = np.argmin(np.array(e_list))
    spg_min = spg_list[e_min_idx]
    return get_crystal_lattice(spg_min, dim=dim)


def pick(n, bias):
    """
    reference:
    https://stackoverflow.com/questions/5872153/random-selection-weighted-by-rank
    D. Whitley, Proceedings of the third international conference on
    Genetic algorithms, 1989, 116-121.
    """
    r = np.random.uniform(0, 1)
    f = n * (bias - np.sqrt(bias * bias - 4.0 * (bias - 1.0) * r))
    f = f / 2.0 / (bias - 1)
    return int(f)


""" ------------------------------------------- """
""" ---- pair-wise distance matrix related ---- """
""" ------------------------------------------- """


# average lattice length a, b, c of materials with number of atoms x <= 20, 20 < x <= 50,
# 50 < x <= 80, 80 < x <= 100, and x > 100,
# based on more than 150,000 materials from Material Project

# threshold = [20, 50, 80, 100, 'large']
# results_aver_a = [5.859180830043813, 7.391743829404799, 9.074192498061459,
#                   10.082653839712718, 11.30067570941344]
# results_aver_b = [6.125914597335389, 8.169390782111698, 9.637091463252037,
#                   10.56358747132569, 10.862278427360812]
# results_aver_c = [7.883718134982488, 10.779934874718478, 13.756538741124768,
#                   15.800706725323218, 20.107668351531817]

# threshold = [20, 50, 80, 'large']
# results_aver_a = [5.859180830043813, 7.391743829404799, 9.074192498061459, 10.890674465020503]
# results_aver_b = [6.125914597335389, 8.169390782111698, 9.637091463252037, 10.761735352798937]
# results_aver_c = [7.883718134982488, 10.779934874718478, 13.756538741124768, 18.65789173402984]

with open(PROJECT_ROOT + '/script/lattice-length-scale.pkl', 'rb') as f:
    ll_dict = pickle.load(f)
threshold = list(ll_dict.keys())
results_aver_a, results_aver_b, results_aver_c = [], [], []
for thre in threshold:
    results_aver_a.append(ll_dict[thre][0])
    results_aver_b.append(ll_dict[thre][1])
    results_aver_c.append(ll_dict[thre][2])

default_ang_lb = 50
default_ang_ub = 130
default_ang_mean = np.mean([default_ang_lb, default_ang_ub])


class LatticeLengthController:
    def __init__(self, num_atom):
        self.num_atom = num_atom
        self.scale = 1.5
        return

    def _min(self, i):
        lat_mean = np.mean([results_aver_a[i][0], results_aver_b[i][0], results_aver_c[i][0]])
        lat_std = self.scale * np.mean([results_aver_a[i][1], results_aver_b[i][1], results_aver_c[i][1]])
        a = np.floor(lat_mean - lat_std)
        # log_printer.info(f'Get minimum lattice length: {a}')
        return a

    def min(self, low_d=False):
        for i in range(len(threshold) - 1):
            if self.num_atom <= threshold[i]:
                return self._min(i) + int(low_d) * _vacuum_thickness
        return self._min(-1) + int(low_d) * _vacuum_thickness

    def _max(self, i):
        lat_max = np.max([results_aver_a[i][0], results_aver_b[i][0], results_aver_c[i][0]])
        lat_std = self.scale * np.mean([results_aver_a[i][1], results_aver_b[i][1], results_aver_c[i][1]])
        a = np.ceil(lat_max + lat_std)
        # log_printer.info(f'Get maximum lattice length: {a}')
        return a

    def max(self, low_d=False):
        for i in range(len(threshold) - 1):
            if self.num_atom < threshold[i]:
                return self._max(i) + int(low_d) * _vacuum_thickness
        return self._max(-1) + int(low_d) * _vacuum_thickness

    def mean(self, low_d=False):
        a = np.mean([self.min(low_d=low_d), self.max(low_d=low_d)])
        # log_printer.info(f'Get default lattice length: {a}')
        return a


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
    def __init__(self, atom_num_list, dist, cryspy_dict=False):
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
        
        self.llc = LatticeLengthController(num_atom=sum(atom_num_list))
        
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

    def _load_uniform_dist(self, dist):
        self.min_dist_list = np.ones((self.num_species, self.num_species)) * dist
        return

    def __call__(self, pmg_struct, test_lattice, compare_direction, pbc=None, *args, **kwargs):
        """ Make sure that the coordinates are listed
            in the same sequence as atom_num_list does. """
        mat = min_pair_dist_mat(pmg_struct)
        pbc = [1, 1, 1] if pbc is None else pbc
        vacuum = [1-pbc[0], 1-pbc[1], 1-pbc[2]]
        if isinstance(mat, bool):
            return False
        if compare_direction == '>':
            if not np.all(mat > self.filter):
                return False
        elif compare_direction == '<':
            if np.all(mat > self.filter):
                return False
        else:
            raise NotImplementedError
        if test_lattice:
            lat = pmg_struct.lattice
            if not (lat.alpha <= default_ang_ub) & (lat.beta <= default_ang_ub) & (lat.gamma <= default_ang_ub) & \
                   (lat.alpha >= default_ang_lb) & (lat.beta >= default_ang_lb) & (lat.gamma >= default_ang_lb):
                return False
            if not (lat.a >= self.llc.min(bool(vacuum[0]))) & (lat.b >= self.llc.min(bool(vacuum[1]))) & (lat.c >= self.llc.min(bool(vacuum[2]))) & \
                   (lat.a <= self.llc.max(bool(vacuum[0]))) & (lat.b <= self.llc.max(bool(vacuum[1]))) & (lat.c <= self.llc.max(bool(vacuum[2]))):
                return False
        return True

    @classmethod
    def test(cls, pmg_struct, test_lattice=True, compare_direction='both', min_scale=_mid_scale_lb, max_scale=_mid_scale_ub, pbc=None):
        species_list = pmg_struct.species
        elements, num_species_atom = unique_elements(species_list)
        mid_lb, mid_ub = mid_scale_for_system(element_list=elements, min_scale=min_scale, max_scale=max_scale)
        if compare_direction == '>':
            dist_ckr = cls(atom_num_list=num_species_atom, dist=mid_lb)
            return dist_ckr(pmg_struct=pmg_struct, test_lattice=test_lattice, compare_direction=compare_direction, pbc=pbc)
        elif compare_direction == '<':
            dist_ckr = cls(atom_num_list=num_species_atom, dist=mid_ub)
            return dist_ckr(pmg_struct=pmg_struct, test_lattice=test_lattice, compare_direction=compare_direction, pbc=pbc)
        elif compare_direction == 'both':
            dist_ckr_1 = cls(atom_num_list=num_species_atom, dist=mid_lb)
            dist_ckr_2 = cls(atom_num_list=num_species_atom, dist=mid_ub)
            pass_flag = (dist_ckr_1(pmg_struct=pmg_struct, test_lattice=test_lattice, compare_direction='>', pbc=pbc)) and \
                        (dist_ckr_2(pmg_struct=pmg_struct, test_lattice=test_lattice, compare_direction='<', pbc=pbc))
            return pass_flag
        else:
            raise NotImplementedError


def mid_scale_for_system(element_list, min_scale=_mid_scale_lb, max_scale=_mid_scale_ub):
    """ we assume the element list is unique """
    bond_length = (1, 4)
    cov_radii, ion_radii_minus, ion_radii_plus = [], [], []
    element_list = [Element(e) for e in element_list]
    for ele in element_list:
        ionic_radii = ele.ionic_radii
        for k, v in ionic_radii.items():
            if k < 0:
                ion_radii_minus.append(v)
            elif k > 0:
                ion_radii_plus.append(v)
            else:
                cov_radii.append(v)
                print("Attention! Neutral ionic radius")
        atomic_radius = ele.atomic_radius
        if atomic_radius is not None:
            cov_radii.append(atomic_radius)
    if len(ion_radii_plus) != 0 and len(ion_radii_minus) != 0:
        bond_length = (min(ion_radii_plus) + min(ion_radii_minus), max(ion_radii_plus) + max(ion_radii_minus))
    if len(cov_radii) != 0:
        bond_length = (min(cov_radii) * 2, max(cov_radii) * 2)
    min_bond_length, max_bond_length = min(bond_length), max(bond_length)
    return min_scale * min_bond_length, max_scale * max_bond_length


def mid_scale_for_system2(element_list, min_scale=_mid_scale_lb, max_scale=_mid_scale_ub):
    """ we assume the element list is unique """
    bond_length = (1, 4)
    element_list = [Element(e) for e in element_list]
    is_metal = [e.is_metal for e in element_list]
    ionic_system = True if (True in is_metal) and (False in is_metal) else False

    if ionic_system:
        ion_radii_minus, ion_radii_plus = [], []
        for ele in element_list:
            ionic_radii = ele.ionic_radii
            for k, v in ionic_radii.items():
                if k < 0:
                    ion_radii_minus.append(v)
                elif k > 0:
                    ion_radii_plus.append(v)
                else:
                    raise Exception("Attention! Neutral ionic radius")
        if len(ion_radii_plus) != 0 and len(ion_radii_minus) != 0:
            bond_length = (min(ion_radii_plus) + min(ion_radii_minus), max(ion_radii_plus) + max(ion_radii_minus))

    else:
        cov_radii = []
        for ele in element_list:
            atomic_radius = ele.atomic_radius
            if atomic_radius is not None:
                cov_radii.append(atomic_radius)
        if len(cov_radii) != 0:
            bond_length = (min(cov_radii) * 2, max(cov_radii) * 2)

    min_bond_length, max_bond_length = min(bond_length), max(bond_length)
    return min_scale * min_bond_length, max_scale * max_bond_length


""" -------------------------------------------- """
""" ---- low-dimensional structures related ---- """
""" -------------------------------------------- """


def dim2pbc(dim):
    if dim == 3:
        return [1, 1, 1]
    elif dim == 2:
        return [1, 1, 0]
    elif dim == 1:
        return [1, 0, 0]
    elif dim == 0:
        return [0, 0, 0]
    else:
        raise Exception('Not supported dimension')


# from https://pyxtal.readthedocs.io/en/stable/_modules/pyxtal/lattice.html#
# Note: the initial structure cannot be recovered if atoms are at z=0 and z=1, both of which will be regarded as z=0 due to PBC.
def add_vacuum(frac_coords, lat_mat, consider_pbc_in_unit, vacuum=_vacuum_thickness, pbc=None):
    if pbc is None:
        pbc = [0, 0, 0]
    lat_mat = lat_mat.copy()
    if consider_pbc_in_unit:
        frac_coords_0 = frac_coords.copy()  # ATTENTION! pbc is not considered in pyxtal, which is not correct
        frac_coords = frac_pbc(frac_coords.copy())
        if not np.all(frac_coords_0 == frac_coords):
            print("PBC tuning activated")
    else:
        frac_coords = frac_coords.copy()
    if not lat_mat.flags.writeable:
        lat_mat.flags.writeable = True
    absolute_coords = np.dot(frac_coords, lat_mat)
    for i, a in enumerate(pbc):
        if not a:
            ratio = 1 + vacuum / np.linalg.norm(lat_mat[i])
            lat_mat[i] *= ratio
            absolute_coords[:, i] += vacuum / 2
    frac_coords = np.dot(absolute_coords, np.linalg.inv(lat_mat))
    return frac_coords, lat_mat


def add_vacuum_to_pmg(pmg_struct, consider_pbc_in_unit, vacuum=_vacuum_thickness, pbc=None):
    if pbc is None:
        pbc = [1, 1, 0]
    frac_coords, lat_mat = pmg_struct.frac_coords, pmg_struct.lattice.matrix
    fc, lm = add_vacuum(frac_coords=frac_coords, lat_mat=lat_mat, consider_pbc_in_unit=consider_pbc_in_unit, vacuum=vacuum, pbc=pbc)
    if pbc == [1, 1, 0] and not consider_pbc_in_unit:  # squeeze the layer to prevent unphysical thickness for 2d
        fc, lm = reduce_vacuum(frac_coords=fc, lat_mat=lm, vacuum=vacuum, pbc=pbc)
        fc, lm = add_vacuum(frac_coords=fc, lat_mat=lm, consider_pbc_in_unit=False, vacuum=vacuum, pbc=pbc)
    return Structure(lattice=lm, species=pmg_struct.species, coords=fc, coords_are_cartesian=False)


def pyx_init_to_pmg(pyx_struct):
    lat_mat = pyx_struct.lattice.matrix
    frac_coords = np.concatenate([i.coords.reshape(-1, 3) for i in pyx_struct.atom_sites])
    pmg_struct = Structure(lattice=lat_mat, coords=frac_coords, coords_are_cartesian=False,
                           species=get_species_list(get_system_name(pyx_struct.species, pyx_struct.numIons)))
    return pmg_struct


def add_vacuum_to_pyxtal_init(pyx_struct, vacuum=_vacuum_thickness, pbc=None):
    pmg_struct = pyx_init_to_pmg(pyx_struct)
    return add_vacuum_to_pmg(pmg_struct, consider_pbc_in_unit=True, vacuum=vacuum, pbc=pbc)


def reduce_vacuum(frac_coords, lat_mat, vacuum=_vacuum_thickness, pbc=None):
    if pbc is None:
        pbc = [0, 0, 0]
    lat_mat = lat_mat.copy()
    frac_coords_0 = frac_coords.copy()
    frac_coords = frac_pbc(frac_coords.copy())
    if not lat_mat.flags.writeable:
        lat_mat.flags.writeable = True
    if not frac_coords.flags.writeable:
        frac_coords.flags.writeable = True
    absolute_coords = np.dot(frac_coords, lat_mat)
    frac_coords_patch = {}
    for i, a in enumerate(pbc):
        frac_coords_patch[i] = None
        if not a:
            min_thickness = np.max(absolute_coords[:, i]) - np.min(absolute_coords[:, i])

            # lattice
            init_lat_len = np.linalg.norm(lat_mat[i])
            if vacuum / init_lat_len >= 1:
                log_printer.info('Cut too much vacuum, ignore input; cut maximum possible')
                vacuum = init_lat_len - min_thickness - 0.5  # leave 0.5 angstrom space in lat
            ratio = 1 - vacuum / init_lat_len
            lat_mat[i] *= ratio

            # coords
            if np.any(init_lat_len - vacuum <= min_thickness):
                log_printer.info(f'Atoms inside vacuum to be cut; squeeze in the axis {i}')
                max_frac, min_frac = np.max(frac_coords[:, i]), np.min(frac_coords[:, i])
                ratio_frac = 0.8 / (max_frac - min_frac)  # leave 0.2 frac space for coords
                frac_coords_patch[i] = ratio_frac * (frac_coords[:, i] - min_frac) + 0.1  # above 0.1, below 0.1
            else:
                absolute_coords[:, i] -= vacuum / 2

    frac_coords = np.dot(absolute_coords, np.linalg.inv(lat_mat))
    for i, fc in frac_coords_patch.items():
        if fc is not None:
            frac_coords[:, i] = fc
    return np.round(frac_coords, 5), lat_mat


def reduce_vacuum_from_pmg(pmg_struct, vacuum=_vacuum_thickness, pbc=None):
    if pbc is None:
        pbc = [1, 1, 0]
    frac_coords, lat_mat = pmg_struct.frac_coords, pmg_struct.lattice.matrix
    fc, lm = reduce_vacuum(frac_coords=frac_coords, lat_mat=lat_mat, vacuum=vacuum, pbc=pbc)
    return Structure(lattice=lm, species=pmg_struct.species, coords=fc, coords_are_cartesian=False)


def reduce_vacuum_from_pyxtal(pmg_struct, pbc=None):
    return reduce_vacuum_from_pmg(pmg_struct, vacuum=15, pbc=pbc)


def compute_thickness_for_2d(pmg_struct):
    c = pmg_struct.lattice.c
    z_frac = pmg_struct.frac_coords[:, 2]
    thickness = np.max(z_frac) - np.min(z_frac)
    if thickness < 0.5:
        return thickness * c
    else:  # the largest distance among z-coords is seen as the vacuum layer
        z_frac = np.sort(z_frac)
        z_diff = np.diff(z_frac)
        z_diff = np.append(z_diff, z_frac[0] + 1.0 - z_frac[-1])
        thickness_vacuum = np.max(z_diff)
        thickness_new = 1 - thickness_vacuum
        log_printer.info(f"Reduce thickness from {thickness} to {thickness_new} by considering pbc")
        return thickness_new * c
