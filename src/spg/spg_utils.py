import sympy as sym
import numpy as np
import pandas as pd
from itertools import permutations
from pyxtal.lattice import para2matrix
from pymatgen.core import Structure
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application

from src.spg.get_wyckoff_position import get_all_wyckoff_combination
from src.struct_utils import crystal_system_dict, get_bravais_lattice, perturb_frac_atom, frac_pbc, \
    unique_elements


sg_list = list(range(2, 230 + 1))


def is_var(wp_str):
    return ('x' in wp_str) or ('y' in wp_str) or ('z' in wp_str)


def make_list_str(wp):
    return ', '.join(wp) if isinstance(wp, list) else wp


def split_wp_pairs(wp_pairs):
    if isinstance(wp_pairs, str):
        wp_pairs = np.array(list(eval(wp_pairs))).flatten()
    elif isinstance(wp_pairs, tuple):
        wp_pairs = np.array(wp_pairs).flatten()
    if len(wp_pairs.shape) > 1:  # shape == 2
        num_p = wp_pairs.shape[0]
    else:
        num_p = int(len(wp_pairs) / 3)
    wp_pairs = np.split(wp_pairs.flatten(), num_p)
    return wp_pairs  # isinstance(wp_pairs, list) == True


class WPElementProcess:
    def __init__(self, wp_list_element):
        self.wp_list_element = wp_list_element
        self.fix, self.fix_num = [], 0
        self.var, self.var_form, self.var_idx, self.var_num = [], [], [0] * len(wp_list_element), 0
        return

    def _variable_in_wyckoff_positions(self):
        for i, wp in enumerate(self.wp_list_element):
            wp_str = ', '.join(wp)
            # wp represent a set of atoms sharing the same x, y, z as wps, such as ['x, y, z', '-x, -y, -z'].
            if is_var(wp_str):
                self.var_idx[i] = 1
            else:
                sites = split_wp_pairs(wp_str)
                sites = [tuple(s) for s in sites]
                self.fix += sites
                self.fix_num += len(wp)
        self.var_num = np.sum(self.var_idx)
        return

    def load_variables(self):
        self._variable_in_wyckoff_positions()
        return

    def variable_in_wyckoff_positions(self):
        self.load_variables()
        return self.var_num

    def _collect_all_variables(self, formula=True, modify=True):
        if self.var_num == 0:
            print('There is no variables in the current wp combination')
            return
        for i in np.where(np.array(self.var_idx))[0].astype(int):
            wp_f = self.wp_list_element[i]
            wp_f = make_list_str(wp_f)
            if modify:
                wp_f = modify_variables(wp_f)
            self.var_form.append(wp_f)
            if formula:
                wp_f = sympify(wp_f)
            self.var.append(wp_f)
        return

    def load_formulas(self, formula=True, modify=True):
        if self.var_num + self.fix_num == 0:
            self._variable_in_wyckoff_positions()
        self._collect_all_variables(formula, modify)
        return


def sympify(wp, lamb=True):
    x, y, z = sym.symbols('x y z')
    wp_str = make_list_str(wp)
    transformations = (standard_transformations + (implicit_multiplication_application,))
    wp_f = parse_expr(wp_str.replace("^", "**"), transformations=transformations)
    # cannot use sym.sympify, since '2x' exists;
    # sympy.parsing.mathematica.mathematica is deprecated;
    # from https://stackoverflow.com/questions/26249993
    if lamb:
        return sym.lambdify([x, y, z], wp_f)
    else:
        return wp_f


def modify_variables(wp_f):
    wp_str_f = make_list_str(wp_f)
    for v1, v2, v3 in permutations(('x', 'y', 'z')):
        if (v1 in wp_str_f) and (v2 not in wp_str_f) and (v3 not in wp_str_f):
            wp_str_f = wp_str_f.replace(v1, f'({v1} + 0.2 * {v2} + 0.2 * {v3})')
            # ['1/3, 2/3, z'] -> ['1/3, 2/3, (z + 0.2 * x + 0.2 * y)']
            break
        elif (v1 in wp_str_f) and (v2 in wp_str_f) and (v3 not in wp_str_f):
            replacement_dict = {v1: f'({v1} + 0.4 * {v3})',
                                v2: f'({v2} + 0.4 * {v3})'}
            wp_str_f = ''.join(replacement_dict.get(c, c) for c in wp_str_f)
            # ['1/3, -y, z'] -> ['1/3, -(y + 0.4 * x), (z+ 0.4 * x)']
            break
    return wp_str_f


def get_wp_list(num_species_atom, spg, wp):
    wyckoffs_dict, wyckoffs_max = get_all_wyckoff_combination(
        sg_list=sg_list, atom_num=num_species_atom)
    wp_list = wyckoffs_dict[spg]
    return wp_list[int(len(wp_list) * wp / wyckoffs_max)]


def get_wp_list2(wyckoffs_dict, wyckoffs_max, spg, wp):
    wp_list = wyckoffs_dict[spg]
    return wp_list[int(len(wp_list) * wp / wyckoffs_max)]


def choose_wp(num_atom, spg=None):
    spg = np.random.choice(sg_list) if spg is None else spg
    print(f'Current spg: {spg}')
    wyckoffs_dict, wyckoffs_max = get_all_wyckoff_combination(sg_list=sg_list, atom_num=num_atom)
    while len(wyckoffs_dict[spg]) == 0:
        print(f'No available Wycokff positions exist for spg {spg}, trying another one')
        spg = np.random.choice(sg_list)
        print(f'Current spg: {spg}')
    wp_idx = np.random.choice(list(range(wyckoffs_max)))
    return wp_idx, spg


def shift_spg(spg, wyckoffs_dict, spg_u=230, spg_l=2, logger=None):
    flag = 0
    if spg < spg_l:
        spg = spg_l
    if spg > spg_u:
        spg = spg_u
    while len(wyckoffs_dict[spg]) == 0:
        print('trying to shift')
        flag = 1
        if spg >= spg_u:
            spg = spg_l
        else:
            spg = spg + 1
        print(f'after shifting: {spg}')
    if flag == 1:
        if logger is not None:
            logger.record_anything5(operation=f'Space Group moves to', result=spg)
    return spg


def get_spg_list(wyckoffs_dict):
    spg_list = []
    for spg, wyc in wyckoffs_dict.items():
        if len(wyc) != 0:
            spg_list.append(spg)
    return spg_list


def spg_oh_vec(space_group_number):
    spg = np.zeros(shape=(len(sg_list),))
    spg[space_group_number - 2] = 1
    return spg


def perturb_decoded(decoded, form):
    decode_x, decode_y, decode_z = decoded
    # replacement_dict = {}
    new_decode = [float(decode_x), float(decode_y), float(decode_z)]
    for j, v in enumerate(['x', 'y', 'z']):
        if v in form:
            new_decode[j] = new_decode[j] + perturb_frac_atom()
            # replacement_dict[v] = str(new_decode[j])
    # new_pos = ''.join(replacement_dict.get(c, c) for c in form)  # ChatGPT
    sym_form = sympify(form)
    new_pos = sym_form(x=new_decode[0], y=new_decode[1], z=new_decode[2])
    new_pos = np.array(list(new_pos)).reshape(-1, 3)
    return new_pos


def wp_list_validity_test(wp_list):
    """ if there are two fixed positions being the same, the wp combination is invalid.
        len(wp_list) == # of elements;
        len(wp_list[0]) == DOF of coordinates for the 0-th element """
    wps = []
    for wp_element_list in wp_list:
        wps += wp_element_list
    fixed_wps = []
    for wp in wps:
        if not is_var(make_list_str(wp)):
            fixed_wps += wp
    if len(set(fixed_wps)) != len(fixed_wps):
        return False
    else:
        return True


class StandardSPGDecoder:
    def __init__(self):
        self.atom_list, self.atom_form, self.decoded_atom_list = [], [], []
        return

    def decode_free_atom(self, wp_formula, wp_formula_f, atom):
        wp_pairs = wp_formula(x=atom[0], y=atom[1], z=atom[2])  # isinstance(wp_pairs, tuple) == True
        wp_pairs = split_wp_pairs(wp_pairs)
        self.atom_list += wp_pairs
        self.atom_form += [wp_formula_f] * len(wp_pairs)
        self.decoded_atom_list += [list(atom) for _ in range(len(wp_pairs))]
        return

    def append_fixed_atom(self, site):
        self.atom_list.append(np.array(site))
        self.atom_form.append('fix')
        self.decoded_atom_list.append(['fix', 'fix', 'fix'])
        return

    def clear(self):
        self.atom_list, self.atom_form, self.decoded_atom_list = [], [], []
        return

    def build(self, elements):
        self.atom_list = np.vstack(self.atom_list).round(3)  # some close atoms are regarded as the same
        self.atom_list = frac_pbc(self.atom_list)
        self.decoded_atom_list = np.vstack(self.decoded_atom_list)
        atom_pd = pd.DataFrame({'pos_x': self.atom_list[:, 0],
                                'pos_y': self.atom_list[:, 1],
                                'pos_z': self.atom_list[:, 2],
                                'decode_x': self.decoded_atom_list[:, 0],
                                'decode_y': self.decoded_atom_list[:, 1],
                                'decode_z': self.decoded_atom_list[:, 2],
                                'form': self.atom_form, 'ele': elements})
        return atom_pd

    @staticmethod
    def build_atom_pd(atom_list, atom_form, decoded_atom_list, elements):
        atom_list = np.vstack(atom_list).round(3)  # some close atoms are regarded as the same
        atom_list = frac_pbc(atom_list)
        atom_list = np.vstack(split_wp_pairs(atom_list))
        decoded_atom_list = np.vstack(decoded_atom_list)
        atom_pd = pd.DataFrame({'pos_x': atom_list[:, 0],
                                'pos_y': atom_list[:, 1],
                                'pos_z': atom_list[:, 2],
                                'decode_x': decoded_atom_list[:, 0],
                                'decode_y': decoded_atom_list[:, 1],
                                'decode_z': decoded_atom_list[:, 2],
                                'form': atom_form, 'ele': elements})
        return atom_pd

    @staticmethod
    def _shift_repeated_atoms(elements, atom_pd):
        for ele in elements:
            # search for repeated sites in the whole system
            indices_to_check = atom_pd.duplicated(subset=['pos_x', 'pos_y', 'pos_z'], keep=False)

            # only deal with one element each time
            filtered_indices = indices_to_check & (atom_pd['form'] != 'fix') & (atom_pd['ele'] == ele)
            result_indices = atom_pd[filtered_indices].index
            if len(result_indices) == 0:
                continue

            print('Dealing with overlapped atoms in decoding')
            atoms_to_check = atom_pd.loc[result_indices]
            result_formulas_indices = atoms_to_check.drop_duplicates(
                ['form', 'decode_x', 'decode_y', 'decode_z']).index

            for i in result_formulas_indices:
                form, decode_x, decode_y, decode_z = atom_pd['form'][i], atom_pd['decode_x'][i], \
                                                        atom_pd['decode_y'][i], atom_pd['decode_z'][i]

                new_pos = perturb_decoded(decoded=(decode_x, decode_y, decode_z), form=form)

                # we must substitute all related atoms simultaneously to maintain the symmetry
                formula_filter = (atom_pd['form'] == form) & (atom_pd['ele'] == ele) & \
                                 (atom_pd['decode_x'] == decode_x) & (atom_pd['decode_y'] == decode_y) & \
                                 (atom_pd['decode_z'] == decode_z)
                pos_form_ids = atom_pd[formula_filter].index
                atom_pd.loc[pos_form_ids, 'pos_x'] = new_pos[:, 0]
                atom_pd.loc[pos_form_ids, 'pos_y'] = new_pos[:, 1]
                atom_pd.loc[pos_form_ids, 'pos_z'] = new_pos[:, 2]

                print(f'Atoms generated with the wp {form} have been updated')

        atom_list = atom_pd[['pos_x', 'pos_y', 'pos_z']].to_numpy()
        atom_list = frac_pbc(atom_list).round(4)
        atom_pd[['pos_x', 'pos_y', 'pos_z']] = atom_list
        atom_list = split_wp_pairs(atom_list)
        return atom_list, atom_pd

    def shift_repeated_atoms(self, elements, atom_pd):
        atom_list, atom_pd = self._shift_repeated_atoms(elements, atom_pd)
        indices_to_check = atom_pd.duplicated(subset=['pos_x', 'pos_y', 'pos_z'], keep=False)
        while sum(indices_to_check) != 0:
            atom_list, atom_pd = self._shift_repeated_atoms(elements, atom_pd)
            indices_to_check = atom_pd.duplicated(subset=['pos_x', 'pos_y', 'pos_z'], keep=False)
        return atom_list


class Random2Wyckoff:
    def __init__(self, wyckoffs_dict, wyckoffs_max, species_list, lat_para, pos, spg, wp, logger=None,
                 modify=True, jump=True):
        self.species_list = species_list
        self.elements, self.num_species_atom = unique_elements(self.species_list)
        self.lat_para = lat_para
        self.pos = pos
        self.crystal_system = get_bravais_lattice(spg)
        self.crystal_system = crystal_system_dict()[self.crystal_system]
        self.logger = logger
        self.wp_list = self._get_valid_wp_list(wyckoffs_dict, wyckoffs_max, spg, wp)
        self.modify = modify  # whether to modify wp formula when decode
        self.jump = jump  # whether to jump to the next atom if current wp does not contain variables
        self.decoder = StandardSPGDecoder()
        self.used_decode = {element: [] for element in self.elements}
        return

    def _get_valid_wp_list(self, wyckoffs_dict, wyckoffs_max, spg, wp):
        flag = 0
        wp_0 = wp
        wp_unit = int(wyckoffs_max / len(wyckoffs_dict[spg]))
        if wp < 0:
            wp = 0
        if wp >= wyckoffs_max:
            wp = wyckoffs_max - 1
        wp_list = get_wp_list2(wyckoffs_dict, wyckoffs_max, spg, wp)
        while not wp_list_validity_test(wp_list):
            print('trying to shift wp')
            flag = 1
            if wp >= wyckoffs_max - 1:
                wp = 0
            else:
                wp = wp + wp_unit
            print(f'after shifting: {wp}')
            wp_list = get_wp_list2(wyckoffs_dict, wyckoffs_max, spg, wp)
        if flag == 1:
            if self.logger is not None:
                self.logger.record_anything5(operation=f'Wycokff position index moves from {wp_0} to', result=wp)
        return wp_list

    def _wy_pos(self):
        # independent = sum([len(wp_list_element) for wp_list_element in self.wp_list])
        # pos = np.array_split(self.pos, independent, axis=0)
        # pos = [np.mean(pos, axis=0) for pos in pos]  # use mean to prevent too random positions
        pos = self.pos
        count = 0
        for i, wp_list_element in enumerate(self.wp_list):
            element = self.elements[i]
            for wp_site_f in wp_list_element:  # wp_site: list of dependent sites
                pos_to_fill = pos[count]
                if self.modify:
                    wp_site_f = modify_variables(wp_site_f)
                wp_site_f = make_list_str(wp_site_f)
                wp_site = sympify(wp_site_f)
                if is_var(wp_site_f):
                    self.decoder.decode_free_atom(wp_site, wp_site_f, pos_to_fill)
                    self.used_decode[element].append(pos_to_fill)
                    count += 1
                else:
                    sites = split_wp_pairs(wp_site_f)
                    sites = [tuple(s) for s in sites]
                    for site in sites:
                        self.decoder.append_fixed_atom(site)
                    if self.jump:
                        count += 1
        return self._check_and_shift()

    def _check_and_shift(self):
        fixed_test = np.unique(self.decoder.atom_form)
        if (len(fixed_test) == 1) and (fixed_test[0] == 'fix'):  # only containing fixed sites
            return self.decoder.atom_list
        atom_pd = self.decoder.build(self.species_list)
        return self.decoder.shift_repeated_atoms(self.elements, atom_pd)

    def _wy_lat_param(self):
        a = self.lat_para[0]
        b = self.lat_para[1] if self.crystal_system['b'] != 'a' else a
        c = self.lat_para[2] if self.crystal_system['c'] != 'a' else a
        alpha = self.lat_para[3] if self.crystal_system['alpha'] is None else self.crystal_system['alpha']
        beta = self.lat_para[4] if self.crystal_system['beta'] is None else self.crystal_system['beta']
        gamma = self.lat_para[5] if self.crystal_system['gamma'] is None else self.crystal_system['gamma']
        return [a, b, c, alpha, beta, gamma]

    def get_pmg_struct(self):
        pos = self._wy_pos()
        lat_param = self._wy_lat_param()
        lat = para2matrix(cell_para=lat_param, radians=False, format='lower')
        default_lat = 8
        if np.isnan(np.sum(lat)):  # para2matrix can lead to nan
            lat[np.isnan(lat)] = default_lat
        pmg_struct = Structure(lattice=lat, species=self.species_list,
                               coords=pos, coords_are_cartesian=False)
        return pmg_struct


# class Wyckoff2xyz:
#     def __init__(self, wyckoffs_dict, wyckoffs_max, elements, encoded_pos, spg, wp, modify=True):
#         self.wp_list = get_wp_list2(wyckoffs_dict, wyckoffs_max, spg, wp)
#         self.elements = elements
#         self.modify = modify
#         self.encoded_pos = encoded_pos
#         return
#
#     def decode_formula(self):
#         for i, wp_list_element in self.wp_list:
#             element = self.elements[i]
#             for wp_site_f in wp_list_element:
#                 if self.modify:
#                     wp_site_f = modify_variables(wp_site_f)
#                 wp_site_f = make_list_str(wp_site_f)
#                 if not is_var(wp_site_f):
#                     continue
#                 wp_site = sympify(wp_site_f, lamb=False)
#                 for
#                     # TODO


    # def _decode_formula_element(self, wp_list_element, pos):
