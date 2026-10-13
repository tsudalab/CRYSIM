from copy import deepcopy
from logzero import logger as log_printer

import sympy as sym
import numpy as np
import pandas as pd
from itertools import permutations
from functools import partial
from pyxtal.lattice import para2matrix
from pyxtal.symmetry import Group, Wyckoff_position
from pyxtal.util import get_symmetrized_pmg
from pymatgen.core import Structure
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application

from fractions import Fraction
from pymatgen.symmetry.analyzer import SpacegroupOperations, SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.core.operations import SymmOp
import spglib

from crysim.src import DEFAULT_NUMPY_DTYPE
from crysim.src.spg.get_wyckoff_position import get_all_wyckoff_combination
from crysim.src.struct_utils import crystal_system_dict, get_crystal_lattice, perturb_frac_atom, frac_pbc, \
    unique_elements, add_vacuum_to_pmg, dim2pbc, pyx_init_to_pmg, map_species_to_int, min_pair_dist_mat, _hard_min_mid


def sg_list(dim):
    if dim == 3:
        return list(range(1, 230 + 1))
    elif dim == 2:
        init_2d = list(range(2, 80 + 1))
        init_2d.remove(2)
        for i in range(8, 18 + 1):
            init_2d.remove(i)  # remove lattice in which alpha or beta is not 90 degree
        return init_2d
    else:
        raise NotImplementedError


def regular_sg_list(dim):
    # exclude monoclinic
    if dim == 3:
        return list(range(16, 230 + 1))
    elif dim == 2:
        return list(range(19, 80 + 1))
    else:
        raise NotImplementedError


def is_var(wp_str):
    return ('x' in wp_str) or ('y' in wp_str) or ('z' in wp_str)


def make_list_str(wp):
    return ', '.join(wp) if isinstance(wp, list) else wp


def dof_compute(wp_str):  # ex.: ['x, y, z', '-x, -y, -z']
    dof = 0
    wp_str = ', '.join(wp_str)
    if 'x' in wp_str:
        dof += 1
    if 'y' in wp_str:
        dof += 1
    if 'z' in wp_str:
        dof += 1
    return dof


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


def analyze_wpc_list_element(wpc_list_element, modify=True):
    usable_wpc_list_element, ind_site_num = [], 0
    for i, wp in enumerate(wpc_list_element):
        wp_str = ', '.join(wp)
        # wp represent a set of atoms sharing the same x, y, z as wps, such as ['x, y, z', '-x, -y, -z'].
        if is_var(wp_str):
            wp_form = make_list_str(wp)
            if modify:
                wp_form = modify_variables(wp_form)
            wp_f = sympify(wp_form)
            usable_wpc_list_element.append(('f', wp_form, wp_f))
            ind_site_num += 1
        else:
            sites = split_wp_pairs(wp_str)
            sites = [tuple(s) for s in sites]
            usable_wpc_list_element.append(('s', sites))
    return usable_wpc_list_element, ind_site_num


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


def wpc_idx_to_specific_wpc_of_sg(wpc_list_len, wp, wyckoffs_max):
    return int(wpc_list_len * wp / wyckoffs_max)


def get_wp_list(num_species_atom, spg, wpc, dim, to_str):
    wyckoffs_dict, wyckoffs_max = get_all_wyckoff_combination(
        sg_list=sg_list(dim=dim), atom_num=num_species_atom, mode='pyx', dim=dim)
    wpc_list = wyckoffs_dict[spg]
    sampled_wpc = wpc_list[wpc]
    if to_str:
        sampled_wpc = convert_pyx_wpc_to_str(spg, sampled_wpc, dim)
    return sampled_wpc


def get_wp_list2(wyckoffs_dict, spg, wpc, dim, to_str):
    wpc_list = wyckoffs_dict[spg]
    sampled_wpc = wpc_list[wpc]
    if to_str:
        sampled_wpc = convert_pyx_wpc_to_str(spg, sampled_wpc, dim)
    return sampled_wpc


def choose_wp(num_atom, dim, spg=None):
    spg = np.random.choice(sg_list(dim)) if spg is None else spg
    print(f'Current spg: {spg}')
    wyckoffs_dict, wyckoffs_max = get_all_wyckoff_combination(sg_list=sg_list(dim=dim), atom_num=num_atom, mode='pyx', dim=dim)
    while len(wyckoffs_dict[spg]) == 0:
        print(f'No available Wycokff positions exist for spg {spg}, trying another one')
        spg = np.random.choice(sg_list(dim))
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


def check_sg(sg, num_species_atom, dim):
    sg = Group(sg, dim=dim)
    compat, has_dof = sg.check_compatible(numIons=num_species_atom)
    return compat


def get_spg_list(wyckoffs_dict):
    spg_list = []
    for spg, wyc in wyckoffs_dict.items():
        if len(wyc) != 0:
            spg_list.append(spg)
    return spg_list


def get_spg_list2(num_species_atom, dim, include_spg_list):
    compat_sg_list = []
    for i in sg_list(dim):
        compat = check_sg(i, num_species_atom, dim)
        if compat:
            compat_sg_list.append(i)
    if include_spg_list is not None:
        compat_sg_list = list(set(compat_sg_list).intersection(include_spg_list))
    return compat_sg_list


def spg_oh_vec(space_group_number, dim):
    spg = np.zeros(shape=(len(sg_list(dim)),))
    spg[space_group_number - 2] = 1
    return spg


# def perturb_decoded(decoded, form):
#     decode_x, decode_y, decode_z = decoded
#     # replacement_dict = {}
#     new_decode = [float(decode_x), float(decode_y), float(decode_z)]
#     for j, v in enumerate(['x', 'y', 'z']):
#         if v in form:
#             new_decode[j] = new_decode[j] + perturb_frac_atom()
#             # replacement_dict[v] = str(new_decode[j])
#     # new_pos = ''.join(replacement_dict.get(c, c) for c in form)  # ChatGPT
#     sym_form = sympify(form)
#     new_pos = sym_form(x=new_decode[0], y=new_decode[1], z=new_decode[2])
#     new_pos = np.array(list(new_pos)).reshape(-1, 3)
#     return new_pos


def perturb_decoded(decoded, form, keep_axis):
    decode_x, decode_y, decode_z = decoded
    sym_form = sympify(form)
    x = float(decode_x) + perturb_frac_atom() if 'x' not in keep_axis else float(decode_x)
    y = float(decode_y) + perturb_frac_atom() if 'y' not in keep_axis else float(decode_y)
    z = float(decode_z) + perturb_frac_atom() if 'z' not in keep_axis else float(decode_z)
    new_pos = sym_form(x=x, y=y, z=z)
    new_pos = np.array(list(new_pos)).reshape(-1, 3).astype(DEFAULT_NUMPY_DTYPE)
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
        self.atom_list += [np.array(s) for s in site]
        self.atom_form += ['fix' for _ in site]
        self.decoded_atom_list += [['fix', 'fix', 'fix'] for _ in site]
        return

    def clear(self):
        self.atom_list, self.atom_form, self.decoded_atom_list = [], [], []
        return

    def build(self, elements):
        return self.build_atom_pd(atom_list=self.atom_list, atom_form=self.atom_form,
                                  decoded_atom_list=self.decoded_atom_list, elements=elements)

    @staticmethod
    def build_atom_pd(atom_list, atom_form, decoded_atom_list, elements, use_pbc=True):
        atom_list = np.vstack(atom_list)
        atom_list = frac_pbc(atom_list) if use_pbc else atom_list
        decoded_atom_list = np.vstack(decoded_atom_list)
        atom_pd = pd.DataFrame({'pos_x': atom_list[:, 0],
                                'pos_y': atom_list[:, 1],
                                'pos_z': atom_list[:, 2],
                                'decode_x': decoded_atom_list[:, 0],
                                'decode_y': decoded_atom_list[:, 1],
                                'decode_z': decoded_atom_list[:, 2],
                                'form': atom_form, 'ele': elements})
        return atom_pd

    """ Following are functions for rattling structures """

    @staticmethod
    def _shift_repeated_atoms(elements, atom_pd, keep_axis, use_pbc):

        def _check_overlap_by_pd(_atom_pd, _r, _element):
            _atom_pd_check = deepcopy(_atom_pd)
            _atom_pd_check[['pos_x', 'pos_y', 'pos_z']] = _atom_pd_check[['pos_x', 'pos_y', 'pos_z']].round(_r)
            indices_to_check = _atom_pd_check.duplicated(subset=['pos_x', 'pos_y', 'pos_z'], keep=False)
            filtered_indices = indices_to_check & (_atom_pd_check['form'] != 'fix') & (_atom_pd_check['ele'] == _element)
            return _atom_pd_check[filtered_indices].index
        
        def _split_pd(_atom_pd):
            _ind_site_ids = []
            _checking_idx = 0
            while _checking_idx < len(_atom_pd):
                form = _atom_pd['form'][_checking_idx]
                if form == 'fix':
                    _checking_idx += 1
                else:
                    _ind_site_ids.append(_checking_idx)
                    _checking_idx += len(form.split(',')) // 3
            _ind_site_ids.append(len(_atom_pd))
            return _ind_site_ids

        frac_overlap_criterion = 2
        ind_site_ids = _split_pd(atom_pd)
        for ele in elements:
            # search for repeated sites in the whole system, but only deal with one element each time
            result_indices = _check_overlap_by_pd(atom_pd, frac_overlap_criterion, ele)
            num_overlap = len(result_indices)
            if num_overlap == 0:
                continue

            # enter the loop, until no atoms are overlapped with the criterion of round(frac_overlap_criterion)
            update_time = 0
            while num_overlap != 0:
                print(f'Dealing with overlapped atoms in decoding for {ele} for {update_time}-th coordinates update')
                atoms_to_check = atom_pd.loc[result_indices]
                # result_formulas_indices = atoms_to_check.drop_duplicates(
                #     ['form', 'decode_x', 'decode_y', 'decode_z']).index
                result_formulas_indices = []
                for j in atoms_to_check.index:
                    for s in range(len(ind_site_ids) - 1):
                        if ind_site_ids[s] <= j and j < ind_site_ids[s + 1]:
                            if ind_site_ids[s] not in result_formulas_indices:
                                result_formulas_indices.append(ind_site_ids[s])
                            break

                for i in result_formulas_indices:
                    form, decode_x, decode_y, decode_z = atom_pd['form'][i], atom_pd['decode_x'][i], \
                                                            atom_pd['decode_y'][i], atom_pd['decode_z'][i]

                    new_pos = perturb_decoded(decoded=(decode_x, decode_y, decode_z), form=form, keep_axis=keep_axis)

                    # we must substitute all related atoms simultaneously to maintain the symmetry
                    num_points_form = len(form.split(',')) // 3
                    pos_form_ids = np.arange(i, i + num_points_form)

                    formula_filter = (atom_pd['form'] == form) & (atom_pd['ele'] == ele) & \
                                     (atom_pd['decode_x'] == decode_x) & (atom_pd['decode_y'] == decode_y) & \
                                     (atom_pd['decode_z'] == decode_z)
                    if len(atom_pd[formula_filter]) != num_points_form:
                        log_printer.warning("Two identical ind_sites are filled into two identical wpc forms")
                    
                    atom_pd.loc[pos_form_ids, 'pos_x'] = new_pos[:, 0]
                    atom_pd.loc[pos_form_ids, 'pos_y'] = new_pos[:, 1]
                    atom_pd.loc[pos_form_ids, 'pos_z'] = new_pos[:, 2]

                    print(f'Atoms generated with the wp {form} have been updated')

                    # check overlap
                    result_indices = _check_overlap_by_pd(atom_pd, frac_overlap_criterion, ele)
                    num_overlap = len(result_indices)
                    
                update_time += 1
                if update_time == 10:
                    log_printer.warning("Reach the maximum trial number for purturbing overlapped atoms")
                    break

        atom_list = atom_pd[['pos_x', 'pos_y', 'pos_z']].to_numpy()
        # atom_list = frac_pbc(atom_list).round(4)
        atom_list = frac_pbc(atom_list) if use_pbc else atom_list
        atom_pd[['pos_x', 'pos_y', 'pos_z']] = atom_list
        atom_list = split_wp_pairs(atom_list)
        return atom_list, atom_pd

    def shift_repeated_atoms(self, elements, atom_pd, keep_axis='', use_pbc=True):
        atom_list, atom_pd = self._shift_repeated_atoms(elements, atom_pd, keep_axis=keep_axis, use_pbc=use_pbc)
        indices_to_check = atom_pd.duplicated(subset=['pos_x', 'pos_y', 'pos_z'], keep=False)
        update_time = 0
        while sum(indices_to_check) != 0:
            atom_list, atom_pd = self._shift_repeated_atoms(elements, atom_pd, keep_axis=keep_axis, use_pbc=use_pbc)
            indices_to_check = atom_pd.duplicated(subset=['pos_x', 'pos_y', 'pos_z'], keep=False)
            update_time += 1
            if update_time == 10:
                log_printer.warning("Reach the maximum trial number for purturbing overlapped atoms")
                break
        return atom_list


class Random2Wyckoff:
    def __init__(self, dim, wyckoffs_dict, species_list, lat_para, pos, spg, wp, 
                 is_add_vacuum=True, add_vacuum_to_pmg_fn=None, modify=False, jump=True):
        self.species_list = species_list
        self.elements, self.num_species_atom = unique_elements(self.species_list)
        self.lat_para = lat_para
        self.pos = pos
        self.dim = dim
        self.crystal_system = get_crystal_lattice(spg, dim=dim)
        self.crystal_system = crystal_system_dict(dim=dim)[self.crystal_system]
        self.is_add_vacuum = is_add_vacuum
        if add_vacuum_to_pmg_fn is not None:
            self.add_vacuum_to_pmg_fn = add_vacuum_to_pmg_fn
        else:
            self.add_vacuum_to_pmg_fn = partial(add_vacuum_to_pmg, consider_pbc_in_unit=True)
        # self.wp_list = self._get_valid_wp_list(wyckoffs_dict, wyckoffs_max, spg, wp)
        self.wp_list = get_wp_list2(wyckoffs_dict, spg, wp, dim=dim, to_str=True)
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
        wp_list = get_wp_list2(wyckoffs_dict, spg, wp, dim=self.dim, to_str=True)
        while not wp_list_validity_test(wp_list):
            print('trying to shift wp')
            flag = 1
            if wp >= wyckoffs_max - 1:
                wp = 0
            else:
                wp = wp + wp_unit
            print(f'after shifting: {wp}')
            wp_list = get_wp_list2(wyckoffs_dict, spg, wp, dim=self.dim, to_str=True)
        if flag == 1:
            log_printer.info(f'Wycokff position index moves from {wp_0} to {wp}')
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
                    # sites = [tuple(s) for s in sites]
                    self.decoder.append_fixed_atom(sites)
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
        if np.isnan(np.sum(lat)):  # para2matrix can lead to nan
            return None
        pmg_struct = Structure(lattice=lat, species=self.species_list,
                               coords=pos, coords_are_cartesian=False)
        if self.dim != 3 and self.is_add_vacuum:
            pmg_struct = self.add_vacuum_to_pmg_fn(pmg_struct, pbc=dim2pbc(self.dim))
        return pmg_struct


def pyx_wp_to_str(spg, pyx_wp, dim):
    wp = Wyckoff_position.from_group_and_letter(group=spg, letter=pyx_wp[1], dim=dim)
    wp_str = [op.as_xyz_str() for op in wp.ops]
    return wp_str


def pyx_wpc_to_str(spg, pyx_wpc, dim):
    """ form of pyx_wpc: [[(multiplicity, letter, dof), (multiplicity, letter, dof)], [], [], ...]"""
    wpc_str = []
    for pyx_wpc_ele in pyx_wpc:
        wpc_str_ele = []
        for pyx_wp in pyx_wpc_ele:
            wp_str = pyx_wp_to_str(spg=spg, pyx_wp=pyx_wp, dim=dim)
            wpc_str_ele.append(wp_str)
        wpc_str.append(wpc_str_ele)
    return wpc_str


def is_wpc_pyx_or_not(wpc):
    """ wpc = [wpc_ele, wpc_ele, ...], wpc_ele = [wp, wp, ...], wp is a list of str or a tuple(mul, letter, dof) """
    return not isinstance(wpc[0][0][0], str)


def convert_pyx_wpc_to_str(spg, wpc, dim):
    return pyx_wpc_to_str(spg, wpc, dim) if is_wpc_pyx_or_not(wpc) else wpc


def analyze_pmg_symmetry(pmg_struct):
    sym_struct, spg = get_symmetrized_pmg(pmg_struct)  # only support 3d
    wpc_dict = {ele.name: [] for ele in sym_struct.elements}
    sites = sym_struct.equivalent_sites
    wp_symbols = sym_struct.wyckoff_symbols
    for i, site in enumerate(sites):
        site_ele = site[0].specie.name
        wp_s = wp_symbols[i]
        wp = Wyckoff_position.from_group_and_letter(spg, wp_s, dim=3)
        wpc_dict[site_ele].append((wp.multiplicity, wp.letter, wp.get_dof()))
    return spg, wpc_dict


def extract_pyx_struct_wpc(pyx_struct):
    wpc_dict = {}
    for site in pyx_struct.atom_sites:
        element = site.specie
        if element not in wpc_dict:
            wpc_dict[element] = []
        wp = site.wp
        wpc_dict[element].append((wp.multiplicity, wp.letter, wp.get_dof()))
    return wpc_dict


def analyze_pyx_struct(pyx_struct, dim):
    if dim == 3:
        pmg_struct = pyx_struct.to_pymatgen()
    else:
        pmg_struct = pyx_init_to_pmg(pyx_struct)
    used_atoms = {}
    for site in pyx_struct.atom_sites:
        if site.specie not in used_atoms:
            used_atoms[site.specie] = [site.position]
        else:
            used_atoms[site.specie].append(site.position)
    # used_atoms = [site.position for site in pyx_struct.atom_sites]
    spg = pyx_struct.group.number
    wp = extract_pyx_struct_wpc(pyx_struct)
    return pmg_struct, used_atoms, spg, wp


class IndSitesFitter:
    def __init__(self, pmg_struct):
        self.pmg_struct = pmg_struct
        return
    
    def sort_coords(self, arr):
        arr = arr.round(5)
        return arr[np.lexsort([arr[:, 2], arr[:, 1], arr[:, 0]])]
    
    def build_struct_part(self, idx, f_wp):
        site = self.pmg_struct[idx].frac_coords
        symm_struct_part = f_wp(x=site[0], y=site[1], z=site[2])
        symm_coords = np.array(symm_struct_part).reshape(-1, 3)
        symm_coords = self.sort_coords(frac_pbc(symm_coords))
        return symm_coords
    
    def fit_wp(self, ids_group, wp):
        struct_part = np.vstack([self.pmg_struct[i].frac_coords for i in ids_group])
        struct_part = self.sort_coords(frac_pbc(struct_part))
        wp_str = [op.as_xyz_str() for op in wp.ops]
        f_wp = sympify(wp_str)
        if wp.get_dof() == 0:
            # print(f"only static")
            ind_idx = None
            symm_coords = self.build_struct_part(0, f_wp)
            symm_distance = np.sum(np.abs(symm_coords - struct_part))
        else:
            symm_distance = []
            for i in ids_group:
                symm_coords = self.build_struct_part(i, f_wp)
                symm_distance.append(np.sum(np.abs(symm_coords - struct_part)))
            ind_idx = ids_group[np.argmin(symm_distance)]
            symm_distance = np.min(symm_distance)
        return ind_idx, symm_distance

    def fit_ind_sites_from_pyx_init(self, elements, spg, wpc_list, dim):
        wp_dict, independent_sites_dict = {}, {}
        symm_distance_all = 0
        split_ids = [[i for i in range(len(self.pmg_struct)) if self.pmg_struct[i].species_string == e] for e in elements]
        for i, (ele, ele_ids) in enumerate(zip(elements, split_ids)):
            wpc_ele = wpc_list[i]
            starting_idx = ele_ids[0]
            if not ele in independent_sites_dict:
                independent_sites_dict[ele] = []
                wp_dict[ele] = []
            for wp in wpc_ele:
                multiplicity, letter, dof = wp
                wp_obj = Wyckoff_position.from_group_and_letter(spg, letter, dim)
                ids_group = [starting_idx + x for x in range(multiplicity)]
                starting_idx += multiplicity
                ind_idx, symm_distance = self.fit_wp(ids_group, wp_obj)
                ind_site = None if ind_idx is None else self.pmg_struct[ind_idx].frac_coords
                independent_sites_dict[ele].append(ind_site)
                wp_dict[ele].append((wp_obj.multiplicity, wp_obj.letter, wp_obj.get_dof()))
                symm_distance_all += symm_distance
        return independent_sites_dict, symm_distance_all, spg, wp_dict

    def fit_ind_sites_from_pmg_struct(self):
        if np.min(min_pair_dist_mat(self.pmg_struct)) < _hard_min_mid:
            log_printer.error(f"Unphysical structure with overlapped atoms")
            return None, None, None, None
        try:
            symm_struct = pmg_symmetrize_structure(self.pmg_struct)
            spg = symm_struct.get_space_group_info()[1]
            wp_dict, independent_sites_dict = {}, {}
        except Exception:
            log_printer.error(f"Error raised when deciding symmetry by spglib")
            return None, None, None, None
        symm_distance_all = 0
        groups_ids = symm_struct.equivalent_indices if spg != 1 else [[p] for p in range(len(self.pmg_struct))]
        for num, group in enumerate(groups_ids):
            ele = self.pmg_struct[group[0]].species_string
            if not ele in independent_sites_dict:
                independent_sites_dict[ele] = []
                wp_dict[ele] = []
            w_letter = symm_struct.wyckoff_letters[group[0]]
            wp = Wyckoff_position.from_group_and_letter(spg, w_letter, dim=3)
            if wp.multiplicity == len(group):
                ind_idx, symm_distance = self.fit_wp(group, wp)
            else:
                log_printer.error(f"Detected wrong wp letters, try other ones")
                possible_wp, group_size = [], len(group)
                for wp_idx in range(100):
                    try:
                        new_wp = Wyckoff_position.from_group_and_index(spg, wp_idx, dim=3)
                    except IndexError:
                        break
                    if new_wp.multiplicity > group_size:
                        continue
                    elif new_wp.multiplicity == group_size:
                        possible_wp.append(new_wp)
                    elif new_wp.multiplicity < group_size:
                        break
                possible_dist, possible_ind_idx = [], []
                if len(possible_wp) == 0:
                    log_printer.error(f"No compatible WP for this SG, return None")
                    return independent_sites_dict, 1e10, spg, wp_dict
                for p_wp in possible_wp:
                    ind_idx, symm_distance = self.fit_wp(group, p_wp)
                    possible_ind_idx.append(ind_idx)
                    possible_dist.append(symm_distance)
                best_idx = np.argmin(possible_dist)
                ind_idx, symm_distance = possible_ind_idx[best_idx], possible_dist[best_idx]
                print(f"wp moved from {wp.letter} to {possible_wp[best_idx].letter}")
                wp = possible_wp[best_idx]
            # log_printer.info(f"{num}/{len(symm_struct.equivalent_indices)} group have {symm_distance} distance")
            symm_distance_all += symm_distance
            ind_site = None if ind_idx is None else self.pmg_struct[ind_idx].frac_coords
            independent_sites_dict[ele].append(ind_site)
            wp_dict[ele].append((wp.multiplicity, wp.letter, wp.get_dof()))
        return independent_sites_dict, symm_distance_all, spg, wp_dict
    
    def fit(self, idx, elements, dim, input_spg=None, input_wpc_list=None):
        if dim < 3:
            independent_sites_dict, symm_distance_all, spg, wp_dict = self.fit_ind_sites_from_pyx_init(elements, input_spg, input_wpc_list, dim=dim)
            log_printer.info(f"{idx}-th struct, from pyx init: {spg}, {symm_distance_all}")
            return independent_sites_dict, spg, wp_dict
        if input_spg is None or input_wpc_list is None:
            independent_sites_dict, symm_distance_all, spg, wp_dict = self.fit_ind_sites_from_pmg_struct()
            log_printer.info(f"{idx}-th 3d struct, from spglib: {spg}, {symm_distance_all}")
            return independent_sites_dict, spg, wp_dict
        independent_sites_dict1, symm_distance_all1, spg1, wp_dict1 = self.fit_ind_sites_from_pmg_struct()
        independent_sites_dict2, symm_distance_all2, spg2, wp_dict2 = self.fit_ind_sites_from_pyx_init(elements, input_spg, input_wpc_list, dim=3)
        if independent_sites_dict1 is None:
            independent_sites_dict, spg, wp_dict = independent_sites_dict2, spg2, wp_dict2
        else:
            symm_distance_all1, symm_distance_all2 = np.round(symm_distance_all1, 5), np.round(symm_distance_all2, 5)
            if symm_distance_all1 < symm_distance_all2:
                independent_sites_dict, spg, wp_dict = independent_sites_dict1, spg1, wp_dict1
            else:  # including equal
                if symm_distance_all1 == symm_distance_all2:
                    if spg2 == 1:
                        independent_sites_dict, spg, wp_dict = independent_sites_dict1, spg1, wp_dict1
                    else:
                        independent_sites_dict, spg, wp_dict = independent_sites_dict2, spg2, wp_dict2
                else:
                    independent_sites_dict, spg, wp_dict = independent_sites_dict2, spg2, wp_dict2
        log_printer.info(f"{idx}-th struct, from spglib: {spg1}, {symm_distance_all1}, from pyx init: {spg2}, {symm_distance_all2}, choose {spg}")
        return independent_sites_dict, spg, wp_dict


def build_struct(dim, lattice, species, ind_pos_list, wpc_list, spg=None):
    pos_list = []
    for ind_pos_ele_list, wpc_ele_list in zip(ind_pos_list, wpc_list):
        for ind_pos, wpc in zip(ind_pos_ele_list, wpc_ele_list):
            if spg is not None:
                wpc = convert_pyx_wpc_to_str(spg, wpc, dim)
            wpc = make_list_str(wpc)
            wpc_f = sympify(wpc)
            pos_wpc = list(wpc_f(x=ind_pos[0], y=ind_pos[1], z=ind_pos[2]))
            for i in range(len(pos_wpc) // 3):
                pos_list.append(pos_wpc[3 * i: 3 * (i + 1)])
    pos_list = frac_pbc(np.array(pos_list))
    return Structure(lattice=lattice, species=species, coords=pos_list, coords_are_cartesian=False)


def pmg_symmetrize_structure(pmg_struct):
    # sga = SpacegroupAnalyzer(pmg_struct, symprec=1e-4, angle_tolerance=-1.0)
    sga = SpacegroupAnalyzer(pmg_struct, symprec=0.01)
    symm_struct = sga.get_symmetrized_structure()
    return symm_struct


def pmg_symmetrize_structure2(pmg_struct):
    """ from pymatgen.symmetry.analyzer.SpacegroupAnalyzer """
    def get_symmetry_operations(_pmg_struct, _symm_dict, cartesian=False):
        translations = np.array(
            [[float(Fraction(c).limit_denominator(1000)) for c in trans] for trans in _symm_dict["translations"]]
        )
        translations[np.abs(translations) == 1] = 0
        rotation = _symm_dict["rotations"]
        sym_ops = []
        mat = _pmg_struct.lattice.matrix.T
        inv_mat = np.linalg.inv(mat)
        for rot, trans in zip(rotation, translations, strict=True):
            if cartesian:
                rot = np.dot(mat, np.dot(rot, inv_mat))
                trans = np.dot(trans, _pmg_struct.lattice.matrix)
            op = SymmOp.from_rotation_and_translation(rot, trans)
            sym_ops.append(op)
        return sym_ops
    
    species_list = [s.species_string for s in pmg_struct]
    species_int_list = map_species_to_int(species_list)
    cell = (
        tuple(map(tuple, pmg_struct.lattice.matrix.tolist())),
        tuple(map(tuple, pmg_struct.frac_coords.tolist())),
        tuple(species_int_list),
    )
    symm_data = spglib.get_symmetry_dataset(cell=cell, symprec=0.01, angle_tolerance=5)  # pymatgen
    symm_dict = spglib.get_symmetry(cell=cell, symprec=0.01, angle_tolerance=5)
    spg_ops = SpacegroupOperations(
        symm_data.international,
        int(symm_data.number),
        get_symmetry_operations(pmg_struct, symm_dict),
    )
    symm_struct = SymmetrizedStructure(pmg_struct, spg_ops, symm_data.equivalent_atoms, symm_data.wyckoffs)
    return symm_struct
    