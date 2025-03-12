#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import pickle
from pathlib import Path
import numpy as np
import argparse
from logzero import logfile, logger
from ase.io import read
from ase.atoms import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import matgl
import matgl.ext.ase as mea
import re


class Component:
    def __init__(self, name, real_name):
        self.name = name
        self.real_name = real_name
        return


class EnergyEvaluator(Component):
    def __init__(self, name, real_name, struct_type):
        super().__init__(name, real_name)
        assert struct_type in ['ase']
        self.struct_type = struct_type
        return

    def build_atoms(self, struct):
        self.atoms = struct
        return

    def cal_energy(self):
        raise NotImplementedError

    def cal_relax(self):
        raise NotImplementedError


class EnergyM3GNET(EnergyEvaluator):
    def __init__(self):
        super().__init__(name='M3GNet', real_name='M3GNet', struct_type='ase')
        self.pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        return

    def cal_energy(self):
        # define the M3GNet calculator
        calc = mea.M3GNetCalculator(self.pot)
        # set up the calculator for atoms object
        self.atoms.set_calculator(calc)
        return self.atoms.get_potential_energy()

    def _cal_relax(self):
        relaxer = mea.Relaxer(potential=self.pot)
        relax_results = relaxer.relax(self.atoms, fmax=0.005)
        return relax_results

    def cal_relax(self):
        relax_results = self._cal_relax()
        # extract results
        final_structure = relax_results["final_structure"]
        final_energy = relax_results["trajectory"].energies[-1]
        return final_structure, final_energy


'''
structure optimization with MLP model and ASE
'''


def Get_Element_Num(elements):
    """ Using the Atoms.symples to Know Element&Num """
    element = []
    ele = {}
    element.append(elements[0])
    for x in elements:
        if x not in element:
            element.append(x)
    for x in element:
        ele[x] = elements.count(x)
    return element, ele


def Write_Contcar(output_dir, i, element, ele, lat, pos):
    '''Write CONTCAR'''
    f = open(f'{output_dir}/CONTCAR_{i}', 'w')
    f.write('ASE-DPKit-Optimization\n')
    f.write('1.0\n')
    for i in range(3):
        f.write('%15.10f %15.10f %15.10f\n' % tuple(lat[i]))
    for x in element:
        f.write(x + '  ')
    f.write('\n')
    for x in element:
        f.write(str(ele[x]) + '  ')
    f.write('\n')
    f.write('Direct\n')
    na = sum(ele.values())
    dpos = np.dot(pos, np.linalg.inv(lat))
    for i in range(na):
        f.write('%15.10f %15.10f %15.10f\n' % tuple(dpos[i]))
    f.close()


def Write_Outcar(output_dir, i, element, ele, volume, lat, pos, ene, force, stress, pstress):
    '''Write OUTCAR'''
    f = open(f'{output_dir}/OUTCAR_{i}', 'w')
    for x in element:
        f.write('VRHFIN =' + str(x) + '\n')
    f.write('ions per type =')
    for x in element:
        f.write('%5d' % ele[x])
    f.write('\nDirection     XX             YY             ZZ             XY             YZ             ZX\n')
    f.write('in kB')
    f.write('%15.6f' % stress[0])
    f.write('%15.6f' % stress[1])
    f.write('%15.6f' % stress[2])
    f.write('%15.6f' % stress[3])
    f.write('%15.6f' % stress[4])
    f.write('%15.6f' % stress[5])
    f.write('\n')
    ext_pressure = np.sum(stress[0] + stress[1] + stress[2]) / 3.0 - pstress
    f.write('external pressure = %20.6f kB    Pullay stress = %20.6f  kB\n' % (ext_pressure, pstress))
    f.write('volume of cell : %20.6f\n' % volume)
    f.write('direct lattice vectors\n')
    for i in range(3):
        f.write('%10.6f %10.6f %10.6f\n' % tuple(lat[i]))
    f.write('POSITION                                       TOTAL-FORCE(eV/Angst)\n')
    f.write('-------------------------------------------------------------------\n')
    na = sum(ele.values())
    for i in range(na):
        f.write('%15.6f %15.6f %15.6f' % tuple(pos[i]))
        f.write('%15.6f %15.6f %15.6f\n' % tuple(force[i]))
    f.write('-------------------------------------------------------------------\n')
    f.write('energy  without entropy= %20.6f %20.6f\n' % (ene, ene / na))
    enthalpy = ene + pstress * volume / 1602.17733
    f.write('enthalpy is  TOTEN    = %20.6f %20.6f\n' % (enthalpy, enthalpy / na))
    f.close()


# def read_stress_fmax():
#     pstress = 0
#     fmax = 0.01
#     # assert os.path.exists('./input.dat'), 'input.dat does not exist!'
#     try:
#         f = open('input.dat', 'r')
#     except:
#         assert os.path.exists('../input.dat'), ' now we are in %s, do not find ../input.dat' % (os.getcwd())
#         f = open('../input.dat', 'r')
#     lines = f.readlines()
#     f.close()
#     for line in lines:
#         if line[0] == '#':
#             continue
#         if 'PSTRESS' in line or 'pstress' in line:
#             pstress = float(line.split('=')[1])
#         if 'fmax' in line:
#             fmax = float(line.split('=')[1])
#     return fmax, pstress


def get_savable_structures(struct):
    if isinstance(struct, Atoms):
        struct = AseAtomsAdaptor.get_structure(struct)
    return Structure(lattice=struct.lattice, species=struct.species,
                     coords=struct.frac_coords, coords_are_cartesian=False)


def run_opt(i, n_steps, now_step, pop_size):
    '''Using the ASE&DP to Optimize Configures'''

    task_name = f"calypso_energy_{n_steps}_{pop_size}"
    logfile("./log")

    logger.info(f'======== Working on {now_step}-th iteration ========')

    # separately save OUTCAR
    output_dir = f"../output_{now_step}"
    Path(output_dir).mkdir(exist_ok=True)

    # os.system(f'mv OUTCAR_{i} OUTCAR_{i}-last')
    # try:
    #     os.rename(f'OUTCAR_{i}', f'OUTCAR_{i}-last')
    #     logger.info(f"Successfully move OUTCAR_{i} to OUTCAR_{i}-last")
    # except FileNotFoundError:
    #     logger.info(f'File OUTCAR_{i} does not exist')

    logger.info(f'======== Start to Optimize {i}-th Structures in {now_step}-th step ========')

    start = time.time()
    to_be_opti = read(f'POSCAR_{i}')
    e_cal = EnergyM3GNET()
    e_cal.build_atoms(to_be_opti)

    try:
        opti_ed, energy = e_cal.cal_relax()
    except RuntimeError:
        opti_ed, energy = to_be_opti, 999

    logger.info(f"[calculated energy]: {energy}")
    energy_dir = Path(f"./{task_name}.pkl")
    if not energy_dir.exists():
        logger.info(f"creating results file")
        data = {'energy': [energy],
                'init_struct': [get_savable_structures(to_be_opti)],
                'relaxed_struct': [get_savable_structures(opti_ed)]
                }
        with open(energy_dir, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"finish creating results file for {i}")
    else:
        logger.info(f"updating results file for {i}")
        with open(energy_dir, "rb") as f:
            data = pickle.load(f)
        energy_list, init_struct_list, relaxed_struct_list = \
            data['energy'], data['init_struct'], data['relaxed_struct']
        energy_list.append(energy)
        init_struct_list.append(get_savable_structures(to_be_opti))
        relaxed_struct_list.append(get_savable_structures(opti_ed))
        data['energy'], data['init_struct'], data['relaxed_struct'] = \
            energy_list, init_struct_list, relaxed_struct_list
        with open(energy_dir, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"finish updating results file for {i}")

    to_be_opti = opti_ed
    if isinstance(to_be_opti, Structure):
        to_be_opti = AseAtomsAdaptor.get_atoms(structure=to_be_opti)

    atoms_lat = to_be_opti.cell
    atoms_pos = to_be_opti.positions
    atoms_force = to_be_opti.get_forces()
    atoms_stress = to_be_opti.get_stress()
    # eV/A^3 to GPa
    atoms_stress = atoms_stress / (0.01 * 0.6242)
    atoms_symbols = to_be_opti.get_chemical_symbols()
    atoms_ene = to_be_opti.get_potential_energy()
    atoms_vol = to_be_opti.get_volume()
    element, ele = Get_Element_Num(atoms_symbols)

    Write_Contcar(output_dir, i=i, element=element, ele=ele, lat=atoms_lat, pos=atoms_pos)
    logger.info(f"Successfully save CONTCAR_{i} to {output_dir}")
    Write_Outcar(output_dir, i=i, element=element, ele=ele, volume=atoms_vol, lat=atoms_lat,
                 pos=atoms_pos, ene=atoms_ene, force=atoms_force,
                 stress=atoms_stress * -10.0, pstress=0)
    logger.info(f"Successfully save OUTCAR_{i} to {output_dir}")

    stop = time.time()
    _cwd = os.getcwd()
    _cwd = os.path.basename(_cwd)
    logger.info('%s is done, time: %s' % (_cwd, stop - start))


def main():
    parser = argparse.ArgumentParser(description='test-calypso')
    parser.add_argument('-i', '--struct-idx', type=int)
    parser.add_argument('-n', '--n-steps', type=int)
    parser.add_argument('-nn', '--now-step', type=int)
    parser.add_argument('-p', '--pop-size', type=int)
    args = parser.parse_args()
    ids = args.struct_idx
    _n_steps = args.n_steps
    _now_step = args.now_step
    _pop_size = args.pop_size
    for idx in range(1, ids + 1):
        run_opt(idx, _n_steps, _now_step, _pop_size)
    return


def analyze_results(log_file_path):
    energy_values = []

    # Regular expression to match the pattern "[calculated energy]: number"
    pattern = r'\[calculated energy\]:\s*(-?\d+\.?\d*)'

    # Read the file line by line
    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                energy_value = float(match.group(1))  # Convert to float
                energy_values.append(energy_value)

    return energy_values


if __name__ == '__main__':
    main()
    # _log_dir = '../../results/calypso-y6co51'
    # for i in range(5):
    #     e_values = analyze_results(f'{_log_dir}/srun-run-Y6Co51/task{i}/log')
    #     with open(f'calypso_e_results{i}', 'wb') as f:
    #         pickle.dump(e_values, f)
