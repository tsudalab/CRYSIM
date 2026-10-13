from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.outputs import Vasprun, Oszicar
from pymatgen.core import Structure
from ase import Atoms
from ase.io import read
import yaml
import pandas as pd
import numpy as np
import bz2
import pickle
from pathlib import Path
import os
import time
import subprocess
import signal
from logzero import logger as log_printer

from crysim.src.registry import EnergyEvaluatorName
from crysim.src.evaluator.e_eval import EnergyEvaluator
from crysim.src.struct_utils import Struct, pymatgen_struct_to_array


general_settings = {
    'gga': 'PE', 'algo': 'Fast', 'lreal': 'Auto', 'prec': 'Accurate',
    'ediff': 1e-5,
    'nelm': 60, 'ismear': 0, 'sigma': 0.1, 
    # 'ispin': 2,
    'kspacing': 0.3,
}
relax_incar = {
    'isif': 3, 'ibrion': 2, 'nsw': 200, 'encut': 520, 'ediffg': -0.02,
}
relax_incar.update(general_settings)
single_point_incar = {
    'isif': 2, 'ibrion': -1, 'nsw': 0, 'encut': 400,
}
single_point_incar.update(general_settings)


hubbard_u = {
    'Co': {'L': 2, 'U': 3.32, 'J': 0},
    'Cr': {'L': 2, 'U': 3.7, 'J': 0},
    'Fe': {'L': 2, 'U': 5.3, 'J': 0},
    'Mn': {'L': 2, 'U': 3.9, 'J': 0},
    'Mo': {'L': 2, 'U': 4.38, 'J': 0},
    'Ni': {'L': 2, 'U': 6.2, 'J': 0},
    'V': {'L': 2, 'U': 3.25, 'J': 0},
    'W': {'L': 2, 'U': 6.2, 'J': 0},
}
lmaxmix_dict = {
    'Co': 4,
    'Cr': 4,
    'Fe': 4,
    'Mn': 4,
    'Mo': 4,
    'Ni': 4,
    'V': 4,
    'W': 4,
}
magmom = {
    'Co': 5,
    'Cr': 5,
    'Fe': 5,
    'Mn': 5,
    'Mo': 5,
    'Ni': 5,
    'V': 5,
    'W': 5,
}


def shell_gen_kpt_with_vaspkit(k_density, working_dir):
    os.system(f'cd {working_dir} && (echo 102; echo 2; echo {k_density})|vaspkit')
    return


def shell_gen_potcar_with_vaspkit_and_get_nbands(working_dir):
    nele = subprocess.run(f"cd {working_dir} && " + "vaspkit  -task 103 | grep 'Total Valence Electrons:' | awk '{print $4}'",
                          shell=True,
                          check=True,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          text=True)
    nele = nele.stdout.strip()
    return int(np.ceil(int(eval(nele)) * 2 * 1.3))


def gen_incar(working_dir, incar_setting_dict):
    incar_setting_dict = {k.upper(): v for k, v in incar_setting_dict.items()}
    incar = Incar.from_dict(incar_setting_dict)
    incar.write_file(working_dir + '/INCAR')
    return


def cmd_run_vasp(working_dir, mpirun_np, vasp_dire):
    return f'cd {working_dir} && mpiexec -np {mpirun_np} {vasp_dire}/bin/vasp_std > vasp.log 2>&1'


def cmd_run_vasp_with_timeout(working_dir, mpirun_np, vasp_dire, timeout):
    log_printer.info(f"Starting VASP with maximum time {timeout} seconds")
    return f'cd {working_dir} && timeout -k 5s {timeout}s mpiexec -np {mpirun_np} {vasp_dire}/bin/vasp_std > vasp.log 2>&1'


def get_u_list(struct):
    ldaul, ldauu, ldauj = [], [], []
    lmaxmix = 2
    elements = [e.symbol for e in struct.elements]
    for ele in elements:
        if ele in hubbard_u.keys():
            ldaul.append(hubbard_u[ele]['L'])
            ldauu.append(hubbard_u[ele]['U'])
            ldauj.append(hubbard_u[ele]['J'])
            lmaxmix = max([lmaxmix, lmaxmix_dict[ele]])
        else:
            ldaul.append(-1)
            ldauu.append(0)
            ldauj.append(0)
    return ldaul, ldauu, ldauj, lmaxmix


class EnergyVASP(EnergyEvaluator):
    def __init__(self, incar_file_rlx, incar_file_sin):
        with open("token.yml", "r") as tk_dict:
            params = yaml.safe_load(tk_dict)
        self.vasp_dire = params["vasp_dire"]
        # vasp_pp_dire = params["vasp_pp_dire"]
        npar = params["vasp_npar"]
        self.mpirun_np = params["vasp_mpirun_np"]
        assert self.vasp_dire != "-", "Please specify directory of the VASP package"
        # assert vasp_pp_dire != "-", "Please specify directory of pseudopotential for the VASP package"
        assert npar != "-" and self.mpirun_np != "-", "Please specify NPAR and number of CPU for mpirun command"
        self.npar = npar
        self.timelimit = 43200

        super().__init__(name=EnergyEvaluatorName.vasp(),
                         real_name=EnergyEvaluatorName.vasp(),
                         struct_type='pymatgen')
        self.incar_file_rlx = incar_file_rlx
        self.incar_file_sin = incar_file_sin
        self.incar_settings_rlx = self.ext_calc_settings(incar_file_rlx) if incar_file_rlx is not None else relax_incar
        self.incar_settings_sin = self.ext_calc_settings(incar_file_sin) if incar_file_sin is not None else single_point_incar
        return

    def ext_calc_settings(self, incar_file):
        if not isinstance(incar_file, list):
            incar_dict = Incar.from_file(incar_file)
            return {k.lower(): v for k, v in incar_dict.items()}
        else:
            incar_setting_list = []
            for file in incar_file:
                incar_dict = Incar.from_file(file)
                incar_setting_list.append({k.lower(): v for k, v in incar_dict.items()})
            return incar_setting_list

    def set_logger(self, logger):
        self.log_dir = '.' if logger is None else logger.log_dir
        self.log_dir += f'/vaspout'
        self.logger = logger if logger is not None else None
        return

    def set_output(self, label):
        self.out_dire = self.log_dir + f'/{label}'
        Path(self.out_dire).mkdir(parents=True, exist_ok=False)
        return

    def _vasp_preprocess(self, incar_settings, dir_suffix, add_nbands=False, k_density=0.04):
        out_dire = self.out_dire + f'/incar{dir_suffix}'
        Path(out_dire).mkdir(parents=True, exist_ok=False)
        self.atoms.to(out_dire + '/POSCAR', fmt='poscar')  # gen poscar
        if 'kspacing' not in incar_settings:
            shell_gen_kpt_with_vaspkit(k_density, out_dire)  # gen kpoints
        n_bands = shell_gen_potcar_with_vaspkit_and_get_nbands(out_dire)  # gen potcar

        # gen incar
        incar_settings['npar'] = self.npar
        if add_nbands:
            incar_settings['nbands'] = n_bands

        # not provide incar, follow settings in MP to add hubbard U & magmom
        if self.incar_file_rlx is None and self.incar_settings_sin is None:
            # add init magnetic moment
            self.atom_magmom_list = []
            for i in range(len(self.atoms)):
                try:
                    atom_magmom = magmom[self.atoms[i].specie.symbol]
                    self.atom_magmom_list.append(atom_magmom)
                except KeyError:
                    self.atom_magmom_list.append(0.6)  # high-throughput in MP

            # add U
            ldaul, ldauu, ldauj, lmaxmix = get_u_list(self.atoms)
            if not ((len(np.unique(ldaul)) == 1) and (np.unique(ldaul)[0] == -1)):
                self.hubbard_u_dict = {'ldau': '.TRUE.', 'ldaul': ldaul, 'ldauu': ldauu, 'ldauj': ldauj,
                                       'lmaxmix': lmaxmix, 'ldautype': 2}

            incar_settings['ispin'] = 2
            incar_settings.update({'magmom': self.atom_magmom_list})
            if hasattr(self, 'hubbard_u_dict'):
                incar_settings.update(self.hubbard_u_dict)

        gen_incar(out_dire, incar_settings)
        
        print(out_dire)
        return out_dire

    def _run(self, out_dire):
        os.system(cmd_run_vasp(working_dir=out_dire, mpirun_np=self.mpirun_np, vasp_dire=self.vasp_dire))
        return
    
    def run(self, out_dire, timelimit):
        os.system(cmd_run_vasp_with_timeout(working_dir=out_dire, mpirun_np=self.mpirun_np, vasp_dire=self.vasp_dire, timeout=timelimit))
        return
    
    # def run(self, out_dire, timelimit):
    #     try:
    #         log_printer.info(f"Starting VASP with maximum time {timelimit} seconds")
    #         result = subprocess.run(
    #             cmd_run_vasp(working_dir=out_dire, mpirun_np=self.mpirun_np, vasp_dire=self.vasp_dire),
    #             shell=True,
    #             timeout=timelimit  # Let subprocess handle the timeout
    #         )
    #         log_printer.info(f"VASP process completed with return code: {result.returncode}")
    #         return result.returncode
    #     except subprocess.TimeoutExpired:
    #         log_printer.info(f"VASP process exceeded time limit ({timelimit}s)")
    #         return -1
    #     except Exception as e:
    #         log_printer.info(f"Error running VASP process: {e}")
    #         return -2

    def _cal_sin(self):
        out_dire = self.out_dire + f'/incar0'
        if Path(out_dire + '/vasprun.xml').exists():
            final_atom, save_flag = read_outcar(out_dire, index='-1')
        else:
            out_dire = self._vasp_preprocess(incar_settings=self.incar_settings_sin, dir_suffix=0)
            self.run(out_dire=out_dire, timelimit=self.timelimit)
            final_atom, save_flag = read_outcar(out_dire, index='-1')
        if not save_flag:
            log_printer.info('Calculation failed')
            return None
        else:
            return final_atom

    def cal_energy(self):
        out = self._cal_sin()
        if out is None:
            return None
        else:
            try:
                return out.get_potential_energy()
            except RuntimeError:
                log_printer.info('Calculation failed')
                return None

    def cal_forces(self):
        out = self._cal_sin()
        if out is None:
            return None
        else:
            try:
                return out.get_forces()
            except RuntimeError:
                log_printer.info('Calculation failed')
                return None

    def _cal_relax_one(self, incar_settings, dir_suffix, at_least_one, timelimit_one):
        out_dire = self._vasp_preprocess(incar_settings=incar_settings, dir_suffix=dir_suffix)
        self.run(out_dire=out_dire, timelimit=timelimit_one)
        traj, flag = read_outcar(out_dire, index=':')
        if not flag:
            log_printer.info('Calculation failed')
            traj = []
        try:
            last_atom = read(out_dire + '/CONTCAR')
        except Exception:
            log_printer.info('Fail to read CONTCAR of last calculation, probabily due to calculation failure')
            last_atom = read(out_dire + '/POSCAR')
        obs = TrajectoryObserverMimic(atoms_list=traj, last_atom=last_atom, at_least_one=at_least_one)
        return obs

    def _cal_relax(self, fix_sym=None, steps=None):
        assert isinstance(self.incar_settings_rlx, list) or isinstance(self.incar_settings_rlx, dict)
        if isinstance(self.incar_settings_rlx, dict):
            obs = self._cal_relax_one(self.incar_settings_rlx, 0, at_least_one=False, timelimit_one=self.timelimit)
        else:
            timelimit_one = self.timelimit / len(self.incar_settings_rlx)
            obs = self._cal_relax_one(self.incar_settings_rlx[0], 0, at_least_one=False, timelimit_one=timelimit_one)
            for i in range(1, len(self.incar_settings_rlx)):
                incar_file = self.incar_settings_rlx[i]
                self.build_atoms(struct=obs.atoms)
                obs += self._cal_relax_one(incar_file, i, at_least_one=False, timelimit_one=timelimit_one)
                # if i == len(self.incar_settings_rlx) - 1:
                #     obs += self._cal_relax_one(incar_file, i, at_least_one=True)
                # else:
                #     obs += self._cal_relax_one(incar_file, i, at_least_one=False)
        if len(obs) == 0:
            return None
        return {
            "final_structure": AseAtomsAdaptor.get_structure(obs.atoms),
            "trajectory": obs,
        }


class TrajectoryObserverMimic:
    def __init__(self, atoms_list, last_atom=None, at_least_one=False):
        f_atoms = atoms_list[-1][0] if last_atom is None else last_atom
        if isinstance(f_atoms, Atoms):
            self.atoms = f_atoms
        elif isinstance(f_atoms, Structure):
            self.atoms = AseAtomsAdaptor.get_atoms(f_atoms)
        elif isinstance(f_atoms, Struct):
            self.atoms = f_atoms.atoms()
        else:
            raise NotImplementedError
        
        self.energies, self.forces, self.stresses, self.atom_positions, self.cells = \
            [], [], [], [], []
        for atoms, save_flag in atoms_list:
            if save_flag:
                energy, forces, stress, positions, cell = self._extract_calculated_results(atoms)
                self.energies.append(energy)
                self.forces.append(forces)
                self.stresses.append(stress)
                self.atom_positions.append(positions)
                self.cells.append(cell)
        if at_least_one:
            if len(self.energies) == 0:
                print("Collect at least the last one")
                atoms = atoms_list[-1][0]
                energy, forces, stress, positions, cell = self._extract_calculated_results(atoms)
                self.energies.append(energy)
                self.forces.append(forces)
                self.stresses.append(stress)
                self.atom_positions.append(positions)
                self.cells.append(cell)
        return

    def _extract_calculated_results(self, atoms):
        p, c = atoms.get_positions(), atoms.get_cell()[:]
        try:
            e, f, s = float(atoms.get_potential_energy()), atoms.get_forces(), atoms.get_stress()
        except Exception:
            log_printer.info("No results due to failed calculation")
            e, f, s = None, None, None
        return e, f, s, p, c

    def __getitem__(self, item):
        return self.energies[item], self.forces[item], self.stresses[item], self.cells[item], self.atom_positions[item]

    def __len__(self):
        return len(self.energies)

    def __add__(self, other):
        self.atoms = other.atoms
        self.energies += other.energies
        self.forces += other.forces
        self.stresses += other.stresses
        self.atom_positions += other.atom_positions
        self.cells += other.cells
        return self

    def as_pandas(self) -> pd.DataFrame:
        """Returns: DataFrame of energies, forces, stresses, cells and atom_positions."""
        return pd.DataFrame(
            {
                "energies": self.energies,
                "forces": self.forces,
                "stresses": self.stresses,
                "cells": self.cells,
                "atom_positions": self.atom_positions,
            }
        )

    def save(self, filename: str) -> None:
        """Save the trajectory to file.
        Args:
            filename (str): filename to save the trajectory.
        """
        out = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with bz2.open(filename, "wb") as file:
            pickle.dump(out, file)


def read_outcar(out_dire, index=':'):
    try:
        vasp_out = read(out_dire + '/vasprun.xml', ':')
    except Exception:
        try:
            vasp_out = Vasprun(out_dire + '/vasprun.xml')
        except Exception:
            log_printer.info(f'VASP output file reading fails')
            return None, False
    
    oszicar = Oszicar(out_dire + '/OSZICAR')
    if len(oszicar.electronic_steps) == 0:
        log_printer.info(f'Calculation failed with 0 electronic steps')
        return None, False
        
    save_flag = True
    
    # 1. check ediff
    incar = Incar.from_file(out_dire + '/INCAR')
    try:
        ediff = incar['EDIFF']
    except KeyError:
        ediff = 1e-4  # default value in https://www.vasp.at/wiki/EDIFF
    if ediff > 1e-5:
        log_printer.info(f'EDIFF in {out_dire} is not low enough, pass')
        save_flag = False
    
    # 2. check electronic convergence status
    def is_e_step_converged(_i, _incar, _oszicar):
        _num_e_step = len(_oszicar.electronic_steps[_i])
        try:
            _nelm = _incar['NELM']
        except KeyError:
            _nelm = 60  # default value in https://www.vasp.at/wiki/index.php/NELM
        if _num_e_step >= _nelm:
            log_printer.info(f'{_i}-th electronic SCF is not converged in {out_dire} with {_num_e_step}/{_nelm}, pass')
            return False
        return True and save_flag
    
    def read_pmg_vasp_out(_frame):
        _s = pymatgen_struct_to_array(_frame['structure'])
        _s.potential_energy = _frame['electronic_steps'][-1]['e_0_energy']
        _s.forces = _frame['forces']
        stress = np.array([_frame['stress'][0][0], _frame['stress'][1][1], _frame['stress'][2][2], 
                           _frame['stress'][0][1], _frame['stress'][0][2], _frame['stress'][1][2]])
        _s.stress = stress * 6.241509125883258e-4
        return _s

    if index == ':':
        if isinstance(vasp_out, Vasprun):
            return [(read_pmg_vasp_out(frame), is_e_step_converged(i, incar, oszicar)) for i, frame in enumerate(vasp_out.ionic_steps)], True
        else:  # ase gives a list
            return [(frame, is_e_step_converged(i, incar, oszicar)) for i, frame in enumerate(vasp_out)], True
    else:
        if isinstance(vasp_out, Vasprun):
            frame = vasp_out.ionic_steps[int(index)]
            return read_pmg_vasp_out(frame), is_e_step_converged(int(index), incar, oszicar)
        else:  # ase gives a list
            frame = vasp_out[int(index)]
            return frame, is_e_step_converged(int(index), incar, oszicar)
        
    # with open(out_dire + '/OUTCAR', 'r') as f:
    #     out = f.readlines()
    # out = ' '.join(out)
    # if 'reached required accuracy' not in out:
    #     print_or_record(logger, f'OUTCAR in {out_dire} did not reach convergence, pass')
    #     return None
    # 
    # if index == ':':
    #     traj, i = [], 0
    #     while True:
    #         try:
    #             frame = read(out_dire + '/OUTCAR', index=f'{i}')
    #             traj.append(frame)
    #             i += 1
    #         except ValueError:
    #             i += 1
    #         except StopIteration:
    #             break
    #     return traj
    # else:
    #     return read(out_dire + '/OUTCAR', index=index)
