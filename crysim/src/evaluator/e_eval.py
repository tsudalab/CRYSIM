import ase
from ase.atoms import Atoms
from ase.optimize import BFGS, FIRE
from ase.filters import ExpCellFilter
from ase.constraints import FixSymmetry
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import pandas as pd
import pickle
import collections
import bz2
from pathlib import Path
from logzero import logger as log_printer

from crysim.src import DATASET_ROOT, DEBUG
from crysim.src.struct_utils import Struct, array_to_pymatgen_struct, pick, get_savable_structures
from crysim.src.registry import Component


# _fmax = 0.005
# _steps = 500
_fmax = 0.01
if not DEBUG:
    _steps = 1000
else:
    _steps = 50
_opt = 'FIRE'


class EnergyEvaluator(Component):
    def __init__(self, name, real_name, struct_type):
        super().__init__(name, real_name)
        assert struct_type in ['ase', 'pymatgen']
        self.struct_type = struct_type
        return

    def build_atoms(self, struct, cart=False):
        if self.struct_type == 'ase':
            if isinstance(struct, Atoms):
                self.atoms = struct
            elif isinstance(struct, Structure):
                self.atoms = AseAtomsAdaptor.get_atoms(structure=struct)
            elif isinstance(struct, Struct):
                if cart:
                    self.atoms = ase.Atoms(symbols=struct.species, positions=struct.pos,
                                           cell=struct.lattice, pbc=struct.pbc)
                else:
                    self.atoms = ase.Atoms(symbols=struct.species, scaled_positions=struct.pos,
                                           cell=struct.lattice, pbc=struct.pbc)
            else:
                raise NotImplementedError
            self.atom_name = str(self.atoms.symbols)

        elif self.struct_type == 'pymatgen':
            if isinstance(struct, Atoms):
                self.atoms = AseAtomsAdaptor.get_structure(atoms=struct)
            elif isinstance(struct, Structure):
                self.atoms = struct
            elif isinstance(struct, Struct):
                self.atoms = array_to_pymatgen_struct(lattice=struct.lattice,
                                                      pos=struct.pos,
                                                      species_list=struct.species,
                                                      cart=struct.cart,
                                                      pbc=struct.pbc)
            else:
                raise NotImplementedError
            self.atom_name = self.atoms.formula.replace(' ', '')

        else:
            raise NotImplementedError

        # self.atoms.set_constraint([FixSymmetry(self.atoms)])  # fix space group

        return

    def set_traj_logger(self, logger, traj_dir=None):
        if logger is not None:
            self.traj_dir = logger.log_dir
        else:
            if traj_dir is not None:
                self.traj_dir = DATASET_ROOT + f'/{traj_dir}'
            else:
                self.traj_dir = DATASET_ROOT
        self.traj_dir += f'/traj_out'
        Path(self.traj_dir).mkdir(parents=True, exist_ok=True)
        self.logger = logger
        return

    def set_traj_label(self, label):
        self.traj_label = label
        return

    def cal_energy(self):
        raise NotImplementedError

    def cal_forces(self):
        raise NotImplementedError

    def _cal_relax(self, fix_sym, steps, logfile='-', trajfile=None):
        raise NotImplementedError

    def _traj_save(self, obs):
        traj_path = self.traj_dir + '/' + f'traj-{self.traj_label}.pkl.bz2'
        obs.save(filename=traj_path)
        # log_printer.info(f"Save traj at {traj_path}")
        return

    def cal_relax(self, simple=False, fix_sym=False, steps=_steps, logfile='-', trajfile=None):
        relax_results = self._cal_relax(fix_sym=fix_sym, steps=steps, logfile=logfile, trajfile=trajfile)
        if relax_results is None:
            return None, None, None
        # extract results
        final_structure = relax_results["final_structure"]
        final_energy = relax_results["trajectory"].energies[-1]
        final_forces = relax_results["trajectory"].forces[-1]
        if not simple:
            self._traj_save(obs=relax_results["trajectory"])
        return final_structure, final_energy, final_forces

    @staticmethod
    def _build_struct_from_traj(trajectory, idx, species):
        cell = trajectory.cells[idx]
        positions = trajectory.atom_positions[idx]
        traj_struct = Structure(lattice=cell, species=species, coords=positions, coords_are_cartesian=True)
        return traj_struct

    def collect_trajectory(self, init_e_threshold=10.0, fix_sym=False, steps=_steps, logfile='-', trajfile=None):
        relax_results = self._cal_relax(fix_sym=fix_sym, steps=steps, logfile=logfile, trajfile=trajfile)
        if relax_results is None:
            return None, None, None
        trajectory = relax_results['trajectory']
        self._traj_save(obs=trajectory)
        init_idx = 0
        if init_e_threshold is not None:
            while trajectory.energies[init_idx] > init_e_threshold:
                # if len(traj) < sam_num, then init_idx == 0
                init_idx += 1
        species = trajectory.atoms.symbols
        structure_list, energies, forces = [], [], []
        for i in range(init_idx, len(trajectory)):
            structure_list.append(self._build_struct_from_traj(trajectory, i, species))
            energies.append(trajectory.energies[i])
            forces.append(trajectory.forces[i])
        return structure_list, energies, forces

    def sample_trajectory(self, init_e_threshold=10.0, sample_number=10, fix_sym=False, steps=_steps, logfile='-', trajfile=None):
        """ For M3GNet and CHGNet, trajectory saves Atoms with keys 'atoms', 'energies',
        'forces', 'stresses', 'atom_positions', 'cell'. """
        relax_results = self._cal_relax(fix_sym=fix_sym, steps=steps, logfile=logfile, trajfile=trajfile)
        if relax_results is None:
            return None, None, None
        trajectory = relax_results['trajectory']
        self._traj_save(obs=trajectory)
        sample_number = sample_number - 2  # remove init & relaxed
        init_idx = 0
        if init_e_threshold is not None:
            while (trajectory.energies[init_idx] > init_e_threshold) and \
                    (init_idx < len(trajectory) - sample_number - 2):
                # if len(traj) < sam_num, then init_idx == 0
                init_idx += 1
        n = len(trajectory) - 1 - (init_idx + 1)  # remove init & relaxed
        species = trajectory.atoms.symbols
        structure_list, energies, forces = [], [], []

        def _append_struct(_sample, _threshold):
            traj_struct = self._build_struct_from_traj(trajectory=trajectory, idx=_sample, species=species)
            structure_list.append(traj_struct)
            # e = trajectory.energies[_sample] if trajectory.energies[_sample] <= _threshold else _threshold
            e = trajectory.energies[_sample]
            energies.append(e)
            forces.append(trajectory.forces[_sample])
            return

        _append_struct(init_idx, init_e_threshold)  # get the closest init struct

        if n <= sample_number:
            # do not append relaxed, since always traj[n] == traj[n - 1]
            for i in range(n):
                _append_struct(init_idx + i + 1, init_e_threshold)

        elif n <= sample_number * 2:
            for i in range(sample_number):
                _append_struct(init_idx + 1 + i, init_e_threshold)
            _append_struct(len(trajectory) - 1, init_e_threshold)  # get relaxed struct

        else:
            count, count_save = 0, [init_idx, len(trajectory) - 1]
            while count < sample_number:
                sample = pick(n=n, bias=5) + init_idx
                while sample in count_save:
                    if sample == len(trajectory) - 2:
                        sample = init_idx
                    sample += 1
                _append_struct(sample, init_e_threshold)
                count += 1
                count_save.append(sample)
            _append_struct(len(trajectory) - 1, init_e_threshold)  # get relaxed struct

        return structure_list, energies, forces


# from https://github.com/materialsvirtuallab/matgl/blob/main/src/matgl/ext/ase.py
class TrajectoryObserver(collections.abc.Sequence):
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """
        Init the Trajectory Observer from a Atoms.

        Args:
            atoms (Atoms): Structure to observe.
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """The logic for saving the properties of an Atoms during the relaxation."""
        self.energies.append(float(self.atoms.get_potential_energy()))
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress())
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:])

    def __getitem__(self, item):
        return self.energies[item], self.forces[item], self.stresses[item], self.cells[item], self.atom_positions[item]

    def __len__(self):
        return len(self.energies)

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


class GeneralCalculator(EnergyEvaluator):
    def __init__(self, name, calc, opt, dim):
        super().__init__(name=name, real_name=name, struct_type='ase')
        self.calc = calc
        assert opt in ['BFGS', 'FIRE']
        self.opt = opt
        self.dim = dim
        return

    def _attach_calc(self):
        if self.atoms.calc is None:
            self.atoms.calc = self.calc
        return

    def cal_energy(self):
        self._attach_calc()
        return self.atoms.get_potential_energy()

    def cal_forces(self):
        self._attach_calc()
        return self.atoms.get_forces()

    def _cal_relax(self, fix_sym, steps, logfile='-', trajfile=None):
        self._attach_calc()
        if fix_sym:
            self.atoms.set_constraint(FixSymmetry(self.atoms))
        obs = TrajectoryObserver(self.atoms)
        if self.dim == 3:
            self.atoms = ExpCellFilter(self.atoms)
        if self.opt == 'BFGS':
            dyn = BFGS(self.atoms, logfile=logfile, trajectory=trajfile)
        elif self.opt == 'FIRE':
            dyn = FIRE(self.atoms, logfile=logfile, trajectory=trajfile)
        else:
            raise NotImplementedError
        dyn.attach(obs, interval=1)
        dyn.run(fmax=_fmax, steps=steps)
        obs()
        if isinstance(self.atoms, ExpCellFilter):
            self.atoms = self.atoms.atoms
        struct = get_savable_structures(self.atoms)
        return {
            "final_structure": struct,
            "trajectory": obs,
        }
        

def frame2struct(traj, i):
    struct = Structure(lattice=traj['cell'][i], species=traj['atomic_number'],
                      coords=traj['atom_positions'][i], coords_are_cartesian=True)
    return struct


def output_traj_to_poscar(traj_dir, filename=None):
    traj_dir = Path(traj_dir)
    filename = Path(traj_dir.stem).stem if filename is None else filename
    filename = traj_dir.parent / filename
    filename.mkdir(exist_ok=True)
    with bz2.open(traj_dir, 'rb') as f:
        traj = pickle.load(f)
    pmg_list = []
    for i in range(len(traj['energy'])):
        pmg = frame2struct(traj, i)
        pmg.to(filename / f'POSCAR_{i}', fmt='poscar')
        pmg_list.append(pmg)
    return pmg_list


def average_force(forces, eps=1e-6):
    """ forces.shape == (n, 3) """
    if forces is None:
        return None
    forces = np.abs(forces) + eps
    forces = np.sqrt(np.sum(forces ** 2, axis=-1))
    return np.mean(forces, axis=-1)
