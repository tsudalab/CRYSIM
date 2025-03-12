import ase
from ase.atoms import Atoms
# from ase.spacegroup.symmetrize import FixSymmetry
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

import matgl
import matgl.ext.ase as mea

from src.struct_utils import Struct, array_to_pymatgen_struct
from src.registry import Component, EnergyEvaluatorName


_fmax = 0.005
_steps = 500


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

        elif self.struct_type == 'pymatgen':
            if isinstance(struct, Atoms):
                self.atoms = AseAtomsAdaptor.get_structure(atoms=struct)
            elif isinstance(struct, Structure):
                self.atoms = struct
            elif isinstance(struct, Struct):
                self.atoms = array_to_pymatgen_struct(lattice=struct.lattice,
                                                      pos=struct.pos,
                                                      species_list=struct.species,
                                                      cart=struct.cart)
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        # self.atoms.set_constraint([FixSymmetry(self.atoms)])  # fix space group

        return

    def cal_energy(self):
        raise NotImplementedError

    def _cal_relax(self):
        raise NotImplementedError

    def cal_relax(self):
        relax_results = self._cal_relax()
        # extract results
        final_structure = relax_results["final_structure"]
        final_energy = relax_results["trajectory"].energies[-1]
        return final_structure, final_energy

    def sample_trajectory(self, init_e_threshold=10.0, sample_number=10):
        """ For M3GNet and CHGNet, trajectory saves Atoms with keys 'atoms', 'energies',
        'forces', 'stresses', 'atom_positions', 'cell'. """

        trajectory = self._cal_relax()['trajectory']
        sample_number = sample_number - 2  # remove init & relaxed
        init_idx = 0
        while (trajectory.energies[init_idx] > init_e_threshold) and \
                (init_idx < len(trajectory) - sample_number - 2):
            # if len(traj) < sam_num, then init_idx == 0
            init_idx += 1
        n = len(trajectory) - 1 - (init_idx + 1)  # remove init & relaxed
        species = trajectory.atoms.symbols
        structure_list, energies = [], []

        def _append_struct(_sample, _threshold):
            cell = trajectory.cells[_sample]
            positions = trajectory.atom_positions[_sample]
            traj_struct = Structure(lattice=cell, species=species, coords=positions,
                                    coords_are_cartesian=True)
            structure_list.append(traj_struct)
            # e = trajectory.energies[_sample] if trajectory.energies[_sample] <= _threshold else _threshold
            e = trajectory.energies[_sample]
            energies.append(e)
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

        return structure_list, energies


class EnergyM3GNET(EnergyEvaluator):
    def __init__(self):
        super().__init__(name=EnergyEvaluatorName.m3gnet(),
                         real_name=EnergyEvaluatorName.m3gnet(),
                         struct_type='ase')
        self.pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        return

    def cal_energy(self):
        calc = mea.M3GNetCalculator(self.pot)
        self.atoms.set_calculator(calc)
        return self.atoms.get_potential_energy()

    def _cal_relax(self):
        relaxer = mea.Relaxer(potential=self.pot)
        relax_results = relaxer.relax(self.atoms, fmax=_fmax, steps=_steps)
        return relax_results
