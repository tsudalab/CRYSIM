from ase.optimize import BFGS
import torch
import matgl.ext.ase as mea
from pymatgen.io.ase import AseAtomsAdaptor

from src.registry import EnergyEvaluatorName
from src.evaluator.e_eval import EnergyEvaluator, _fmax, _steps

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator


class EnergyOrb(EnergyEvaluator):
    def __init__(self):
        super().__init__(name=EnergyEvaluatorName.orb(),
                         real_name=EnergyEvaluatorName.orb(),
                         struct_type='ase')
        self.device = 'cpu' if not torch.cuda.is_available() else 'cuda'
        self.pot = pretrained.orb_v1(device=self.device)
        self.calc = ORBCalculator(self.pot, device=self.device)
        return

    def cal_energy(self):
        self.atoms.calc = self.calc
        return self.atoms.get_potential_energy()

    def _cal_relax(self):
        self.atoms.calc = self.calc
        # Rattle the atoms to get them out of the minimum energy configuration
        self.atoms.rattle(0.5)
        obs = mea.TrajectoryObserver(self.atoms)
        dyn = BFGS(self.atoms, logfile=None)
        dyn.attach(obs, interval=1)
        dyn.run(fmax=_fmax, steps=_steps)
        obs()
        return {
            "final_structure": AseAtomsAdaptor.get_structure(self.atoms),
            "trajectory": obs,
        }
