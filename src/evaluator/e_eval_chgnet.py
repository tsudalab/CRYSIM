import ase

from src.registry import EnergyEvaluatorName
from src.evaluator.e_eval import EnergyEvaluator, _fmax, _steps
import chgnet.model as chg


class EnergyCHGNet(EnergyEvaluator):
    def __init__(self):
        super().__init__(name=EnergyEvaluatorName.chgnet(),
                         real_name=EnergyEvaluatorName.chgnet(),
                         struct_type='pymatgen')
        self.pot = chg.CHGNet.load()
        return

    def cal_energy(self):
        prediction = self.pot.predict_structure(self.atoms)
        return prediction['energy']

    def _cal_relax(self):
        relaxer = chg.StructOptimizer()
        # ase_filter = ase.filters.ExpCellFilter
        ase_filter = ase.filters.FrechetCellFilter
        result = relaxer.relax(self.atoms, fmax=_fmax, steps=_steps, verbose=False, ase_filter=ase_filter)
        return result
