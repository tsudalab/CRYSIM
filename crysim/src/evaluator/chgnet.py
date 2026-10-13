import ase
try:
    import chgnet.model as chg
except ModuleNotFoundError:
    pass

from crysim.src.registry import EnergyEvaluatorName
from crysim.src.evaluator.e_eval import EnergyEvaluator, _fmax, _steps


class EnergyCHGNet(EnergyEvaluator):
    def __init__(self):
        super().__init__(name=EnergyEvaluatorName.chgnet(),
                         real_name=EnergyEvaluatorName.chgnet(),
                         struct_type='pymatgen')
        self.pot = chg.model.CHGNet.load()
        return

    def cal_energy(self):
        prediction = self.pot.predict_structure(self.atoms)
        return prediction['energy']

    def _cal_relax(self, fix_sym):
        relaxer = chg.StructOptimizer()
        # ase_filter = ase.filters.ExpCellFilter
        ase_filter = ase.filters.FrechetCellFilter
        result = relaxer.relax(self.atoms, fmax=_fmax, steps=_steps, verbose=False, ase_filter=ase_filter)
        return result
