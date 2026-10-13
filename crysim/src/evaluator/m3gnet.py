from crysim.src.registry import EnergyEvaluatorName
from crysim.src.evaluator.e_eval import EnergyEvaluator, _fmax, _steps
from crysim.src.evaluator.e_eval import GeneralCalculator

try:
    import matgl
    import matgl.ext.ase as mea
except ModuleNotFoundError:
    pass


# class EnergyM3GNET(EnergyEvaluator):
#     def __init__(self):
#         super().__init__(name=EnergyEvaluatorName.m3gnet(),
#                          real_name=EnergyEvaluatorName.m3gnet(),
#                          struct_type='ase')
#         self.pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
#         return

#     def cal_energy(self):
#         calc = mea.M3GNetCalculator(self.pot)
#         self.atoms.set_calculator(calc)
#         return self.atoms.get_potential_energy()

#     def _cal_relax(self, fix_sym, steps):
#         relaxer = mea.Relaxer(potential=self.pot)
#         relax_results = relaxer.relax(self.atoms, fmax=_fmax, steps=steps)
#         return relax_results


class EnergyM3GNET(GeneralCalculator):
    def __init__(self):
        pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        calc = mea.M3GNetCalculator(pot)
        super().__init__(name=EnergyEvaluatorName.m3gnet(), calc=calc)
        return
