try:
    from mattersim.forcefield import MatterSimCalculator
except ModuleNotFoundError:
    pass


import torch
from crysim.src.registry import EnergyEvaluatorName
from crysim.src.evaluator.e_eval import GeneralCalculator


class EnergyMattersim(GeneralCalculator):
    def __init__(self, opt, dim):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=device)
        super().__init__(name=EnergyEvaluatorName.mattersim(), calc=calc, opt=opt, dim=dim)
        return
