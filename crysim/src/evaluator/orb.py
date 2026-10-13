try:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator
except ModuleNotFoundError:
    pass


import torch
from crysim.src.registry import EnergyEvaluatorName
from crysim.src.evaluator.e_eval import GeneralCalculator


class EnergyOrb(GeneralCalculator):
    def __init__(self, opt, dim):
        device = 'cpu' if not torch.cuda.is_available() else 'cuda'
        pot, atoms_adapter = pretrained.orb_v3_conservative_inf_omat(
            device=device,
            precision="float32-high",  # or "float32-highest" / "float64
            # precision="float32-highest",  # or "float32-highest" / "float64
        )
        calc = ORBCalculator(pot, atoms_adapter=atoms_adapter, device=device)
        super().__init__(name=EnergyEvaluatorName.orb(), calc=calc, opt=opt, dim=dim)
        return
