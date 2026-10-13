try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
except ModuleNotFoundError:
    pass


import torch
from crysim.src.registry import EnergyEvaluatorName
from crysim.src.evaluator.e_eval import GeneralCalculator


class EnergyUMA(GeneralCalculator):
    def __init__(self, name, pot_name, opt, dim):
        device = "cpu" if not torch.cuda.is_available() else "cuda"
        predictor = pretrained_mlip.get_predict_unit(pot_name, device=device)
        calc = FAIRChemCalculator(predictor, task_name="omat")
        super().__init__(name=name, calc=calc, opt=opt, dim=dim)
        return


class EnergyUMAS(EnergyUMA):
    def __init__(self, opt, dim):
        super().__init__(name=EnergyEvaluatorName.uma_s(), pot_name="uma-s-1p1", opt=opt, dim=dim)
        return


class EnergyUMAM(EnergyUMA):
    def __init__(self, opt, dim):
        super().__init__(name=EnergyEvaluatorName.uma_m(), pot_name="uma-m-1p1", opt=opt, dim=dim)
        return
