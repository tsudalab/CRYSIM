from crysim.src import PROJECT_ROOT
import sys
sys.path.append(PROJECT_ROOT + "/base")


try:
    from base.inference.calculator import BaseCalculator
except ModuleNotFoundError:
    pass


import torch
from crysim.src.registry import EnergyEvaluatorName
from crysim.src.evaluator.e_eval import GeneralCalculator


test_ckpt = "-"
config_path = "-"
# model_ckpt = "-"


class EnergyBase(GeneralCalculator):
    def __init__(self, opt, dim):
        device = 'cpu' if not torch.cuda.is_available() else 'cuda'
        calc = BaseCalculator(ckpt_path=test_ckpt, dtype=torch.float32, config_path=config_path, device=device, cutoff=5.0)
        super().__init__(name=EnergyEvaluatorName.base_simulator(), calc=calc, opt=opt, dim=dim)
        return
