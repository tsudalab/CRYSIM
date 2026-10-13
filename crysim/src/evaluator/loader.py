import time
from crysim.src.registry import EnergyEvaluatorName
from crysim.src.evaluator.m3gnet import EnergyM3GNET
from crysim.src.evaluator.vasp import EnergyVASP
from crysim.src.evaluator.chgnet import EnergyCHGNet
from crysim.src.evaluator.orb import EnergyOrb
from crysim.src.evaluator.mattersim import EnergyMattersim
from crysim.src.evaluator.uma import EnergyUMAS, EnergyUMAM
from crysim.src.evaluator.base import EnergyBase
from crysim.src.evaluator.e_eval import _opt


def load_energy_evaluator(e_cal_name, dim, opt=_opt, incar_sin_dir=None, incar_rlx_dir=None):
    # energy evaluator to do structure relaxation and energy estimation
    start_time = time.process_time()
    if e_cal_name == EnergyEvaluatorName.m3gnet():
        e_cal = EnergyM3GNET()
    elif e_cal_name == EnergyEvaluatorName.vasp():
        e_cal = EnergyVASP(incar_file_sin=incar_sin_dir, incar_file_rlx=incar_rlx_dir)
    elif e_cal_name == EnergyEvaluatorName.chgnet():
        e_cal = EnergyCHGNet()
    elif e_cal_name == EnergyEvaluatorName.orb():
        e_cal = EnergyOrb(opt=opt, dim=dim)
    elif e_cal_name == EnergyEvaluatorName.mattersim():
        e_cal = EnergyMattersim(opt=opt, dim=dim)
    elif e_cal_name == EnergyEvaluatorName.uma_s():
        e_cal = EnergyUMAS(opt=opt, dim=dim)
    elif e_cal_name == EnergyEvaluatorName.uma_m():
        e_cal = EnergyUMAM(opt=opt, dim=dim)
    elif e_cal_name == EnergyEvaluatorName.base_simulator():
        e_cal = EnergyBase(opt=opt, dim=dim)
    else:
        raise NotImplementedError
    print(f"load energy evaluator: {time.process_time() - start_time}")
    return e_cal
