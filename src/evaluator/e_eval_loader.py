import time
from src.registry import EnergyEvaluatorName
from src.evaluator.e_eval import EnergyM3GNET
# from src.evaluator.e_eval_chgnet import EnergyCHGNet
# from src.evaluator.e_eval_orb import EnergyOrb


def load_energy_evaluator(e_cal_name):
    # energy evaluator to do structure relaxation and energy estimation
    start_time = time.process_time()
    if e_cal_name == EnergyEvaluatorName.m3gnet():
        e_cal = EnergyM3GNET()
    # elif e_cal_name == EnergyEvaluatorName.chgnet():
    #     e_cal = EnergyCHGNet()
    # elif e_cal_name == EnergyEvaluatorName.orb():
    #     e_cal = EnergyOrb()
    else:
        raise NotImplementedError
    print(f"load energy evaluator: {time.process_time() - start_time}")
    return e_cal
