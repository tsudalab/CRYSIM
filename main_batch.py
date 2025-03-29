import sys
sys.path.append('./src')
import argparse
from src.wrapper import ModelWrapper
from src.registry import EnergyEvaluatorName, SolverName, OptimizerName, RegressionName

import os
import torch
gpu_no = 0
device = f'cuda:{gpu_no}'
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{gpu_no}'
    torch.set_default_device(device)

# sys_name = 'Na8Cl8'
# sys_name = 'Y12Co102'
# sys_name = 'Y6Co51'
# sys_name = 'Ca24Al16Si24O96'
# sys_name = 'Si96O192'
init_pos_num = 1000
init_sample_rate = 1
iter_sample_num = 30

"------------parameters------------"

parser = argparse.ArgumentParser(description='test')
parser.add_argument('-sys', '--system', type=str)
parser.add_argument('-sol', '--solver', type=str)
parser.add_argument('-reg', '--regressor', type=str, default=RegressionName.fm()[0])
parser.add_argument('-eval', '--evaluator', type=str, default=EnergyEvaluatorName.m3gnet())
parser.add_argument('-s', '--seeds', type=int, action='append')
parser.add_argument('-p', '--precs', type=int, action='append')
parser.add_argument('-n', '--n-steps', type=int, default=300)
parser.add_argument('-t', '--timeout', type=int, default=None)
parser.add_argument('-f', '--filter-struct', type=bool, default=True)
parser.add_argument('-d', '--dist-min', type=int, default=None)
parser.add_argument('-w', '--n-wps', type=int, default=100)
parser.add_argument('--fail-num', type=int, default=1)
parser.add_argument('--follow-dir', type=str, default=None)
args = parser.parse_args()

sys_name = args.system

seeds = args.seeds
seeds = [None] if seeds is None else seeds
precs = args.precs
precs = [None] if precs is None else precs

n_steps = args.n_steps
# n_steps = 300

timeout = args.timeout
# timeout = None

filter_struct = args.filter_struct

dist_min = args.dist_min
# dist_min = None
# dist_min = 2

wp_sample_num = args.n_wps
follow_dir = args.follow_dir
# follow_dir = None
fail_num = args.fail_num

"------------components------------"

sol_name = args.solver
learning_model_type = args.regressor
e_cal_name = args.evaluator

# sol_name = SolverName.amplify_leap_hybrid()[0]
# sol_name = SolverName.amplify()[0]
# sol_name = SolverName.amplify_dwave()[0]
# sol_name = SolverName.hyperopt_onehot_bayesian_optimization()[0]
# sol_name = SolverName.hyperopt_bayesian_optimization()[0]

# learning_model_type = RegressionName.fm()[0]
# learning_model_type = RegressionName.sim()[0]

if sol_name in [SolverName.amplify()[0],
                SolverName.amplify_leap_hybrid()[0],
                SolverName.amplify_dwave()[0],
                ]:
    print('Doing amp')
    for opt_name in [OptimizerName.onehot()[0]]:
        for prec in precs:
            for seed in seeds:
                ow3 = ModelWrapper(opt_name=opt_name, system_name=sys_name,
                                   e_cal_name=e_cal_name, sol_name=sol_name,
                                   learning_model_type=learning_model_type,
                                   num_steps=n_steps, seed=seed, filter_gen_struct=filter_struct,
                                   init_pos_num=init_pos_num,
                                   init_sample_rate=init_sample_rate,
                                   iter_sample_num=iter_sample_num,
                                   fail_num=fail_num, wp_sample_num=wp_sample_num,
                                   precs=prec, dist_min=dist_min, timeout=timeout,
                                   control_lat_shape=True, control_crys_sys=True, control_spg=True,
                                   use_constraints=False,
                                   follow_dir=follow_dir)
                g1, h1, i1 = ow3.test()

elif sol_name in [SolverName.hyperopt_onehot_bayesian_optimization()[0],
                  SolverName.hyperopt_onehot_simulated_annealing()[0]]:
    print('Doing hyper')
    for opt_name in [OptimizerName.onehot()[0]]:
        for prec in precs:
            for seed in seeds:
                ow3 = ModelWrapper(opt_name=opt_name, system_name=sys_name,
                                   e_cal_name=e_cal_name, sol_name=sol_name,
                                   learning_model_type=learning_model_type,
                                   num_steps=n_steps, seed=seed, filter_gen_struct=filter_struct,
                                   init_pos_num=init_pos_num,
                                   init_sample_rate=init_sample_rate,
                                   iter_sample_num=iter_sample_num,
                                   precs=prec, wp_sample_num=wp_sample_num, fail_num=fail_num,
                                   control_lat_shape=True, control_crys_sys=True, control_spg=True,
                                   follow_dir=follow_dir)
                g1, h1, i1 = ow3.test()

elif sol_name in [SolverName.hyperopt_bayesian_optimization()[0],
                  SolverName.hyperopt_simulated_annealing()[0]]:
    print('Doing pure hyper')
    for opt_name in [OptimizerName.crystal_params_vector()[0]]:
        for seed in seeds:
            ow3 = ModelWrapper(opt_name=opt_name, system_name=sys_name,
                               e_cal_name=e_cal_name, sol_name=sol_name,
                               init_pos_num=init_pos_num,
                               num_steps=n_steps, seed=seed, fail_num=fail_num, follow_dir=None)
            g1, h1, i1 = ow3.test()

elif sol_name in [f'{SolverName.amplify()[0]}_c',
                  f'{SolverName.amplify_leap_hybrid()[0]}_c',
                  f'{SolverName.amplify_dwave()[0]}_c',
                  ]:
    sol_name = sol_name[:-2]
    print('Doing constrained amp')
    for opt_name in [OptimizerName.onehot()[0]]:
        for prec in precs:
            for seed in seeds:
                ow3 = ModelWrapper(opt_name=opt_name, system_name=sys_name,
                                   e_cal_name=e_cal_name, sol_name=sol_name,
                                   learning_model_type=learning_model_type,
                                   num_steps=n_steps, seed=seed, filter_gen_struct=filter_struct,
                                   init_pos_num=init_pos_num,
                                   init_sample_rate=init_sample_rate,
                                   iter_sample_num=iter_sample_num,
                                   fail_num=fail_num, wp_sample_num=wp_sample_num,
                                   precs=prec, dist_min=dist_min, timeout=timeout,
                                   control_lat_shape=True, control_crys_sys=True, control_spg=True,
                                   use_constraints=True,
                                   follow_dir=follow_dir)
                g1, h1, i1 = ow3.test()

elif sol_name in [OptimizerName.random_search()[0]]:
    print('Doing random search')
    for opt_name in [OptimizerName.random_search()[0]]:
        for seed in seeds:
            ow3 = ModelWrapper(opt_name=opt_name, system_name=sys_name,
                               e_cal_name=e_cal_name, init_pos_num=10,
                               num_steps=n_steps, seed=seed,
                               filter_gen_struct=False, follow_dir=None)
            g2, h2, i2 = ow3.test()

else:
    raise NotImplementedError
