import sys
sys.path.append('./src')
import argparse
from crysim.src.wrapper import ModelWrapper
from crysim.src.registry import OptimizerName, rlx_type, qubo_list, coords_emb, sym_emb, enc_obj
from crysim.src import INCAR_ROOT
from crysim.src.spg.spg_utils import regular_sg_list
from crysim.src.evaluator.e_eval import _steps

import os
import torch
gpu_no = 0
device = f'cuda:{gpu_no}'
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{gpu_no}'
    torch.set_default_device(device)

"------------parameters------------"

parser = argparse.ArgumentParser(description='test')
parser.add_argument('-sys', '--system', type=str)
parser.add_argument('-d', '--dim', type=int, default=3)
parser.add_argument('-cmode', '--coords-type', type=int, default=0)
parser.add_argument('-smode', '--sym-type', type=int, default=0)
parser.add_argument('-encmode', '--enc-obj', type=int, default=0)
parser.add_argument('-s', '--seed', type=int)
parser.add_argument('-p', '--prec', nargs='+', type=int, action='append')
# Usage: python script.py --list item1 item2 --list item3
# Result: [['item1', 'item2'], ['item3']]
parser.add_argument('-pl', '--latlen-prec', nargs='+', type=int, default=0)
parser.add_argument('-pa', '--latang-prec', nargs='+', type=int, default=0)
# Usage: python script.py --list item1 item2
# Result: ['item1', 'item2']
parser.add_argument('-pw', '--wpc-prec', type=int, default=0)

parser.add_argument('-eval', '--evaluator', type=str)
parser.add_argument('-sol', '--solver', type=str)
parser.add_argument('-reg', '--regressor', type=str)
parser.add_argument('-f', '--mt-lambda', type=float, default=0)
# parser.add_argument('-c', '--constr', action='store_true')
# parser.add_argument('-b', '--bit', action='store_true')

parser.add_argument('-init', '--init-points', type=int, default=None)
parser.add_argument('-data', '--init-dataset', type=str, default=OptimizerName.wpc_l_rs()[0] + '-' + rlx_type[0])
parser.add_argument('-a', '--add-points', type=int, default=100)
parser.add_argument('-n', '--n-steps', type=int, default=300)

parser.add_argument('-fd', '--is-filter-init-data', action='store_true')
parser.add_argument('-s2d', '--is-2d-scale', action='store_true')
parser.add_argument('-r', '--is-pre-relax', action='store_true')
parser.add_argument('-xy', '--xy-factor', type=float, default=-1)
parser.add_argument('-z', '--z-factor', type=float, default=-1)
parser.add_argument('-ame', '--is-e-amend', action='store_true')
parser.add_argument('-scl', '--is-e-scale', action='store_true')
parser.add_argument('-enl', '--is-enlarge-lat', action='store_true')
parser.add_argument('-sym', '--is-fix-sym', action='store_true')
parser.add_argument('--is-regular-shape', action='store_true')
parser.add_argument('-irn', '--init-rlx-steps', type=int, default=_steps)
parser.add_argument('-gl', '--group-list', nargs='+', type=int, default=0)
parser.add_argument('-dn', '--decode-num', type=int, default=300)

parser.add_argument('-t', '--timeout', type=int, default=None)
parser.add_argument('--fail-num', type=int, default=1)
parser.add_argument('--follow-dir', type=str, default=None)
parser.add_argument('--dir-suffix', type=str, default=None)
parser.add_argument('--grb-license', type=str, default=None)
parser.add_argument('--vasp-incar-sin', type=str, default=None)
parser.add_argument('--vasp-incar-rlx', type=str, action='append', default=['None'])
args = parser.parse_args()

sys_name = args.system
dim = args.dim

emb_type = '+'.join([coords_emb[args.coords_type], sym_emb[args.sym_type], enc_obj[args.enc_obj]])

sol_name = args.solver
learning_model_type = args.regressor
e_cal_name = args.evaluator

seed = args.seed
prec = args.prec
if len(args.prec) == 1:
    prec = prec[0]
if len(prec) == 1:
    prec = prec[0]
len_prec = args.latlen_prec
if len_prec == 0:
    len_prec = None
elif len(len_prec) == 1:
    len_prec = len_prec[0]
ang_prec = args.latang_prec
if ang_prec == 0:
    ang_prec = None
elif len(ang_prec) == 1:
    ang_prec = ang_prec[0]
wpc_prec = args.wpc_prec
if wpc_prec == 0:
    wpc_prec = None

if sol_name in qubo_list:
    use_constraints = True
    is_var_bit = True
else:
    use_constraints = False
    is_var_bit = False
# use_constraints = args.constr
# is_var_bit = args.bit
mt_lambda = args.mt_lambda
timeout = args.timeout

init_pos_num = args.init_points
init_set_type = args.init_dataset
if init_pos_num is None:
    if OptimizerName.uniform_gen()[0] in init_set_type:
        init_pos_num = 50
    else:
        init_pos_num = 1000
iter_sample_num = args.add_points
n_steps = args.n_steps

is_filter_init_data = args.is_filter_init_data
is_pre_relax = args.is_pre_relax
is_2d_scale = args.is_2d_scale
xy_factor, z_factor = args.xy_factor, args.z_factor
is_amend_energy = args.is_e_amend
is_enlarge_lattice = args.is_enlarge_lat
is_scale_energy = args.is_e_scale
is_fix_sym = args.is_fix_sym
decode_num = args.decode_num
init_rlx_steps = args.init_rlx_steps

is_regular_shape = args.is_regular_shape
if is_regular_shape:
    group_list = regular_sg_list(dim=dim)
else:
    group_list = args.group_list
    if isinstance(group_list, int):
        if group_list == 0:
            group_list = None
        else:
            group_list = [group_list]

follow_dir = args.follow_dir
fail_num = args.fail_num
custom_name = args.dir_suffix

incar_rlx_dir = args.vasp_incar_rlx
incar_rlx_dir.remove('None')
if len(incar_rlx_dir) == 0:
    incar_rlx_dir = None
else:
    incar_rlx_dir = [INCAR_ROOT + '/' + incar for incar in incar_rlx_dir]
incar_sin_dir = args.vasp_incar_sin
grb_lic_file = args.grb_license


model = ModelWrapper(embedder_type=emb_type, system_name=sys_name, dim=dim, seed=seed, splits=prec,
                     len_splits=len_prec, ang_splits=ang_prec, wpc_prec=wpc_prec,
                     num_steps=n_steps, follow_dir=follow_dir, custom_name=custom_name, init_sample_rate=1, 
                     init_set_type=init_set_type, iter_sample_num=iter_sample_num, init_pos_num=init_pos_num,
                     
                     e_cal_name=e_cal_name, sol_name=sol_name, learning_model_type=learning_model_type,
                     use_constraints=use_constraints, is_var_bit=is_var_bit, is_pre_relax=is_pre_relax,
                     is_fix_sym=is_fix_sym, init_rlx_steps=init_rlx_steps,
                     
                     is_amend_energy=is_amend_energy, is_enlarge_lattice=is_enlarge_lattice, is_scale_energy=is_scale_energy, 
                     mt_lambda=mt_lambda, is_2d_scale=is_2d_scale, xy_factor=xy_factor, z_factor=z_factor, include_spg_list=group_list,
                     is_filter_init_data=is_filter_init_data, timeout=timeout, decode_num=decode_num, fail_num=fail_num,
                     incar_sin_dir=incar_sin_dir, incar_rlx_dir=incar_rlx_dir, grb_lic_dir=grb_lic_file, lmbda_l=None)
model.test()
