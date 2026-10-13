from crysim.src.registry import OptimizerName, RegressionName, init_set_list, oh_coords_emb, direct_posterior, coords_emb, enc_obj
from crysim.src.crysvec.loader import EmbedderProxy
from crysim.src.evaluator.loader import load_energy_evaluator
from crysim.src.evaluator.e_eval import _steps
from crysim.src.solver.loader import load_solver_or_proxy
from crysim.src.regressor.loader import RegressorProxy
from crysim.src.regressor.bocs_interface import BOCSWrapper, BayesHSWrapper
from crysim.src.optimizer.opt_base import CSP, RandomGenerate4MS, DirectCSP
from crysim.src.optimizer.rand_gen import analyze_dataset_type
from crysim.src.struct_utils import get_species_list
import time
from logzero import logger as log_printer


class ModelWrapper:
    def __init__(self, embedder_type, system_name, dim, seed, is_filter_init_data, e_cal_name, num_steps, follow_dir, 
                 init_set_type, is_amend_energy, is_enlarge_lattice, is_fix_sym, include_spg_list, is_scale_energy=False,
                 mt_lambda=0, custom_name=None, splits=None, len_splits=None, ang_splits=None, wpc_prec=None,
                 is_pre_relax=False, is_2d_scale=False, xy_factor=-1, z_factor=-1, 
                 timeout=None, decode_num=1, fail_num=1, init_sample_rate=None, iter_sample_num=None, init_rlx_steps=_steps,
                 init_pos_num=None, use_constraints=False, is_var_bit=True, learning_model_type=None, sol_name=None,
                 incar_sin_dir=None, incar_rlx_dir=None, grb_lic_dir=None, lmbda_l=None):
        self.embedder_type = embedder_type
        if self.embedder_type is not None:
            self.coords_type, self.sym_type, self.enc_obj = embedder_type.split('+')
            if self.coords_type == coords_emb[0]:
                if self.enc_obj != enc_obj[0]:
                    log_printer.info(f"Switch enc_obj to {enc_obj[0]}; input {self.enc_obj} is not supported with {self.coords_type}")
                    self.enc_obj = enc_obj[0]
                    self.embedder_type = '+'.join([self.coords_type, self.sym_type, self.enc_obj])
            else:
                if self.enc_obj != enc_obj[1]:
                    log_printer.info(f"Switch enc_obj to {enc_obj[1]}; input {self.enc_obj} is not supported with {self.coords_type}")
                    self.enc_obj = enc_obj[1]
                    self.embedder_type = '+'.join([self.coords_type, self.sym_type, self.enc_obj])
        else:
            self.coords_type, self.sym_type, self.enc_obj = None, None, None
        
        # initial dataset
        self.init_set_type = init_set_type
        gen_tool, use_rlx_type, use_mixing = analyze_dataset_type(init_set_type)
        if self.sym_type == 'oh' and learning_model_type is not None:
            assert gen_tool in init_set_list, \
                f"Please select the initial dataset generation method from {init_set_list} to enable WPC index optimization"
        self.include_spg_list = include_spg_list
        
        # system
        self.system_name = system_name
        species_list = get_species_list(self.system_name)
        self.dim = dim
        self.is_2d_scale, self.xy_factor, self.z_factor = is_2d_scale, xy_factor, z_factor
        self.seed = seed
        self.follow_dir = follow_dir
        self.custom_name = custom_name

        self.learning_model_type = learning_model_type
        self.e_cal_name = e_cal_name
        self.sol_name = sol_name
        self.timeout = timeout

        self.is_amend_energy = is_amend_energy
        self.is_filter_dist_for_init = is_filter_init_data
        self.is_scale_energy = is_scale_energy
        self.is_enlarge_lattice = is_enlarge_lattice
        self.mt_lambda = mt_lambda
        self.is_pre_relax = is_pre_relax
        self.is_fix_sym = is_fix_sym
        
        self.fail_num = fail_num
        self.num_steps = num_steps
        self.init_pos_num = init_pos_num
        self.init_sample_rate = init_sample_rate
        self.iter_sample_num = iter_sample_num
        # self.iter_sample_num = iter_sample_num if use_rlx_type != rlx_type[1] else 1
        self.init_rlx_steps = init_rlx_steps

        if self.embedder_type is not None:
            if self.coords_type not in oh_coords_emb and splits is not None:
                log_printer.info(f"The input split {splits} has no effect for the current structure embedder")
            self.embedder_loader = EmbedderProxy(species_list=species_list, emb_type=self.embedder_type, 
                                                 is_var_bit=is_var_bit, dim=self.dim, include_spg_list=self.include_spg_list,
                                                 splits=splits, decode_num=decode_num, use_mixing=use_mixing,
                                                 len_splits=len_splits, ang_splits=ang_splits, wpc_prec=wpc_prec)
        else:
            self.embedder_loader = None

        # ------------------ load components ------------------ "

        if self.learning_model_type in direct_posterior:
            self.reg = RegressorProxy.load(learning_model_type=self.learning_model_type, mt_lambda=mt_lambda)
            if self.coords_type in oh_coords_emb:
                self.reg.is_regression_on_bits(True)
            else:
                self.reg.is_regression_on_bits(False)
        elif self.learning_model_type == RegressionName.bocs()[0]:
            self.reg = BOCSWrapper.load()
        elif self.learning_model_type == RegressionName.bhs()[0]:
            self.reg = BayesHSWrapper.load()
        elif self.learning_model_type is None:
            self.reg = None
        else:
            raise Exception("We don't support the regressor")

        self.e_cal = load_energy_evaluator(self.e_cal_name, dim=self.dim, 
                                           incar_sin_dir=incar_sin_dir, incar_rlx_dir=incar_rlx_dir)

        if sol_name is not None:
            self.sol = load_solver_or_proxy(self.sol_name, timeout=timeout,
                                            use_constraints=use_constraints, mt_lambda=self.mt_lambda,
                                            grb_lic_dir=grb_lic_dir, constraints_lmbda_l=lmbda_l, constraints_lmbda_q=None)
        else:
            self.sol = None

        self._load_optimizer()

        return

    def _load_optimizer(self):
        start_time = time.process_time()
        if self.reg is None and self.sol is None and self.embedder_loader is None:  # solver is not needed for random search           
            self.opt = RandomGenerate4MS(system_name=self.system_name, dim=self.dim, is_scale_2d=self.is_2d_scale,
                                         e_cal=self.e_cal, rs_type=self.init_set_type,
                                         num_steps=self.num_steps, len_lbs=None, len_ubs=None, ang_lbs=None, ang_ubs=None,
                                         xy_factor=self.xy_factor, z_factor=self.z_factor,
                                         is_filter_dist_for_init=self.is_filter_dist_for_init, is_amend_energy=self.is_amend_energy,
                                         is_enlarge_lattice=self.is_enlarge_lattice, is_fix_sym=self.is_fix_sym,
                                         seed=self.seed, include_spg_list=self.include_spg_list, custom_name=self.custom_name)
        elif self.reg is None and self.embedder_loader is None:  # regressor is not needed for direct optimization
            self.opt = DirectCSP(dim=self.dim, model_name=None, real_model_name=None, system_name=self.system_name, 
                                 len_lbs=None, len_ubs=None, ang_lbs=None, ang_ubs=None, is_pre_relax=self.is_pre_relax,
                                 is_scale_2d=self.is_2d_scale, xy_factor=self.xy_factor, z_factor=self.z_factor,
                                 emb_loader=self.embedder_loader, e_cal=self.e_cal, sol_loader=self.sol, init_rlx_steps=self.init_rlx_steps,
                                 is_filter_dist_for_init=self.is_filter_dist_for_init, is_amend_energy=self.is_amend_energy, 
                                 is_scale_energy=self.is_scale_energy, is_enlarge_lattice=self.is_enlarge_lattice, is_fix_sym=self.is_fix_sym,
                                 mt_lambda=self.mt_lambda, num_steps=self.num_steps, init_set_type=self.init_set_type, fail_num=1, 
                                 init_pos_num=self.init_pos_num, init_sample_rate=self.init_sample_rate, iter_sample_num=self.iter_sample_num, 
                                 seed=self.seed, include_spg_list=self.include_spg_list, custom_name=self.custom_name)
        else:
            self.opt = CSP(dim=self.dim, model_name=None, real_model_name=None, system_name=self.system_name, 
                           len_lbs=None, len_ubs=None, ang_lbs=None, ang_ubs=None, is_pre_relax=self.is_pre_relax, 
                           is_scale_2d=self.is_2d_scale, xy_factor=self.xy_factor, z_factor=self.z_factor,
                           emb_loader=self.embedder_loader, e_cal=self.e_cal, sol_loader=self.sol, reg_loader=self.reg, 
                           init_rlx_steps=self.init_rlx_steps, is_filter_dist_for_init=self.is_filter_dist_for_init, is_amend_energy=self.is_amend_energy, 
                           is_scale_energy=self.is_scale_energy, is_enlarge_lattice=self.is_enlarge_lattice, is_fix_sym=self.is_fix_sym,
                           mt_lambda=self.mt_lambda, num_steps=self.num_steps, init_set_type=self.init_set_type, fail_num=1, 
                           init_pos_num=self.init_pos_num, init_sample_rate=self.init_sample_rate, iter_sample_num=self.iter_sample_num, 
                           seed=self.seed, include_spg_list=self.include_spg_list, follow_dir=self.follow_dir, custom_name=self.custom_name)
        log_printer.info(f"load optimizer time: {time.process_time() - start_time}")
        return

    def test(self):
        return self.opt()
