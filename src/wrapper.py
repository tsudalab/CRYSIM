from src.registry import OptimizerName, qubo_list
from src.evaluator.e_eval_loader import load_energy_evaluator
from src.solver.sol_loader import load_solver_or_proxy, load_timeout
from src.regressor.reg_loader import load_regressor
from src.optimizer.opt_base import RandomGenerate4MS
from src.optimizer.opt_cl import SimpleOpt4MS, CrystalParamFitOpt4MS
from src.optimizer.opt_oh import BQM4Onehot, Classical4Onehot
import time
import yaml


class ModelWrapper:
    def __init__(self, opt_name, system_name, seed, e_cal_name, num_steps, follow_dir,
                 precs=10, timeout=None, dist_min=None, wp_sample_num=None, fail_num=1,
                 init_sample_rate=None, iter_sample_num=None, init_pos_num=None,
                 learning_model_type=None,  sol_name=None,
                 filter_gen_struct=False, struct_file_path=None,
                 control_lat_shape=None, control_crys_sys=None, control_spg=None,
                 use_constraints=None):
        # system
        self.system_name = system_name
        self.seed = seed
        self.follow_dir = follow_dir
        start_time = time.process_time()
        # self.struct_file_path = f'./{system_name}_POS_seed{seed}' \
        #     if struct_file_path is None else struct_file_path
        self.struct_file_path = f'dataset/{self.system_name}_training{self.seed}.pkl' \
            if struct_file_path is None else struct_file_path
        load_struct_file_time = time.process_time()
        print(f"load struct file time: {load_struct_file_time - start_time}")

        self.learning_model_type = learning_model_type
        with open("src/regressor/reg_param.yml", "r") as stream:
            params = yaml.safe_load(stream)
        self.main_reg_params = params["reg_param"]

        self.e_cal_name = e_cal_name
        self.sol_name = sol_name
        self.timeout = timeout
        self.opt_name = opt_name

        self.filter_struct = filter_gen_struct
        self.fail_num = fail_num
        self.num_steps = num_steps
        self.init_pos_num = init_pos_num
        self.init_sample_rate = init_sample_rate
        self.iter_sample_num = iter_sample_num

        if opt_name == OptimizerName.onehot()[0]:
            self.precs = precs
            self.wp_sample_num = wp_sample_num
        else:
            print('The parameters "precs", "dist_min" does not have effect')

        if opt_name == OptimizerName.onehot()[0]:
            assert (isinstance(control_lat_shape, bool)) \
                   and (isinstance(control_crys_sys, bool)) \
                   and (isinstance(control_spg, bool)), \
                "Encoding methods should be specified for onehot."
            self.control_lat_shape, self.control_crys_sys, self.control_spg = \
                control_lat_shape, control_crys_sys, control_spg
            if sol_name in qubo_list:
                assert (isinstance(use_constraints, bool)), \
                    "Constraints' implementation methods should be specified for qubo."
                self.dist_min = dist_min
                self.use_constraints = use_constraints
            else:
                assert not use_constraints, \
                    "We only support constraints for qubo solvers."

        else:
            assert (control_lat_shape is None) \
                   and (control_crys_sys is None) \
                   and (control_spg is None) \
                   and (use_constraints is None), \
                "We currently do not support constraints for optimizers except for onehot"
            print('The parameters "dist_min" does not have effect')

        # ------------------ load components ------------------ "

        self.reg = load_regressor(learning_model_type=self.learning_model_type,
                                  reg_arg=self.main_reg_params) \
            if self.learning_model_type is not None else None

        self.e_cal = load_energy_evaluator(self.e_cal_name)

        if sol_name is not None:
            self.sol = load_solver_or_proxy(self.sol_name, self.opt_name,
                                            timeout=timeout, use_constraints=use_constraints)
        else:
            self.sol = None

        self._load_optimizer()

        return

    def _load_optimizer(self):
        start_time = time.process_time()
        if self.opt_name == OptimizerName.random_search()[0]:  # solver is not needed for random search
            self.opt = RandomGenerate4MS(system_name=self.system_name, e_cal=self.e_cal,
                                         seed=self.seed, num_steps=self.num_steps,
                                         init_pos_num=self.init_pos_num,
                                         struct_file_path=self.struct_file_path,
                                         filter_gen_struct=self.filter_struct,
                                         follow_dir=self.follow_dir)

        elif self.opt_name == OptimizerName.crystal_params_vector()[0]:
            self.sol = load_timeout(self.sol, self.num_steps)
            self.opt = SimpleOpt4MS(system_name=self.system_name, sol=self.sol, e_cal=self.e_cal,
                                    seed=self.seed, num_steps=self.num_steps,
                                    init_pos_num=self.init_pos_num,  # for deciding maximum lattice length
                                    struct_file_path=self.struct_file_path,
                                    filter_gen_struct=self.filter_struct,
                                    fail_num=self.fail_num,
                                    follow_dir=self.follow_dir)

        elif self.opt_name == OptimizerName.crystal_params_vector_opt()[0]:
            timeout = 'large' if self.timeout is None else self.timeout
            self.sol = load_timeout(self.sol, timeout)
            self.opt = CrystalParamFitOpt4MS(system_name=self.system_name,
                                             sol=self.sol, e_cal=self.e_cal, reg=self.reg,
                                             seed=self.seed, num_steps=self.num_steps,
                                             init_pos_num=self.init_pos_num,
                                             init_sample_rate=self.init_sample_rate,
                                             iter_sample_num=self.iter_sample_num,
                                             struct_file_path=self.struct_file_path,
                                             filter_gen_struct=self.filter_struct,
                                             fail_num=self.fail_num,
                                             follow_dir=self.follow_dir)

        elif self.opt_name == OptimizerName.onehot()[0]:
            if self.sol_name in qubo_list:
                self.opt = BQM4Onehot(precs=self.precs, dist_min=self.dist_min, wp_sample_num=self.wp_sample_num,
                                      control_lat_shape=self.control_lat_shape,
                                      control_crys_sys=self.control_crys_sys,
                                      control_spg=self.control_spg,
                                      use_constraints=self.use_constraints,
                                      system_name=self.system_name,
                                      sol=self.sol, e_cal=self.e_cal, reg=self.reg,
                                      init_pos_num=self.init_pos_num,
                                      init_sample_rate=self.init_sample_rate,
                                      iter_sample_num=self.iter_sample_num,
                                      seed=self.seed, num_steps=self.num_steps,
                                      struct_file_path=self.struct_file_path,
                                      filter_gen_struct=self.filter_struct,
                                      fail_num=self.fail_num,
                                      follow_dir=self.follow_dir)
            else:
                timeout = 'middle' if self.timeout is None else self.timeout
                self.sol = load_timeout(self.sol, timeout)
                self.opt = Classical4Onehot(precs=self.precs, wp_sample_num=self.wp_sample_num,
                                            control_lat_shape=self.control_lat_shape,
                                            control_crys_sys=self.control_crys_sys,
                                            control_spg=self.control_spg,
                                            system_name=self.system_name,
                                            sol=self.sol, e_cal=self.e_cal, reg=self.reg,
                                            init_pos_num=self.init_pos_num,
                                            init_sample_rate=self.init_sample_rate,
                                            iter_sample_num=self.iter_sample_num,
                                            seed=self.seed, num_steps=self.num_steps,
                                            struct_file_path=self.struct_file_path,
                                            filter_gen_struct=self.filter_struct,
                                            fail_num=self.fail_num,
                                            follow_dir=self.follow_dir)

        else:
            raise NotImplementedError
        print(f"load optimizer time: {time.process_time() - start_time}")
        return

    def test(self):
        return self.opt()
