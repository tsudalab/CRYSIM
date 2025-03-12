import numpy as np
import time
import hyperopt as hy

from src.optimizer.opt_base import ContinuousModel, CSP, Classical4CSP, \
    energy_evaluation_and_get_samples, filter_struct_wrt_pair_dist
from src.registry import OptimizerName
from src.spg.spg_utils import get_spg_list


class Continuous4MS(ContinuousModel, Classical4CSP):
    def __init__(self, model_name, real_model_name, system_name, sol, e_cal,
                 num_steps, init_pos_num, init_sample_rate, iter_sample_num,
                 struct_file_path, seed, filter_gen_struct, fail_num, follow_dir):
        sol_name, real_solver_name = sol.name, sol.real_name
        self.csp = CSP(model_name=model_name, real_model_name=real_model_name,
                       system_name=system_name, num_steps=num_steps,
                       init_pos_num=init_pos_num, init_sample_rate=init_sample_rate,
                       iter_sample_num=iter_sample_num,
                       sol_name=sol_name, real_solver_name=real_solver_name,
                       struct_file_path=struct_file_path, seed=seed, e_cal=e_cal,
                       filter_gen_struct=filter_gen_struct, fail_num=fail_num, follow_dir=follow_dir)
        ContinuousModel.__init__(self, atom_num=self.csp.atom_num, species=self.csp.species,
                                 num_species_atom=self.csp.num_species_atom,
                                 init_spg_list=self.csp.spg_gen,
                                 init_wp_list=self.csp.wp_gen, struct_list=self.csp.struct_list,
                                 follow_dir=follow_dir, logger=self.csp.logger)
        Classical4CSP.__init__(self, csp=self.csp, sol=sol)
        self.spg_list = get_spg_list(self.wyckoffs_dict)

        self.csp.struct_e_list = list(self.point_list_y) if follow_dir is not None else self.csp.struct_e_list
        return


class SimpleOpt4MS(Continuous4MS):
    def __init__(self, system_name, sol, e_cal, num_steps, init_pos_num,
                 struct_file_path, seed, filter_gen_struct, fail_num, follow_dir):
        model_name = OptimizerName.crystal_params_vector()[0]
        real_model_name = OptimizerName.crystal_params_vector()[1]
        super().__init__(model_name=model_name, real_model_name=real_model_name,
                         system_name=system_name, sol=sol, e_cal=e_cal,
                         num_steps=num_steps,
                         init_pos_num=init_pos_num, init_sample_rate=1, iter_sample_num=1,
                         struct_file_path=struct_file_path, seed=seed,
                         filter_gen_struct=filter_gen_struct, fail_num=fail_num, follow_dir=follow_dir)
        return

    def _build_solver(self):
        self.sol.get_params(lat_dim=self.lat_dim, ang_dim=self.ang_dim, pos_dim=self.pos_dim,
                            ll=self.lower_bound(), ul=self.upper_bound(self.lat_max),
                            seed=self.csp.seed, spg_list=self.spg_list)
        self.sol_real = self.sol
        return

    def objective_function(self, x):
        lat_para = np.array([x['l0'], x['l1'], x['l2'], x['a0'], x['a1'], x['a2']])
        atoms = np.array([x[f'x{i}'] for i in range(self.pos_dim)])
        atoms = np.reshape(atoms, (-1, 3))
        spg, wp = x['spg'], x['wp']
        self.csp.logger.record_anything1(operation='Decoded Space Group', result=spg)
        self.csp.logger.record_anything2(operation='Decoded Wyckoff Position', result=wp)
        pmg_struct = self.vec2struct(lat_para=lat_para, pos=atoms, spg=spg, wp=wp)
        self.csp.logger4csp.save_init_struct(pmg_struct, idx=self.csp.logger.new_struct_suc, idx_off=0)

        r_struct, energy = energy_evaluation_and_get_samples(pmg_struct, csp=self.csp)
        if self.csp.filter_struct:
            r_struct, energy = filter_struct_wrt_pair_dist(csp=self.csp, pmg_struct_list=r_struct, energy_list=energy)

        self.csp.success_step(self.csp.logger.new_struct_suc, r_struct, energy, None)

        return {'loss': energy[0], 'status': hy.STATUS_OK}

    def _init_model(self):
        start_time = time.process_time()
        print("Start building model")
        self._build_solver()
        self.csp.logger.record_modelling1_time(operation='Initialization',
                                               delta_t=time.process_time() - start_time)
        return

    def run(self):
        self._init_model()
        print("Start running")
        self.sol.solve(objective_function=self.objective_function)
        return


class CrystalParamFitOpt4MS(Continuous4MS):
    def __init__(self, system_name, sol, e_cal, reg,
                 num_steps, init_pos_num, init_sample_rate, iter_sample_num,
                 struct_file_path, seed, filter_gen_struct, fail_num, follow_dir):

        model_name = OptimizerName.crystal_params_vector_opt()[0] + f'_{reg.reg_name[0]}'
        real_model_name = OptimizerName.crystal_params_vector_opt()[1] + f'_{reg.reg_name[1]}'
        super().__init__(model_name=model_name, real_model_name=real_model_name,
                         system_name=system_name, sol=sol, e_cal=e_cal,
                         num_steps=num_steps, init_pos_num=init_pos_num,
                         init_sample_rate=init_sample_rate, iter_sample_num=iter_sample_num,
                         struct_file_path=struct_file_path, seed=seed,
                         filter_gen_struct=filter_gen_struct, fail_num=fail_num, follow_dir=follow_dir)
        self.reg = reg
        return

    def _build_simulator(self):
        self.reg.get_param(X=self.point_list, y=np.array(self.csp.struct_e_list),
                           for_binary_model=False, logger=self.csp.logger)
        self.model = self.reg.unwrap()
        return

    def _build_solver(self):
        self.sol.get_params(lat_dim=self.lat_dim, ang_dim=self.ang_dim, pos_dim=self.pos_dim,
                            ll=self.lower_bound(), ul=self.upper_bound(self.lat_max),
                            seed=self.csp.seed, spg_list=self.spg_list,
                            learning_model=self.reg.learning_model, q_from_dict=False,
                            interaction_terms=self.reg.interaction_terms)
        self.sol_real = self.sol
        return

    def _init_model(self):
        start_time = time.process_time()
        print("Start building model")
        self._build_simulator()
        self._build_solver()
        self.csp.logger.record_modelling1_time(operation='Initialization',
                                               delta_t=time.process_time() - start_time)
        return

    def _model_learn(self):
        self.model.learn_weights()
        return

    def _build_objective_function(self):
        q = self.model.get_weight()
        return q

    def _model_decode(self, new_x):
        pmg_struct, new_x = self.get_struct_from_vec(new_x)
        return pmg_struct, new_x

    def _model_relaxed_encode(self, r_pmg_struct, new_x):
        return self.get_vec_from_struct(r_pmg_struct=r_pmg_struct, new_x=new_x)

    def _model_add_points(self, r_new_x, energy):
        self.model.add_data(np.vstack(r_new_x), energy)
        return
