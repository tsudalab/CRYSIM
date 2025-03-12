import numpy as np

from src.optimizer.opt_base import CSP, BQM4CSP, Classical4CSP
from src.registry import OptimizerName, RegressionName
from src.log_analysis import load_training_set
from src.onehot.oh import OneHotDirector, OneHotConstraintsType, OneHotImplementationType
from src.struct_utils import SPG, get_best_crys_sys2, get_best_spg2
from src.regressor.poly_sim import fm_objective_function


class Onehot4CSP(CSP):
    def __init__(self, model_name, real_model_name,
                 precs, dist_min, wp_sample_num,

                 control_lat_shape, control_crys_sys, control_spg,
                 use_constraints,

                 system_name, sol_name, real_solver_name, e_cal, reg,
                 init_pos_num, init_sample_rate, iter_sample_num,
                 num_steps, struct_file_path, seed,
                 filter_gen_struct, fail_num, follow_dir):
        """ Usually we will use constraints and num_var is calculated and passed to the solver
            for it to instantiate. Therefore, proxy is given as the solver object instead of
            the solver itself in order to postpone the actual instantiation of the solver from the
            composing process (in the wrapper.py script) to here. """
        assert wp_sample_num > 0
        if not filter_gen_struct:
            assert wp_sample_num == 1

        self.oht = OneHotConstraintsType([control_lat_shape, control_crys_sys, control_spg])
        self.ohi = OneHotImplementationType(use_constraints)
        self.constraints_name = self.oht.constraints_name
        self.constraints_scheme = self.oht.constraints_scheme
        self.implementation_name = self.ohi.implementation_name
        self.implementation_scheme = self.ohi.implementation_scheme
        if self.constraints_scheme not in [3, 4]:
            assert wp_sample_num == 1

        self.reg = reg

        model_name = model_name + '-' + f'{precs}-{dist_min}-{wp_sample_num}-' \
                                        f'{self.constraints_name}_{reg.reg_name[0]}'
        real_model_name += '-' + f'Precs{precs}-MinDist{dist_min}-SplNum{wp_sample_num}-' \
                                 f'Cons{self.constraints_name}_{reg.reg_name[1]}'

        super().__init__(model_name=model_name,
                         real_model_name=real_model_name,
                         system_name=system_name, num_steps=num_steps, init_pos_num=init_pos_num,
                         init_sample_rate=init_sample_rate, iter_sample_num=iter_sample_num,
                         struct_file_path=struct_file_path, seed=seed, e_cal=e_cal,
                         sol_name=sol_name, real_solver_name=real_solver_name, fail_num=fail_num,
                         filter_gen_struct=filter_gen_struct, follow_dir=follow_dir)

        self.learning_model_order = reg.learning_model
        self.interaction_terms = reg.interaction_terms

        self.precs = precs
        self.dist_min = dist_min
        self.wp_sample_num = wp_sample_num

        return

    def init_model(self, lat_scale=0.1):
        lat_list = np.concatenate([struct.lattice.matrix.flatten() for struct in self.struct_list])
        lat_max = np.max(np.abs(lat_list)) * (1 + lat_scale)
        default_cs = get_best_crys_sys2(spg_list=self.spg_gen, e_list=self.struct_e_list)
        default_spg, default_wp = get_best_spg2(spg_list=self.spg_gen, wp_list=self.wp_gen,
                                                e_list=self.struct_e_list)
        self.oh = OneHotDirector(species_list=self.species,
                                 splits=[self.precs, self.precs, self.precs],
                                 lat_ub=lat_max,
                                 dist_min=self.dist_min,
                                 constraint_type=self.oht,
                                 implement_type=self.ohi,
                                 logger=self.logger,
                                 default_crys_sys=default_cs,
                                 default_spg=default_spg, default_wp=default_wp)
        self.oh_real = self.oh.unwrap()
        if self.implementation_scheme == 0:
            self.ohp = self.oh_real
        elif self.implementation_scheme == 1:
            self.ohp = self.oh_real.get_ohp()
        else:
            raise Exception('We do not support this implementation scheme')
        return

    def encode_x(self):
        if self.constraints_scheme in [3, 4]:
            x = [self.ohp.one_hot_encode([self.struct_list[i],
                                          self.spg_gen[i],
                                          self.wp_gen[i]])
                 for i in range(len(self.struct_list))]
        else:
            x = [self.ohp.one_hot_encode(struct) for struct in self.struct_list]
        return x

    def build_simulator(self, encoded_x):
        if self.logger.follow_dir is None:
            x = np.stack(encoded_x)  # (num_init_struct, len_struct_vec)
        else:
            x, y = load_training_set(self.logger.follow_dir)
            self.struct_e_list = list(y)
        self.reg.get_param(X=x, y=np.array(self.struct_e_list),
                           for_binary_model=True, logger=self.logger)
        self.model = self.reg.unwrap()
        return

    def model_learn(self):
        self.model.learn_weights()
        return

    def model_decode(self, new_x):
        if self.wp_sample_num == 1:
            pmg_struct = self.ohp.one_hot_decode(new_x)
            self._detect_decoded_struct_symmetry(pmg_struct)
        else:
            pmg_struct = self.ohp.decode_more_wp(new_x, self.wp_sample_num)
            self._detect_decoded_struct_symmetry(pmg_struct[0])
        return pmg_struct, new_x

    def _detect_decoded_struct_symmetry(self, pmg_struct):
        if self.constraints_scheme == 2:
            real_crys_sys = SPG.struct_bravais_estimate(pmg_struct, None)
            self.logger.record_anything4(operation='Generated Crystal System', result=real_crys_sys)
        if self.constraints_scheme in [3, 4]:
            _, real_spg_n = SPG.struct_spg_estimate(pmg_struct)
            self.logger.record_anything4(operation='Generated Space Group', result=real_spg_n)
        return

    def model_relaxed_encode(self, r_pmg_struct, new_x):
        r_struct_list = []
        if self.constraints_scheme == 3:
            spg, wp = self.ohp.space_group_decode(new_x, get_list=False)
            for rp_s in r_pmg_struct:
                r_struct_list.append(self.ohp.one_hot_encode([rp_s, spg, wp]))
        elif self.constraints_scheme == 4:
            cs, spg, wp = self.ohp.cs_spg_wp_decode(new_x, get_list=False)
            for rp_s in r_pmg_struct:
                r_struct_list.append(self.ohp.one_hot_encode([rp_s, cs, spg, wp]))
        else:
            for rp_s in r_pmg_struct:
                r_struct_list.append(self.ohp.one_hot_encode(rp_s))
        return r_struct_list


class BQM4Onehot(BQM4CSP):
    def __init__(self, precs, dist_min, wp_sample_num,
                 control_lat_shape, control_crys_sys, control_spg,
                 use_constraints,
                 system_name, sol, e_cal, reg,
                 init_pos_num, init_sample_rate, iter_sample_num,
                 num_steps, struct_file_path, seed,
                 filter_gen_struct, fail_num, follow_dir):

        sol_name, real_solver_name = sol.name, sol.real_name
        onehot_csp = Onehot4CSP(model_name=self.model_name[0],
                                real_model_name=self.model_name[1],
                                precs=precs, dist_min=dist_min, wp_sample_num=wp_sample_num,

                                control_lat_shape=control_lat_shape,
                                control_crys_sys=control_crys_sys,
                                control_spg=control_spg,
                                use_constraints=use_constraints,

                                system_name=system_name, e_cal=e_cal, reg=reg,
                                sol_name=sol_name, real_solver_name=real_solver_name,
                                init_pos_num=init_pos_num,
                                init_sample_rate=init_sample_rate, iter_sample_num=iter_sample_num,
                                num_steps=num_steps, struct_file_path=struct_file_path, seed=seed,
                                filter_gen_struct=filter_gen_struct, fail_num=fail_num, follow_dir=follow_dir)
        super().__init__(csp=onehot_csp, sol=sol)
        return

    @property
    def model_name(self):
        return OptimizerName.onehot()[0], OptimizerName.onehot()[1]

    def _build_solver(self):
        self.sol.get_num_var(self.csp.ohp.num_var)
        if self.csp.implementation_scheme == 1:
            lat_mat, atom_mat = self.csp.oh_real.get_constraints()
            self.sol.get_constraints_coefficient_matrix(coef_mat_l=lat_mat, coef_mat_q=atom_mat)
        self.sol_unwrap()
        return

    def _build_bqm(self):
        return self._build_bqm_basic(self.csp.model, self.csp.ohp.num_var,
                                     self.csp.learning_model_order,
                                     self.csp.interaction_terms)


class Classical4Onehot(Classical4CSP):
    def __init__(self, precs, wp_sample_num,
                 system_name, sol, e_cal, reg,
                 init_pos_num, init_sample_rate, iter_sample_num,
                 num_steps, struct_file_path, seed, filter_gen_struct, fail_num, follow_dir,
                 control_lat_shape=None, control_crys_sys=None, control_spg=None):

        sol_name, real_solver_name = sol.name, sol.real_name
        onehot_csp = Onehot4CSP(model_name=self.model_name[0],
                                real_model_name=self.model_name[1],
                                precs=precs, dist_min=None, wp_sample_num=wp_sample_num,

                                control_lat_shape=control_lat_shape,
                                control_crys_sys=control_crys_sys,
                                control_spg=control_spg,
                                use_constraints=False,

                                system_name=system_name, e_cal=e_cal, reg=reg,
                                sol_name=sol_name, real_solver_name=real_solver_name,
                                init_pos_num=init_pos_num,
                                init_sample_rate=init_sample_rate, iter_sample_num=iter_sample_num,
                                num_steps=num_steps, struct_file_path=struct_file_path, seed=seed,
                                filter_gen_struct=filter_gen_struct, fail_num=fail_num, follow_dir=follow_dir)
        super().__init__(csp=onehot_csp, sol=sol)
        return

    @property
    def model_name(self):
        return OptimizerName.onehot()[0], OptimizerName.onehot()[1]

    def _build_solver(self):
        if self.csp.learning_model_order == "fm":
            self.sol.get_params(num_var=self.csp.ohp.num_var,
                                learning_model=self.csp.learning_model_order,
                                seed=self.csp.seed,
                                q_from_dict=True,
                                interaction_terms=self.csp.interaction_terms)
        else:
            self.sol.get_params(num_var=self.csp.ohp.num_var,
                                learning_model=self.csp.learning_model_order,
                                seed=self.csp.seed,
                                q_from_dict=False,
                                interaction_terms=self.csp.interaction_terms)
        self.sol_real = self.sol
        return

    def _build_objective_function(self):
        w = self.csp.model.get_weight()
        # if self.csp.learning_model_order == RegressionName.fm()[0]:
        #     w = fm_objective_function(w, as_dict=True)
        return w
