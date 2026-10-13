import numpy as np
import pandas as pd
import time
from pathlib import Path
import pickle, bz2
from functools import partial
import torch
from copy import deepcopy
from logzero import logger as log_printer
from pymatgen.core import Lattice
from sklearn.model_selection import train_test_split

from crysim.src import DATASET_ROOT, DEBUG
from crysim.src.optimizer.rand_gen import RandGenDirector, RGEnergyDirector, get_training_set, analyze_dataset_type
from crysim.src.evaluator.e_eval import average_force, _steps
from crysim.src.log_analysis import Logger, Logger4CSP, load_training_set, load_spg_dict
from crysim.src.registry import Component, OptimizerName, EnergyEvaluatorName, \
    rlx_type, direct_list, direct_posterior, short_trj_step
from crysim.src.struct_utils import get_species_list, atoms_overlapping, DistFilter, SPG, \
    unique_elements, min_pair_dist_mat, pick, mid_scale_for_system, \
    dim2pbc, add_vacuum_to_pmg, reduce_vacuum_from_pmg, default_ang_lb, default_ang_ub, _vacuum_thickness, _hard_min_mid
from crysim.src.spg.spg_utils import get_spg_list, get_spg_list2, get_wp_list2, wpc_idx_to_specific_wpc_of_sg
from crysim.src.registry import mixing
from crysim.src.crysvec.emb_oh_mix import RandGenMixFilter


# force_limit = 100.0
force_limit = 10.0


def scale_lat_len_for_2d(elements, system_size, xy_factor, z_factor):
    mid_lb, mid_ub = mid_scale_for_system(element_list=elements)
    z_2d_ub = z_factor * float(mid_ub)
    xy_2d_lb = xy_factor * float(mid_lb) * np.sqrt(int(system_size / 2))
    z_2d_ub_relax = float(mid_ub) * 2.0
    return xy_2d_lb, z_2d_ub, z_2d_ub_relax


class CSP(Component):
    def __init__(self, dim, model_name, real_model_name, system_name, 
                 len_lbs, len_ubs, ang_lbs, ang_ubs, is_filter_dist_for_init, include_spg_list,
                 emb_loader, e_cal, sol_loader, reg_loader, is_scale_2d, is_pre_relax, xy_factor, z_factor,
                 is_amend_energy, is_scale_energy, is_enlarge_lattice, is_fix_sym,
                 mt_lambda, num_steps, init_set_type, fail_num, init_pos_num, init_rlx_steps,
                 init_sample_rate, iter_sample_num, seed, follow_dir, custom_name):
        
        self.system_name = system_name
        self.species = get_species_list(self.system_name)
        self.elements, self.num_species_atom = unique_elements(species_list=self.species)
        self.atom_num = int(np.sum(self.num_species_atom))
        self.dim = dim
        self.include_spg_list = include_spg_list
        
        self.len_lbs = np.array([None, None, None]) if len_lbs is None else np.array(len_lbs)
        self.len_ubs = np.array([None, None, None]) if len_ubs is None else np.array(len_ubs)
        self.is_scale_2d = is_scale_2d
        if dim == 2 and is_scale_2d:
            xy_factor = 1.0 if xy_factor < 0 else xy_factor
            z_factor = 1.5 if z_factor < 0 else z_factor
            xy_2d_lb, z_2d_ub, z_2d_ub_relax = scale_lat_len_for_2d(self.elements, len(self.species), xy_factor, z_factor)
            # self.z_2d_ub_relax = z_2d_ub_relax
            if self.len_lbs[0] is not None or self.len_lbs[1] is not None or self.len_ubs[2] is not None:
                log_printer.warning("Provided lattice length bounds will be covered for scaling thickness")
            self.len_lbs[0], self.len_lbs[1] = xy_2d_lb, xy_2d_lb
            self.len_ubs[2] = z_2d_ub
        # self.ang_lbs, self.ang_ubs = arrange_lat_bounds(ang_lbs, ang_ubs, default_ang_lb, default_ang_ub)
        self.ang_lbs = np.array([None, None, None]) if ang_lbs is None else np.array(ang_lbs)
        self.ang_ubs = np.array([None, None, None]) if ang_ubs is None else np.array(ang_ubs)
        
        self.emb_loader = emb_loader
        if self.emb_loader is not None:
            name = self.emb_loader.model_name + f'-dim{dim}'
            real_name = self.emb_loader.real_model_name + f'-dim{dim}'
        else:
            name, real_name = model_name, real_model_name
        if dim == 2 and is_scale_2d:
            name += f'-2dscl-{xy_factor}-{z_factor}'.replace('.', 'p')
            real_name += f'-2dscl-{xy_factor}-{z_factor}'.replace('.', 'p')
        if is_pre_relax:
            name += '_e'
            real_name += '_e'
        else:
            name += '_mid'
            real_name += '_mid'
            if is_fix_sym:
                name += '_fs'
                real_name += '_fs'
        if is_enlarge_lattice:
            name += '_enl'
            real_name += '_enl'
        if is_amend_energy:
            name += '_ame'
            real_name += '_ame'
        if is_scale_energy:
            name += '_scl'
            real_name += '_scl'
        if mt_lambda > 0:
            name += '_f'
            real_name += '_f'
        
        super().__init__(name=name, real_name=real_name)

        # basic components
        self.e_cal = e_cal
        self.sol_loader = sol_loader
        self.reg_loader = reg_loader
        self.init_set_type = init_set_type
        self.gen_tool, self.use_rlx_type, self.use_mixing = analyze_dataset_type(self.init_set_type)
        self.seed = seed
        self.is_filter_dist_for_init = is_filter_dist_for_init
        self.is_fix_test_set = False

        # basic practical settings
        self.is_amend_energy = is_amend_energy
        self.is_scale_energy = is_scale_energy
        self.is_enlarge_lattice = is_enlarge_lattice
        self.mt_lambda = mt_lambda
        self.is_pre_relax = is_pre_relax
        self.is_fix_sym = is_fix_sym
        self.norm_energy = True

        self.num_steps = num_steps
        self.sample_time = 1
        if hasattr(self.reg_loader, 'sample_time'):
            self.sample_time = self.reg_loader.sample_time
        self.init_pos_num = init_pos_num
        self.init_sample_rate = init_sample_rate
        self.iter_sample_num = iter_sample_num
        self.init_rlx_steps = init_rlx_steps
        if self.use_rlx_type == rlx_type[2]:
            self.iter_sample_num = 1
        self.len_scale = 0.1

        # output name
        self._load_output_name()

        # training instances & labels for ML
        self.struct_list = []
        self.struct_e_list = []
        self.e_sampling_init_threshold = 10.0
        self.e_sampling_threshold = self.e_sampling_init_threshold
        self.fail_num = max([int(self.iter_sample_num / 3), 1]) if (fail_num is None) or (fail_num < 1) else fail_num

        # logger
        self.logger = Logger(self.output_name, system_name, n_steps=num_steps,
                             seed=self.seed, follow_dir=follow_dir, custom_name=custom_name)
        self.logger.record_dataset_settings(self.init_pos_num, self.iter_sample_num, self.fail_num)
        self.logger4csp = Logger4CSP(logger_obj=self.logger)
        if self.include_spg_list is not None:
            log_printer.info(f"Only consider space/layer/rod/point groups: {self.include_spg_list}")
        if self.use_rlx_type == rlx_type[2]:
            log_printer.info("# of added points in each iteration is set as 1 for the i2r setting")
        if self.use_rlx_type == rlx_type[0]:
            log_printer.info("Increase ub of lat encoding for enlarged lattices of relaxed structures")
        log_printer.info(f"Each structure will be relaxed for {self.init_rlx_steps} steps during relaxation")
        log_printer.info(f"{self.is_fix_sym} try to maintain symmetry during relaxation")
        if self.is_pre_relax and (not self.is_fix_sym):
            self.is_fix_sym = True
            log_printer.info("Screening by short traj requires fixing symmetry in relaxation; forcely set is_fix_sym=True")

        # save dir for traj files
        self.e_cal.set_traj_logger(self.logger)
        # save dir for VASP outputs
        if self.e_cal.name == EnergyEvaluatorName.vasp():
            self.e_cal.set_logger(self.logger)
        # set loggers
        if self.reg_loader is not None:
            self.reg_loader.get_logger(self.logger)

        return

    def _load_output_name(self):
        self.output_name = f'{self.name}_{self.init_set_type}{self.init_pos_num}'
        self.output_name += f'_{self.reg_loader.reg_name[0]}_{self.sol_loader.name}_{self.e_cal.name}'
        self.real_output_name = f'{self.real_name}_{self.init_set_type}{self.init_pos_num}'
        self.real_output_name += f'_{self.reg_loader.reg_name[1]}_{self.sol_loader.real_name}_{self.e_cal.real_name}'
        return

    def _update_sampling_threshold(self):
        if len(self.struct_e_list) == 0:
            self.e_sampling_threshold = self.e_sampling_init_threshold
        else:
            if self.is_scale_energy:
                self.e_sampling_threshold = np.abs(np.min(self.struct_e_list))
            else:
                self.e_sampling_threshold = np.max(self.struct_e_list)  # stronger punishment on small MID + low energy
        return

    def _generate(self, id_offset, nstruct, filter_dist, seed=None):
        if not self.use_mixing:
            rg = RandGenDirector(
                system_name=self.system_name, gen_tool=self.gen_tool, n_struct=nstruct, dim=self.dim,
                len_lbs=self.len_lbs, len_ubs=self.len_ubs, ang_lbs=self.ang_lbs, ang_ubs=self.ang_ubs, 
                seed=seed, filter_dist=filter_dist, init_spg_list=self.include_spg_list, logger=self.logger, 
                file_root=DATASET_ROOT, load_cache=True, custom_name=None, id_offset_for_rg=id_offset,
                is_insist_wpclist=True)
        else:
            rg = RandGenMixFilter(
                system_name=self.system_name, gen_tool=self.gen_tool, n_struct=nstruct, dim=self.dim,
                len_lbs=self.len_lbs, len_ubs=self.len_ubs, ang_lbs=self.ang_lbs, ang_ubs=self.ang_ubs, 
                seed=seed, filter_dist=filter_dist, init_spg_list=self.include_spg_list, logger=self.logger, 
                file_root=DATASET_ROOT, load_cache=True, id_offset_for_rg=id_offset)
        pmg_struct_list, used_atoms, spg_gen, wp_gen = rg()
        return pmg_struct_list, used_atoms, spg_gen, wp_gen
    
    def load_dataset_obj(self, nstruct, load_cache):
        custom_name = mixing if self.use_mixing else None
        rge = RGEnergyDirector(
            system_name=self.system_name, dim=self.dim, gen_tool=self.gen_tool, use_rlx_type=self.use_rlx_type,
            len_lbs=self.len_lbs, len_ubs=self.len_ubs, ang_lbs=self.ang_lbs, ang_ubs=self.ang_ubs, rlx_steps=self.init_rlx_steps,
            n_struct=nstruct, e_cal=self.e_cal, seed=self.seed, init_spg_list=self.include_spg_list, custom_name=custom_name,
            struct_file_root=DATASET_ROOT, load_cache=load_cache, logger=self.logger)
        return rge

    def load_dataset(self, load_cache=True):
        init_gen_struct_list, _, _, _ = self._generate(id_offset=0, nstruct=self.init_pos_num, filter_dist=self.is_filter_dist_for_init, seed=self.seed)
        self.dataset_struct_num = len(init_gen_struct_list)
        rge = self.load_dataset_obj(nstruct=self.init_pos_num, load_cache=load_cache)
        data = rge()
        log_printer.info(f'Load dataset')
        self.struct_e_list = [i for i in data['struct_e_list'] if i is not None]
        self._update_sampling_threshold()
        self.struct_list, self.struct_e_list, self.struct_f_list, self.used_atoms, self.spg_gen, self.wp_gen = \
            get_training_set(rge, 
                             e_sampling_threshold=self.e_sampling_threshold,
                            #  e_sampling_threshold=99999
                             )
        # self.struct_f_list = [average_force(f) for f in self.struct_f_list]
        log_printer.info(f'Dataset size: {len(self.struct_list)}')
        low_e_idx = np.argmin(self.struct_e_list)
        log_printer.info(f'Current best: {self.struct_e_list[low_e_idx]} with group {self.spg_gen[low_e_idx]}')
        
        if self.dim < 3:
            pbc = dim2pbc(self.dim)
            self.struct_list = [reduce_vacuum_from_pmg(s, pbc=pbc) for s in self.struct_list]
            self.add_vacuum_to_pmg_fn = partial(add_vacuum_to_pmg, consider_pbc_in_unit=not self.use_mixing)
        return
    
    def load_lat_ub_and_lb(self):
        # get unified lattice bounds from initial datasets
        len_list = []    
        for s in self.struct_list:
            len_list.append(s.lattice.lengths)
        len_list = np.vstack(len_list)
        len_min_from_data = np.min(np.abs(len_list), axis=0)
        len_max_from_data = np.max(np.abs(len_list), axis=0) * (1 + self.len_scale)
        self.len_lbs = np.where(self.len_lbs == None, len_min_from_data, self.len_lbs).astype(float)
        self.len_ubs = np.where(self.len_ubs == None, len_max_from_data, self.len_ubs).astype(float)
        self.ang_lbs = np.where(self.ang_lbs == None, default_ang_lb, self.ang_lbs).astype(float)
        self.ang_ubs = np.where(self.ang_ubs == None, default_ang_ub, self.ang_ubs).astype(float)
        return
    
    """ encode structures & build dataset for training """
    
    def encode_dataset(self):
        # already remove vacuums
        x_init = [self.embedder.encode([self.struct_list[i], self.used_atoms[i], self.spg_gen[i], self.wp_gen[i]]) 
                  for i in range(len(self.struct_list))]
        return x_init

    def filter_duplicated_embedded_struct(self, x_list, energy):
        """ Filter duplicated one-hot structure arrays, and leave the one with the lowest energy """
        vec_list = np.vstack(x_list)
        data = np.hstack([np.arange(len(x_list)).reshape(-1, 1), vec_list, np.array(energy).reshape(-1, 1)])
        oh_col = list(range(vec_list.shape[1]))
        oh_pd = pd.DataFrame(data=data, columns=['original_index'] + oh_col + ['energy'])
        oh_pd_left = oh_pd.groupby(oh_col)['energy'].idxmin()
        oh_pd_left = oh_pd.loc[oh_pd_left]
        struct_pd, energy_pd = oh_pd_left.loc[:, oh_col], oh_pd_left.loc[:, 'energy']
        struct_list_uni = struct_pd.to_numpy().tolist()
        energy_uni = energy_pd.to_numpy().tolist()
        selected_indices = oh_pd_left['original_index'].astype(int).tolist()
        log_printer.info(f"Reduce additional samples from {len(x_list)} to {len(struct_list_uni)} by removing duplicates")
        return struct_list_uni, energy_uni, selected_indices
    
    def filter_data_list(self, x, e, f):
        x, e, ids = self.filter_duplicated_embedded_struct(x, e)
        f = [f[i] for i in ids]
        return x, e, f
    
    def _modify_y(self, e_list, f_list):
        # scaling energy values to prevent to large numbers as labels
        if self.is_scale_energy:
            e_list, f_list = self.scaling_energy_and_forces(energy_list=e_list, atom_f_list=f_list)
        e_list, f_list = np.array(e_list), average_force(np.array(f_list))
        e_list = e_list / sum(self.num_species_atom) if self.norm_energy else e_list
        return e_list, f_list
    
    def build_y_from_calc(self, e_list, f_list):
        e_list, f_list = self._modify_y(e_list, f_list)
        assert e_list.ndim == 1 and f_list.ndim == 1
        e_list, f_list = e_list[:, np.newaxis], f_list[:, np.newaxis]
        if self.mt_lambda > 0:
            e_list = e_list + self.mt_lambda * f_list
        return e_list.flatten()

    def build_init_data_list(self):
        # model encode
        x_init = self.encode_dataset()
        
        # split an eternal test set
        if self.is_fix_test_set:
            train_ids, test_ids = train_test_split(np.arange(len(x_init)), test_size=self.reg_loader.test_rate, shuffle=True, random_state=self.seed)
            self.x_test0 = np.vstack([x_init[j] for j in test_ids])
            y_test_e0, y_test_f0 = [self.struct_e_list[j] for j in test_ids], [self.struct_f_list[j] for j in test_ids]
            self.y_test0 = self.build_y_from_calc(y_test_e0, y_test_f0)
        else:
            train_ids = np.arange(len(x_init))

        # build dataset
        if self.logger.follow_dir is None:
            # results with respect to space groups
            self.spg_results_dict = {sg: [] for sg in get_spg_list2(self.num_species_atom, self.dim, include_spg_list=None)}
            for i in range(len(self.spg_gen)):
                self.spg_results_dict[self.spg_gen[i]].append((x_init[i], self.struct_e_list[i], average_force(self.struct_f_list[i])))
            self.x_list, self.struct_e_list, self.struct_f_list = self.filter_data_list(
                [x_init[i] for i in train_ids], [self.struct_e_list[i] for i in train_ids], [self.struct_f_list[i] for i in train_ids])
        else:
            self.spg_results_dict = load_spg_dict(self.logger.follow_dir)
            x, y = load_training_set(self.logger.follow_dir)
            self.struct_e_list, self.struct_f_list = y
            self._update_sampling_threshold()
            self.x_list, self.struct_e_list, self.struct_f_list = self.filter_data_list(x, self.struct_e_list, self.struct_f_list)
        return
    
    """ for solving """
    
    def build_objective(self, w):
        if self.reg_loader.model in direct_posterior:
            self.sol.get_model_params(from_sklearn=True, learning_model=self.reg.learning_model, interaction_terms=self.reg.interaction_terms)
        else:
            self.sol.get_model_params(from_sklearn=False, learning_model=None, interaction_terms=None)
        self.sol.get_obj_params(w=w)
        bqm = self.sol.gen_solver()
        return bqm
    
    def set_opt_starting_points(self, init_vec):
        return
    
    """ relax structures & record results """

    def relax_and_evaluate_energy(self, struct, sample_num, i, trj_dir_label, fix_sym, not_save_rlx_trj=False, nsteps=_steps):
        # if atoms_overlapping(struct.pos):
        #     print('Current setting leads to atoms overlap')
        #     self.logger.record_anything9(operation='Initial structure atoms overlap', result='fail')
        #     struct = array_to_pymatgen_struct(struct.lattice, struct.pos, struct.species, struct.cart)
        #     return [struct], [None]
        if nsteps == 0:
            try:
                i_s = deepcopy(struct)
                self.e_cal.build_atoms(struct)
                i_e = self.e_cal.cal_energy()
                i_fs = self.e_cal.cal_forces()
                return i_s, i_e, i_fs
            except Exception:
                log_printer.error(f"{i}-th {trj_dir_label} estimation failed")
                return None, None, None
        def relax_and_sample(_struct, _sample_num, _fix_sym, _nsteps):
            self.e_cal.build_atoms(_struct)
            if _sample_num == 1:
                _relaxed_struct, _energy, _forces = self.e_cal.cal_relax(simple=not_save_rlx_trj, fix_sym=_fix_sym, steps=_nsteps)
                _relaxed_struct, _energy, _forces = [_relaxed_struct], [_energy], [_forces]
            elif _sample_num is not None:
                _relaxed_struct, _energy, _forces = self.e_cal.sample_trajectory(init_e_threshold=None, sample_number=_sample_num, fix_sym=_fix_sym, steps=_nsteps)
            else:
                _relaxed_struct, _energy, _forces = self.e_cal.collect_trajectory(init_e_threshold=None, fix_sym=_fix_sym, steps=_nsteps)
            return _relaxed_struct, _energy, _forces
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        self.e_cal.set_traj_label(label=f'{trj_dir_label}-{i}')
        if self.e_cal.name == EnergyEvaluatorName.vasp():
            self.e_cal.set_output(label=f'rlx-{trj_dir_label}-{i}')
        try:
            if fix_sym:
                try:
                    relaxed_struct, energy, forces = relax_and_sample(struct, sample_num, fix_sym, nsteps)
                except Exception:
                    log_printer.warning(f"{i}-th {trj_dir_label} relaxation cannot maintain symmetry")
                    relaxed_struct, energy, forces = relax_and_sample(struct, sample_num, False, nsteps)
            else:
                relaxed_struct, energy, forces = relax_and_sample(struct, sample_num, False, nsteps)
        except Exception:
            if not DEBUG:
                log_printer.error(f"{i}-th {trj_dir_label} relaxation failed")
                relaxed_struct, energy, forces = None, None, None
            else:
                raise Exception(f"{i}-th {trj_dir_label} relaxation failed")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        if nsteps == _steps:
            self.logger.record_energy_time(end_time - start_time)
        else:
            log_printer.info(f"{i}-th {trj_dir_label} relaxation -- time cost: {(end_time - start_time):.5} (s)")
        return relaxed_struct, energy, forces

    def convert_fail_results(self, pmg_struct):
        self.logger.record_new_struct_fail()
        r_struct, energy = [pmg_struct] * self.fail_num, [self.e_sampling_threshold] * self.fail_num
        forces_threshold = force_limit if self.mt_lambda > 0 else 0
        forces = [np.ones(shape=(len(pmg_struct), 3)) * forces_threshold] * self.fail_num
        return r_struct, energy, forces
    
    """ main """

    def run(self):
        self.load_dataset()
        self.load_lat_ub_and_lb()
        start_time = time.process_time()
        # load embedder (encoder & decoder)
        self.embedder, precs_settings = self.emb_loader.unwrap(len_lbs=self.len_lbs, len_ubs=self.len_ubs, ang_lbs=self.ang_lbs, ang_ubs=self.ang_ubs)
        self.logger4csp.save_modelling_settings(precs_settings)
        # instantiate regressor
        self.build_init_data_list()
        if self.is_fix_test_set:
            self.reg_loader.get_dataset(X=(np.vstack(self.x_list), self.x_test0), y=(self.build_y_from_calc(self.struct_e_list, self.struct_f_list), self.y_test0))
        else:
            self.reg_loader.get_dataset(X=np.vstack(self.x_list), y=self.build_y_from_calc(self.struct_e_list, self.struct_f_list))
        self.reg = self.reg_loader.unwrap()
        # instantiate solver
        self.sol_loader.get_num_var(self.embedder.num_var)
        if self.sol_loader.use_constraints:
            constr_mat = self.embedder.get_constraints()
            self.sol_loader.get_constraints_coefficient_matrix(coef_mat_l=constr_mat, coef_mat_q=None)
        self.sol = self.sol_loader.unwrap()
        if hasattr(self.sol, 'timeout'):
            log_printer.info(f"Current number of bits: {self.sol.num_var}; timeout / max steps for solver: {self.sol.timeout}")
        elif hasattr(self.sol, 'num_gradient_steps'):
            log_printer.info(f"Current number of vars: {self.sol.num_var}; gradient decent steps for optimization: {self.sol.num_gradient_steps}")
        self.logger.record_modelling1_time(operation='Initialization', delta_t=time.process_time() - start_time)
        log_printer.info("Start running")

        while self.logger.new_struct_suc < self.num_steps:
            # 1. fitting & generate obj based on the equation
            start_time = time.process_time()
            # regressor learns
            self.reg.learn_weights()
            log_printer.info("Finish learning")
            # solver outputs the obj based on regressor-learned weight
            w = self.reg.get_weight()
            self.reg.trainer.cleanup()

            # 2. solver builds bqm and solves
            self.set_opt_starting_points(self.spg_results_dict)
            if self.sample_time == 1:
                obj = self.build_objective(w)
                new_x, solved_value = self.sol.solve(obj)
            else:
                new_x, solved_value = [], []
                for i in range(self.sample_time):
                    obj = self.build_objective(w[i])
                    x, v = self.sol.solve(obj)
                    new_x.append(x)
                    solved_value.append(v)
            log_printer.info("Finish solving obj")

            # judge whether the problem can be solved
            if new_x is None:
                self.logger.record_new_struct_fail()
                print("The 'no feasible solution error' is raised. "
                      "The next iteration will be conducted without constraints.")
                self.logger.record_anything10(operation='No feasible solution', result='fail')
                continue
            
            # 3-5. decode the solution and estimate energies
            if self.sample_time == 1:
                energy, forces, i_struct, r_struct, decoded_used_atoms, decoded_spg, decoded_wpc, iter_dataset, new_x, solved_value = self.decode_and_estimate(start_time, new_x, solved_value)
                self.collect_results_and_build_dataset(energy, forces, i_struct, r_struct, decoded_used_atoms, decoded_spg, decoded_wpc, iter_dataset)
                torch.cuda.empty_cache()
            else:
                for x, v in zip(new_x, solved_value):
                    energy, forces, i_struct, r_struct, decoded_used_atoms, decoded_spg, decoded_wpc, iter_dataset, x, v = self.decode_and_estimate(start_time, x, v)
                    self.collect_results_and_build_dataset(energy, forces, i_struct, r_struct, decoded_used_atoms, decoded_spg, decoded_wpc, iter_dataset)
                    torch.cuda.empty_cache()
                    start_time = time.process_time()

        return
    
    def decode(self, new_x, solved_value):
        pmg_struct = self.embedder.decode(new_x)
        pmg_struct = [pmg_struct] if not isinstance(pmg_struct, list) else pmg_struct
        cs, spg, wpc_idx, wpc, used_atoms = self.embedder.cs, self.embedder.spg, self.embedder.wpc_idx, self.embedder.wpc, self.embedder.used_atoms
        return new_x, solved_value, pmg_struct, cs, spg, wpc_idx, wpc, used_atoms
    
    def decode_and_estimate(self, start_time, new_x, solved_value):
        # 3. decoder decodes
        new_x, solved_value, pmg_struct, cs, spg, wpc_idx, wpc, used_atoms = self.decode(new_x, solved_value)
        # TODO: dict of list for multi-systems
        
        if self.dim < 3:
            core_file = 'generated_core.pkl.bz2'
            if Path(f'{self.logger.log_dir}/{core_file}').exists():
                with bz2.open(f'{self.logger.log_dir}/{core_file}', 'rb') as f:
                    core_struct = pickle.load(f)
            else:
                core_struct = []
            core_struct += pmg_struct
            with bz2.open(f'{self.logger.log_dir}/{core_file}', 'wb') as f:
                pickle.dump(core_struct, f)
            pbc = dim2pbc(self.dim)
            pmg_struct = [self.add_vacuum_to_pmg_fn(s, pbc=pbc) for s in pmg_struct]

        _sp = SPG()
        if self.dim == 3:  # test symmetry
            real_spg_list = [SPG.struct_spg_estimate(pmg_s)[1] for pmg_s in pmg_struct]
            success_struct_ids = np.where(np.array(real_spg_list) == np.array(spg))[0]  # works for scaler & array spg
            log_printer.info(f"Obtain {len(success_struct_ids)} structures with desired SG "
                            f"out of {len(pmg_struct)} ones with current symprec {_sp.symprec}")

        # 3.1 enlarge lattice if needed
        if self.is_enlarge_lattice:
            pmg_struct2 = []
            len_ubs, len_units = self.len_ubs, self.embedder.lat_enc.len_units
            len_ubs = len_ubs * 1.2  # maximum allowed lattice length after enlargement
            for i, p in enumerate(pmg_struct):
                try:
                    p = self.incremental_lattice_length_enlarge(i, p, dim=self.dim, len_ubs=len_ubs, len_units=len_units)
                    pmg_struct2.append(p)
                except Exception:
                    log_printer.warning('Fail to enlarge the lattice')
            if len(pmg_struct2) != 0:
                pmg_struct = pmg_struct2
            else:
                pmg_struct = [(p_s, False) for p_s in pmg_struct]
        else:
            pmg_struct = [(p_s, False) for p_s in pmg_struct]

        # 3.2. select structure based on minimum interatomic distances
        if self.is_pre_relax:
            pmg_struct, selected_idx, iter_dataset = self.find_struct_wrt_energy_and_spg(pmg_struct, i_iter=self.logger.new_struct_suc, trial_steps=short_trj_step)
        else:
            pmg_struct, selected_idx = self.find_struct_wrt_pair_dist(pmg_struct)
            iter_dataset = None
        if spg is not None:
            decoded_used_atoms = used_atoms[selected_idx]
            decoded_cs, decoded_spg, decoded_wpc, decoded_wpc_list = cs[selected_idx], spg[selected_idx], wpc_idx[selected_idx], wpc[selected_idx]
            if np.array(new_x).ndim != 1:
                new_x = new_x[selected_idx]
                solved_value = solved_value[selected_idx]
            # else:  # wpc_idx for ising solvers
            #     decoded_spg = spg
            #     wpc_spg_list = self.embedder.wyckoffs_dict[decoded_spg]
            #     decoded_wpc = wpc + selected_idx
            #     if decoded_wpc >= len(wpc_spg_list):
            #         decoded_wpc = decoded_wpc - len(wpc_spg_list)
            #     decoded_wpc_list = wpc_spg_list[decoded_wpc]
            if iter_dataset is not None:
                i_s_list, i_e_list, i_fs_list, r_s_list, r_e_list, r_fs_list = iter_dataset
                iter_dataset = (i_s_list, i_e_list, i_fs_list, r_s_list, r_e_list, r_fs_list, used_atoms, spg, wpc_idx)
        else:
            decoded_used_atoms, decoded_cs, decoded_spg, decoded_wpc, decoded_wpc_list = used_atoms, cs, spg, wpc_idx, wpc
        
        # 4. collect init structures information
        log_printer.info(f"Finish decoding the material with spg {decoded_spg}, wpc {decoded_wpc}")
        if self.dim == 3:
            real_crys_sys = SPG.struct_lattice_estimate(pmg_struct, None)
            log_printer.info(f'Generated Crystal System: {real_crys_sys}')
            _, real_spg_n = SPG.struct_spg_estimate(pmg_struct)
            log_printer.info(f'Generated Space Group: {real_spg_n} with current symprec {_sp.symprec}')
        self.logger.record_modelling4_time(operation='New structure generating trails', delta_t=time.process_time() - start_time)
        self.logger4csp.record_init_info(i=self.logger.new_struct_suc, i_struct=pmg_struct, decoded_cs=decoded_cs, decoded_spg=decoded_spg, 
                                         decoded_wpc_idx=decoded_wpc, decoded_wpc_list=decoded_wpc_list, decoded_used_atoms=decoded_used_atoms)

        # 5. try to relax the initial structure & save it
        i_struct = deepcopy(pmg_struct)
        fix_sym = self.is_fix_sym
        r_struct, energy, forces = self.relax_and_evaluate_energy(pmg_struct, sample_num=self.iter_sample_num, i=self.logger.new_struct_gen_call, trj_dir_label='opt', fix_sym=fix_sym)
        if (energy is None) or (None in energy) or (r_struct is None) or (None in r_struct):
            r_struct, energy, forces = self.convert_fail_results(pmg_struct)
        # 5.1 assign high energies to structures whose atoms are too close but energies < 0, as a remedy of MLP
        if self.is_amend_energy:
            energy = self.filter_struct_wrt_pair_dist(pmg_struct_list=r_struct, energy_list=energy)
        # 5.1.1 assign high energies to too thick 2d layers
        if self.dim == 2 and self.is_scale_2d and hasattr(self, "z_2d_ub_relax"):
            energy = self.filter_struct_wrt_thickness_for_2d(pmg_struct_list=r_struct, energy_list=energy)

        self.logger4csp.record_success_step(self.logger.new_struct_suc, r_struct=r_struct[-1], energy=energy[-1], forces=forces[-1], solved_value=(new_x, solved_value))
        self.embedder.load_empty_results()
        
        return energy, forces, i_struct, r_struct, decoded_used_atoms, decoded_spg, decoded_wpc, iter_dataset, new_x, solved_value
    
    def collect_results_and_build_dataset(self, energy, forces, i_struct, r_struct, decoded_used_atoms, decoded_spg, decoded_wpc, iter_dataset):
        # 6. add new data point for fitting in the next cycle
        if self.use_rlx_type != rlx_type[2]:
            encode_struct = r_struct
        else:
            encode_struct = [i_struct]
        log_printer.info(f"Encode traj based on symmetry of init: spg {decoded_spg}, wpc {decoded_wpc}")
        r_new_x = []  # encode
        
        # 6.1 encode trajectory
        for i, rp_s in enumerate(encode_struct):
            if self.dim < 3:
                rp_s = reduce_vacuum_from_pmg(rp_s, pbc=dim2pbc(self.dim))
            decoded_used_atoms_traj, decoded_spg_traj, decoded_wpc_traj = self.decide_pmg_symmetry(i, rp_s, decoded_used_atoms, decoded_spg, decoded_wpc)
            new_vec = self.embedder.encode([rp_s, decoded_used_atoms_traj, decoded_spg_traj, decoded_wpc_traj])
            r_new_x.append(new_vec)
            self.spg_results_dict[decoded_spg].append((new_vec, energy[i], average_force(forces[i])))
        # 6.2 encode iterative dataset
        if iter_dataset is not None:
            i_s_list, i_e_list, i_fs_list, r_s_list, r_e_list, r_fs_list, used_atoms_list, spg_list, wpc_list = iter_dataset
            # with bz2.open(self.logger.log_dir + f'/iter_data-{self.logger.new_struct_suc}.pkl.bz2', 'wb') as ff:
            #     pickle.dump(iter_dataset, ff)
            #     log_printer.info(f"Iter dataset in {self.logger.new_struct_suc}-th iteration saved")
            for i in range(len(i_s_list)):
                if i_e_list[i] is not None:
                    new_vec = self.embedder.encode([i_s_list[i], used_atoms_list[i], spg_list[i], wpc_list[i]])
                    r_new_x.append(new_vec)
                    self.spg_results_dict[spg_list[i]].append((new_vec, i_e_list[i], average_force(i_fs_list[i])))
                    energy.append(i_e_list[i])
                    forces.append(i_fs_list[i])
            for i in range(len(r_s_list)):
                if r_e_list[i] is not None:
                    decoded_used_atoms_traj, decoded_spg_traj, decoded_wpc_traj = self.decide_pmg_symmetry(i, r_s_list[i], used_atoms_list[i], spg_list[i], wpc_list[i])
                    new_vec = self.embedder.encode([r_s_list[i], decoded_used_atoms_traj, decoded_spg_traj, decoded_wpc_traj])
                    r_new_x.append(new_vec)
                    self.spg_results_dict[decoded_spg_traj].append((new_vec, r_e_list[i], average_force(r_fs_list[i])))
                    energy.append(r_e_list[i])
                    forces.append(r_fs_list[i])
        
        # 6.3 remove duplicates & records results & transfer to regressor
        r_new_x, energy, forces = self.filter_data_list(r_new_x, energy, forces)
        self.x_list, self.struct_e_list, self.struct_f_list = self.filter_data_list(self.x_list + r_new_x, self.struct_e_list + energy, self.struct_f_list + forces)
        self._update_sampling_threshold()
        self.logger.save_spg_dict(self.spg_results_dict)
        self.logger.save_training_set(X=self.x_list, y=[self.struct_e_list, self.struct_f_list])
        self.reg.set_data(x=np.vstack(self.x_list), y=self.build_y_from_calc(self.struct_e_list, self.struct_f_list))
        log_printer.info(f"[ACCUMULATED RESULTS] Current best: {np.min(self.struct_e_list)}")
        return
    
    def incremental_lattice_length_enlarge(self, idx, pmg_struct, dim, len_ubs, len_units):
        """ All lattice lengths will be increased at the same time based on their initial ratio.
            Therefore, crystal system will not be broken. """
        elements = pmg_struct[0].elements if isinstance(pmg_struct, list) else pmg_struct.elements
        mid_lb, _ = mid_scale_for_system(element_list=elements, min_scale=0.8)
        for i in range(len(len_units)):
            if len_units[i] == 0:
                len_units[i] = 0.1

        def check_lattice(_lattice, _dim, _len_ubs):
            _a, _b, _c = _lattice.abc
            if _dim == 3:
                return (_a < _len_ubs[0]) and (_b < _len_ubs[1]) and (_c < _len_ubs[2])
            elif _dim == 2:
                return (_a < _len_ubs[0]) and (_b < _len_ubs[1])
            else:
                return _a < _len_ubs[0]

        a0, b0, c0 = pmg_struct.lattice.abc
        alpha, beta, gamma = pmg_struct.lattice.angles
        if dim == 3:
            len_units = [len_units[0], len_units[1], len_units[2]]
        elif dim == 2:
            len_units = [len_units[0], len_units[1], 0]
        else:
            len_units = [len_units[0], 0, 0]
        n_iter = 0
        if dim == 3: min_lat_len = min(a0, b0, c0)
        elif dim == 2: min_lat_len = min(a0, b0)
        else: min_lat_len = a0
        unit_a, unit_b, unit_c = a0 / min_lat_len, b0 / min_lat_len, c0 / min_lat_len
        while (np.min(min_pair_dist_mat(pmg_struct)) < mid_lb) and check_lattice(pmg_struct.lattice, dim, len_ubs):
            a, b, c = pmg_struct.lattice.abc
            a, b, c = a + unit_a * len_units[0], b + unit_b * len_units[1], c + unit_c * len_units[2]
            new_lattice = Lattice.from_parameters(a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
            pmg_struct.lattice = new_lattice
            n_iter += 1
            if n_iter > 3000:
                log_printer.warning(f'Enlargement failed after {n_iter} steps')
                break
        log_printer.info(f'{idx}-th enlarge lattice length from {a0}, {b0}, {c0} to '
                         f'{pmg_struct.lattice.a}, {pmg_struct.lattice.b}, {pmg_struct.lattice.c}')
        if round(pmg_struct.lattice.a - a0) > 0.1 or round(pmg_struct.lattice.a - a0) > 0.1 or round(pmg_struct.lattice.a - a0) > 0.1:
            return pmg_struct, True
        else:
            return pmg_struct, False

    def find_struct_wrt_pair_dist(self, pmg_struct_list):
        """ first filter out structures still having too small mid;
            then sort by mid, from large to small; 
            then sort by minimum lattice length, from small to large; 
            finally select the one in the front of the list. """
        if len(pmg_struct_list) == 1:
            return pmg_struct_list[0][0], 0
        
        num_struct_input = len(pmg_struct_list)
        init_pmg_struct_list = [p_s[0] for p_s in pmg_struct_list]
        pmg_struct_not_increase_lat = [p_s[0] for p_s in pmg_struct_list if not p_s[1]]
        if len(pmg_struct_not_increase_lat) == 0:
            log_printer.info('Lattices of all initial structures are increased due to too small mid')
            ids_input = list(range(num_struct_input))
            pmg_struct_list = [p_s[0] for p_s in pmg_struct_list]
        else:
            log_printer.info(f'{len(pmg_struct_not_increase_lat)}/{num_struct_input} structures did not change, select among them')
            ids_input = [i for i, p_s in enumerate(pmg_struct_list) if not p_s[1]]
            pmg_struct_list = pmg_struct_not_increase_lat
        
        elements = pmg_struct_list[0].elements
        filtered_out_mid, _ = mid_scale_for_system(element_list=elements)
        mid_list = np.array([np.min(min_pair_dist_mat(p_struct)) for p_struct in pmg_struct_list])
        idx_sort = np.argsort(mid_list)  # smallest mid -> largest mid
        if mid_list[idx_sort[-1]] <= filtered_out_mid:  # the largest mid is still smaller than the threshold, just return the largest
            log_printer.info('All generated structures have a too small mid')
            self.logger.record_anything6(f'Get {idx_sort[-1]}-th from {len(pmg_struct_list)} structs with dist', result=mid_list[idx_sort[-1]])
            return pmg_struct_list[idx_sort[-1]], ids_input[idx_sort[-1]]

        init_idx = 0
        while mid_list[idx_sort[init_idx]] < filtered_out_mid:
            init_idx += 1
        idx_sort = idx_sort[init_idx:]
        idx_sort = idx_sort[::-1]  # largest mid -> smallest mid
        
        # picked_idx = idx_sort[pick(len(idx_sort), bias=10)]  # large bias tends to select from the top
        picked_idx = idx_sort[np.random.choice(range(int(len(idx_sort) * 0.01) + 1))]
        picked_struct = pmg_struct_list[picked_idx]
        init_picked_idx = ids_input[picked_idx]
        self.logger.record_anything6(f'Get {init_picked_idx}-th from {len(pmg_struct_list)} structs with dist', result=mid_list[picked_idx])
        log_printer.info(f'Largest dist: {mid_list[idx_sort[0]]}')
        assert np.min(min_pair_dist_mat(init_pmg_struct_list[init_picked_idx])) == np.min(min_pair_dist_mat(picked_struct)) == mid_list[picked_idx]
        return picked_struct, init_picked_idx
        
        # min_lat_len_list = [np.min(pmg_struct[idx].lattice.abc) for idx in idx_sort]
        # idx_min_lat_len_list = np.argsort(min_lat_len_list, stable=True)  # smallest mll -> largest mll & largest mid -> smallest mid
        # idx_min_lat_len = idx_min_lat_len_list[pick(len(idx_min_lat_len_list), 3)]
        # picked_idx = idx_sort[idx_min_lat_len]
        # self.logger.record_anything6(f'Get {picked_idx}-th from {len(pmg_struct)} structs with dist', result=mid_list[picked_idx])
        # log_printer.info(f'Largest dist: {mid_list[idx_sort[0]]}')
        # log_printer.info(f'Selected minimum lattice length: {min_lat_len_list[idx_min_lat_len]}')
        # return pmg_struct[picked_idx], picked_idx
    
    def find_struct_wrt_energy_and_spg(self, pmg_struct_list, i_iter, trial_steps):
        if len(pmg_struct_list) == 1:
            return pmg_struct_list[0][0], 0, None

        pmg_struct_list = [p_s[0] for p_s in pmg_struct_list]  # accept all enlarged structures
        i_s_list, i_e_list, i_fs_list, r_s_list, r_e_list, r_fs_list = [], [], [], [], [], []
        for i, s in enumerate(pmg_struct_list):
            s0 = deepcopy(s)
            if np.min(min_pair_dist_mat(s)) <= _hard_min_mid:  # first check mid
                log_printer.error(f"{i}-th init structure has unphysically close atom pairs")
                i_s, i_e, i_fs = s0, 999, np.ones(shape=(self.atom_num, 3)) * 999
                r_s, r_e, r_fs = [s0], [999], [np.ones(shape=(self.atom_num, 3)) * 999]
            else:
                i_s, i_e, i_fs = self.relax_and_evaluate_energy(s, sample_num=None, i=None, trj_dir_label=None, fix_sym=None, nsteps=0)
                if (i_e is None) or (i_fs is None):
                    i_s, i_e, i_fs = s0, 999, np.ones(shape=(self.atom_num, 3)) * 999
                r_s, r_e, r_fs = self.relax_and_evaluate_energy(s, sample_num=1, i=i, trj_dir_label=f's-{i_iter}', not_save_rlx_trj=True, fix_sym=True, nsteps=trial_steps)
                if (r_e is None) or (isinstance(r_e, list) and (None in r_e)) or (r_s is None) or (isinstance(r_s, list) and (None in r_s)):
                    r_s, r_e, r_fs = [s0], [999], [np.ones(shape=(self.atom_num, 3)) * 999]
            i_s_list.append(i_s)
            i_e_list.append(i_e)
            i_fs_list.append(i_fs)
            r_s_list += r_s
            r_e_list += r_e
            r_fs_list += r_fs
        selected_idx = np.argsort(r_e_list)[:int(len(pmg_struct_list) * 0.01) + 1]
        if (r_e_list[selected_idx[0]] - r_e_list[selected_idx[-1]]) / r_e_list[selected_idx[0]] < 0.05:
            selected_idx = np.random.choice(selected_idx)
        else:
            selected_idx = selected_idx[0]
        selected_i_s = i_s_list[selected_idx]
        return selected_i_s, selected_idx, (i_s_list, i_e_list, i_fs_list, r_s_list, r_e_list, r_fs_list)
    
    def filter_struct_wrt_pair_dist(self, pmg_struct_list, energy_list=None):
        """ reassign a high energy to underestimated structures, which have a very small mid. """
        pass_list = []
        energy_remedy = 0
        for i, p_struct in enumerate(pmg_struct_list):
            if not DistFilter.test(p_struct, test_lattice=False):
                pass_list.append(False)
                if energy_list is not None:
                    if energy_list[i] < 0:
                        energy_list[i] = self.e_sampling_threshold
                        energy_remedy += 1
            else:
                pass_list.append(True)
        log_printer.info(f'Current lower and upper bounds for dist: {mid_scale_for_system(element_list=self.elements)}')
        self.logger.record_anything7(f'Finish filtering {len(pmg_struct_list)} structs with passed', result=sum(pass_list))
        if energy_remedy > 0:
            log_printer.info(f'Attention! Energy reassignment activated on {energy_remedy} structures with {self.e_sampling_threshold}')
        if energy_list is not None:
            return energy_list
        else:
            return pass_list
    
    def scaling_energy_and_forces(self, energy_list, atom_f_list):
        energy_list, atom_f_list = np.array(energy_list), np.array(atom_f_list)
        energy_list = np.where(energy_list > self.e_sampling_threshold, self.e_sampling_threshold, energy_list)
        atom_f_list = np.where(atom_f_list > force_limit, force_limit, atom_f_list)
        atom_f_list = np.where(atom_f_list < -force_limit, -force_limit, atom_f_list)
        return list(energy_list), list(atom_f_list)
    
    def filter_struct_wrt_thickness_for_2d(self, pmg_struct_list, energy_list=None):
        """ reassign a high energy to too thick layers for controlling thickness in 2d. """
        if not hasattr(self, "z_2d_ub_relax"):
            log_printer.warning("Filtering on thickness is not performed since no thickness ub provided.")
            if energy_list is not None:
                return energy_list
            else:
                return [True for _ in pmg_struct_list]
        pass_list = []
        energy_remedy = 0
        for i, p_struct in enumerate(pmg_struct_list):
            # not considering pbc since vacuum layers are presumably added properly
            thickness = (max(p_struct.frac_coords[:, 2]) - min(p_struct.frac_coords[:, 2])) * p_struct.lattice.c
            if thickness > self.z_2d_ub_relax + _vacuum_thickness:
                pass_list.append(False)
                if energy_list is not None:
                    if energy_list[i] < 0:
                        energy_list[i] = self.e_sampling_threshold
                        energy_remedy += 1
            else:
                pass_list.append(True)
        log_printer.info(f'Current upper bound for thickness: {self.z_2d_ub_relax}')
        self.logger.record_anything7(f'Finish filtering {len(pmg_struct_list)} structs with passed', result=sum(pass_list))
        if energy_remedy > 0:
            log_printer.info(f'Energy reassignment activated on {energy_remedy} structures with {self.e_sampling_threshold} due to too thick layers')
        if energy_list is not None:
            return energy_list
        else:
            return pass_list

    def decide_pmg_symmetry(self, i, pmg_struct, input_used_atoms, input_spg, input_wpc):
        return input_used_atoms, input_spg, input_wpc
    
    def __call__(self):
        self.run()
        return self.logger4csp.finish_logging()


class RandomGenerate4MS(CSP):
    def __init__(self, system_name, dim, is_scale_2d, xy_factor, z_factor, e_cal, num_steps, rs_type, seed,
                 len_lbs, len_ubs, ang_lbs, ang_ubs, include_spg_list, is_amend_energy, is_filter_dist_for_init, 
                 is_enlarge_lattice, is_fix_sym, custom_name):
        super().__init__(model_name=OptimizerName.random_search()[0],
                         real_model_name=OptimizerName.random_search()[1], is_filter_dist_for_init=is_filter_dist_for_init,
                         system_name=system_name, dim=dim, is_scale_2d=is_scale_2d, xy_factor=xy_factor, z_factor=z_factor, 
                         init_set_type=rs_type, len_lbs=len_lbs, len_ubs=len_ubs, ang_lbs=ang_lbs, ang_ubs=ang_ubs, emb_loader=None,
                         num_steps=num_steps, init_pos_num=1, init_sample_rate=1, iter_sample_num=1, init_rlx_steps=None,
                         seed=seed, e_cal=e_cal, is_enlarge_lattice=is_enlarge_lattice, mt_lambda=0, is_pre_relax=False,
                         sol_loader=None, reg_loader=None, is_amend_energy=is_amend_energy, include_spg_list=include_spg_list,
                         follow_dir=None, custom_name=custom_name, fail_num=1, is_scale_energy=False, is_fix_sym=is_fix_sym)
        return
    
    def _load_output_name(self):
        self.output_name = f'{self.name}_{self.init_set_type}{self.init_pos_num}'
        self.output_name += f'_RG_{self.e_cal.name}'
        self.real_output_name = f'{self.real_name}_{self.init_set_type}{self.init_pos_num}'
        self.real_output_name += f'_RandGen_{self.e_cal.real_name}'
        return

    def run(self):
        # generate new structures
        start_time = time.process_time()
        new_struct = self._generate(id_offset=0, nstruct=self.num_steps, filter_dist=self.is_filter_dist_for_init, seed=self.seed + 1234)
        self.logger4csp.pmg_struct_list += new_struct[0]
        self.logger.record_modelling1_time(operation='Structures generation', delta_t=time.process_time() - start_time)
        log_printer.info('Finish generating new structures')

        # read all structure and compute energies
        log_printer.info("Start running")
        for i in range(self.num_steps):
            self.logger4csp.save_init_struct(self.logger4csp.pmg_struct_list[i], idx=i, idx_off=0)
            r_struct, energy, forces = self.relax_and_evaluate_energy(self.logger4csp.pmg_struct_list[i], sample_num=1, i=i, trj_dir_label='opt', fix_sym=self.is_fix_sym)
            if (energy is None) or (None in energy) or (r_struct is None) or (None in r_struct):
                r_struct, energy, forces = self.convert_fail_results(self.logger4csp.pmg_struct_list[i])
            self.logger4csp.record_success_step(i, r_struct[-1], energy[-1], forces[-1], None)
        self.logger4csp.finish_logging()
        return


class DirectCSP(CSP):
    def __init__(self, dim, model_name, real_model_name, system_name, len_lbs, len_ubs, ang_lbs, ang_ubs,
                 is_filter_dist_for_init, emb_loader, e_cal, sol_loader, is_pre_relax, is_scale_2d, xy_factor, z_factor,
                 is_amend_energy, is_scale_energy, is_fix_sym, is_enlarge_lattice, mt_lambda, num_steps, init_set_type, include_spg_list,
                 init_rlx_steps, fail_num, init_pos_num, init_sample_rate, iter_sample_num, seed, custom_name):
        assert sol_loader.name in direct_list, "This workflow is for direct optimization on crystal parameters. Did you miss the regressor?"
        assert emb_loader.emb_type == 'sitecont+oh+used_atoms', "Direct optimization requires the 'sitecont+oh+used_atoms' representation"
        super().__init__(dim=dim, model_name=model_name, real_model_name=real_model_name, system_name=system_name, 
                         len_lbs=len_lbs, len_ubs=len_ubs, ang_lbs=ang_lbs, ang_ubs=ang_ubs, 
                         is_filter_dist_for_init=is_filter_dist_for_init, include_spg_list=include_spg_list,
                         emb_loader=emb_loader, e_cal=e_cal, sol_loader=sol_loader, reg_loader=None, is_pre_relax=is_pre_relax,
                         is_scale_2d=is_scale_2d, xy_factor=xy_factor, z_factor=z_factor,
                         is_amend_energy=is_amend_energy, is_scale_energy=is_scale_energy, is_enlarge_lattice=is_enlarge_lattice, 
                         is_fix_sym=is_fix_sym, mt_lambda=mt_lambda, num_steps=num_steps, init_set_type=init_set_type, fail_num=fail_num, 
                         init_pos_num=init_pos_num, init_sample_rate=init_sample_rate, iter_sample_num=iter_sample_num, init_rlx_steps=init_rlx_steps,
                         seed=seed, follow_dir=None, custom_name=custom_name)
        return
    
    def _load_output_name(self):
        self.output_name = f'{self.name}_{self.init_set_type}{self.init_pos_num}'
        self.output_name += f'_{OptimizerName.crystal_params_vector()[0]}_{self.sol_loader.name}_{self.e_cal.name}'
        self.real_output_name = f'{self.real_name}_{self.init_set_type}{self.init_pos_num}'
        self.real_output_name += f'_{OptimizerName.crystal_params_vector()[1]}_{self.sol_loader.real_name}_{self.e_cal.real_name}'
        return
    
    def obj(self, x):
        """ following Nat. Commun., 2022, 13, 1492. """
        start_time = time.process_time()
        lat_vec = [np.array([x['l0'], x['l1'], x['l2'], x['a0'], x['a1'], x['a2']])]
        spg, wpc = x['spg'], x['wpc']
        wpc = wpc_idx_to_specific_wpc_of_sg(wpc_list_len=len(self.wyckoffs_dict[spg]), wp=wpc, wyckoffs_max=self.wyckoffs_max)
        used_atoms = []
        for i, num_s in enumerate(self.num_species_atom):
            used_atoms_ele = []
            for j in range(num_s):
                used_atoms_ele.append([x[f'e{i}-{j}x'], x[f'e{i}-{j}y'], x[f'e{i}-{j}z']])
            used_atoms.append(used_atoms_ele)
            
        sym_vec = self.embedder.sym_emb.encode([spg, wpc])
        wpc_gen = get_wp_list2(wyckoffs_dict=self.wyckoffs_dict, spg=spg, wpc=wpc, dim=self.dim, to_str=False)
        pos_vec = self.embedder.pos_enc.encode(used_atoms, wpc_gen)
        struct_vec = np.concatenate(lat_vec + sym_vec + pos_vec)
        
        energy, forces, i_struct, r_struct, decoded_used_atoms, decoded_spg, decoded_wpc, iter_dataset = self.decode_and_estimate(start_time, struct_vec, None)
        
        return energy[-1]
    
    def run(self):
        start_time = time.process_time()
        # load embedder (encoder & decoder)
        self.embedder, precs_settings = self.emb_loader.unwrap(species_list=self.species, len_lbs=self.len_lbs, len_ubs=self.len_ubs, 
                                                               ang_lbs=self.ang_lbs, ang_ubs=self.ang_ubs)
        self.logger4csp.save_modelling_settings(precs_settings)
        # instantiate solver
        self.sol_loader.get_num_var(self.embedder.num_var)
        self.sol_loader.get_timeout_or_read_limit(self.num_steps)
        self.sol = self.sol_loader.unwrap()
        log_printer.info(f"Current number of vars: {self.sol.num_var}; try {self.sol.timeout} iterations directly")
        self.logger.record_modelling1_time(operation='Initialization', delta_t=time.process_time() - start_time)
        log_printer.info("Start running")
        
        self.wyckoffs_dict = self.embedder.wyckoffs_dict
        self.wyckoffs_max = max(len(w) for w in self.wyckoffs_dict.values())
        self.sol.get_space(len_lbs=self.len_lbs, len_ubs=self.len_ubs, ang_lbs=self.ang_lbs, ang_ubs=self.ang_ubs, 
                           compat_spg=get_spg_list(self.wyckoffs_dict), max_wpc_idx=self.wyckoffs_max, num_species_atom=self.num_species_atom)

        self.sol.solve(self.obj)
        
        return
