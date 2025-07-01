import numpy as np
from pathlib import Path
import pickle
import time
import torch

from src.optimizer.rand_gen import random_generate, LatticeLengthController, \
    default_ang_lb, default_ang_ub
from src.log_analysis import Logger, Logger4CSP, load_training_set, print_or_record
from src.registry import Component, OptimizerName, SolverName, EnergyEvaluatorName
from src.struct_utils import StructReader, get_curr_struct_num, get_species_list, \
    array_to_pmg_for_training_and_recording, pymatgen_struct_to_array, \
    atoms_overlapping, DistFilter, SPG, unique_elements, min_pair_dist_mat
from src.spg.spg_utils import sg_list, shift_spg, Random2Wyckoff
from src.spg.get_wyckoff_position import get_all_wyckoff_combination


class CSP(Component):
    def __init__(self, model_name, real_model_name, system_name, e_cal,
                 sol_name, real_solver_name, fail_num,
                 num_steps, init_pos_num, init_sample_rate, iter_sample_num,
                 struct_file_path, seed, filter_gen_struct, follow_dir):

        name = f'{model_name}_filtering' if filter_gen_struct else model_name
        real_name = f'{real_model_name}_filtering' if filter_gen_struct else real_model_name
        super().__init__(name=name, real_name=real_name)

        # basic components
        self.system_name = system_name
        self.species = get_species_list(self.system_name)
        self.elements, self.num_species_atom = unique_elements(species_list=self.species)
        self.atom_num = int(np.sum(self.num_species_atom))
        self.sol_name = sol_name
        self.real_solver_name = real_solver_name
        self.e_cal = e_cal
        self.seed = seed

        # basic practical settings
        self.filter_struct = filter_gen_struct
        if self.filter_struct:
            self._load_filter()
        self.struct_file_path = struct_file_path
        self.num_steps = num_steps
        self.init_pos_num = init_pos_num
        self.init_sample_rate = init_sample_rate
        self.iter_sample_num = iter_sample_num

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
                             seed=self.seed, follow_dir=follow_dir)
        self.logger.record_dataset_settings(self.init_pos_num, self.iter_sample_num, self.fail_num)
        self.logger4csp = Logger4CSP(logger_obj=self.logger)

        # save dir for VASP outputs
        if self.e_cal.name == EnergyEvaluatorName.vasp():
            self.e_cal.set_logger(self.logger)

        self._read_input_structure2()
        return

    def _load_output_name(self):
        self.output_name = f'{self.name}_{self.sol_name}_{self.e_cal.name}'
        self.real_output_name = f'{self.real_name}_{self.real_solver_name}_{self.e_cal.real_name}'
        return

    def _generate(self, id_offset, dire, nstruct, filter_dist, seed=None):
        pmg_struct_list, used_atoms, spg_gen, wp_gen = \
            random_generate(species_list=self.species, nstruc=nstruct,
                            id_offset=id_offset, init_pos_path=dire, seed=seed,
                            filter_dist=filter_dist, logger=self.logger)
        return pmg_struct_list, used_atoms, spg_gen, wp_gen

    def _update_sampling_threshold(self):
        if len(self.struct_e_list) == 0:
            self.e_sampling_threshold = self.e_sampling_init_threshold
        else:
            self.e_sampling_threshold = np.abs(np.min(self.struct_e_list))
        return

    def _read_input_structure(self):
        # generate initial structures if not exist
        # if not Path(self.struct_file_path).exists():
        #     self._generate(id_offset=0, dire=self.struct_file_path, nstruct=self.init_pos_num,
        #                    seed=self.seed)
        init_gen_struct_list, self.used_atoms, self.spg_gen, self.wp_gen = \
            self._generate(id_offset=0, dire=None, nstruct=self.init_pos_num,
                           filter_dist=self.filter_struct,
                           seed=self.seed + 1234)

        # read initial structures
        # struct_reader = StructReader(self.struct_file_path, cart=False)
        # self.struct_list = struct_reader.to_struct()

        # init_gen_struct_list = [pymatgen_struct_to_array(struct) for struct in init_gen_struct_list]
        # self.init_struct_num = get_curr_struct_num(self.struct_file_path, struct_size=self.atom_num)
        self.init_struct_num = len(init_gen_struct_list)
        # self.pmg_struct_list += struct_reader.to_pymatgen()

        # compute energies of initial structures
        for i, struct in enumerate(init_gen_struct_list):
            self.logger4csp.save_init_struct(struct, idx=i, idx_off=0)
            r_struct, energy = energy_evaluation_and_get_samples(pmg_struct=struct, csp=self)
            # r_struct, energy = self.relax_and_evaluate_energy(struct,
            #                                                   self.iter_sample_num *
            #                                                   self.init_sample_rate)
            self.success_step(i, r_struct, energy, None)
        self.used_atoms = [a for a in self.used_atoms for _ in range(self.iter_sample_num *
                                                                     self.init_sample_rate)]
        self.wp_gen = [wp for wp in self.wp_gen for _ in range(self.iter_sample_num *
                                                               self.init_sample_rate)]
        return

    def _read_input_structure2(self, load_cache=True):
        # training_set_file = f'dataset/{self.system_name}_training{self.seed}.pkl'
        training_set_file = self.struct_file_path
        if load_cache:
            if Path(training_set_file).exists():
                with open(training_set_file, 'rb') as f:
                    training_data = pickle.load(f)
                self.struct_list, self.struct_e_list, self.used_atoms, self.spg_gen, self.wp_gen = \
                    training_data['struct_list'], training_data['struct_e_list'], \
                        training_data['used_atoms'], training_data['spg_gen'], training_data['wp_gen']
                print(f'Training set size: {len(self.struct_list)}')
                self._update_sampling_threshold()
                return

        # nstruct = self.init_pos_num * self.iter_sample_num * self.init_sample_rate
        nstruct = self.init_pos_num
        init_gen_struct_list, used_atoms, spg_gen, wp_gen = \
            self._generate(id_offset=0, dire=None, nstruct=nstruct, filter_dist=True,
                           seed=self.seed)
        init_gen_struct_list = [pymatgen_struct_to_array(struct) for struct in init_gen_struct_list]
        self.init_struct_num = len(init_gen_struct_list)

        # compute energies of initial structures
        self.used_atoms, self.spg_gen, self.wp_gen = [], [], []
        for i, struct in enumerate(init_gen_struct_list):
            self.e_cal.build_atoms(struct)
            if self.e_cal.name == EnergyEvaluatorName.vasp():
                self.e_cal.set_output(label=f'est-{i}')
            try:
                energy = self.e_cal.cal_energy()
                print_or_record(self.logger, f'Energy of {i}-th generated structure: {energy}, '
                                f'current threshold: {self.e_sampling_threshold}')
            except Exception:
                print_or_record(self.logger, f'Energy estimation on {i}-th structure failed. '
                                f'Go to the next one.')
                continue

            if energy <= self.e_sampling_threshold:
                struct = array_to_pmg_for_training_and_recording(struct)
                self.used_atoms.append(used_atoms[i])
                self.spg_gen.append(spg_gen[i])
                self.wp_gen.append(wp_gen[i])
                self.struct_list.append(struct)
                self.struct_e_list.append(energy)

            self._update_sampling_threshold()

            with open(training_set_file, 'wb') as f:
                data = {'struct_list': self.struct_list, 'struct_e_list': self.struct_e_list,
                        'used_atoms': self.used_atoms, 'spg_gen': self.spg_gen, 'wp_gen': self.wp_gen}
                pickle.dump(data, f)
        print(f'Training set size: {len(self.struct_list)}')
        return

    def init_model(self):
        raise NotImplementedError

    def build_simulator(self, *args):
        raise NotImplementedError

    def model_learn(self):
        raise NotImplementedError

    def model_decode(self, new_x):
        raise NotImplementedError

    def model_relaxed_encode(self, r_pmg_struct, new_x):
        raise NotImplementedError

    def relax_and_evaluate_energy(self, struct, sample_num, i):
        # if atoms_overlapping(struct.pos):
        #     print('Current setting leads to atoms overlap')
        #     self.logger.record_anything9(operation='Initial structure atoms overlap', result='fail')
        #     struct = array_to_pymatgen_struct(struct.lattice, struct.pos, struct.species, struct.cart)
        #     return [struct], [None]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        self.e_cal.build_atoms(struct)
        if self.e_cal.name == EnergyEvaluatorName.vasp():
            self.e_cal.set_output(label=f'rlx-{i}')
        if sample_num == 1:
            relaxed_struct, energy = self.e_cal.cal_relax()
            relaxed_struct, energy = [relaxed_struct], [energy]
        else:
            relaxed_struct, energy = self.e_cal.sample_trajectory(
                init_e_threshold=self.e_sampling_threshold, sample_number=sample_num)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        self.logger.record_energy_time(end_time - start_time)
        return relaxed_struct, energy

    def _load_filter(self):
        self.dist_filter = DistFilter(self.num_species_atom, dist=0.5)
        return

    def success_step(self, i, r_struct, energy, solved_value):
        # recording
        r_struct_final, energy_final = r_struct[-1], energy[-1]
        self.logger4csp.record_success_step(i, r_struct_final, energy_final, solved_value)

        # preparation for training
        for e in energy:
        # for s, e in zip(r_struct, energy):
            # s = array_to_pmg_for_training_and_recording(s)
            # self.struct_list.append(s)
            self.struct_e_list.append(e)
        self._update_sampling_threshold()
        return


def find_struct_wrt_pair_dist(pmg_struct, csp):
    if not isinstance(pmg_struct, list):
        return pmg_struct
    else:
        best_idx, max_min_dist = 0, 0
        for i, p_struct in enumerate(pmg_struct):
            min_dist = np.min(min_pair_dist_mat(p_struct))
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = i
        csp.logger.record_anything6(f'Get {best_idx}-th from {len(pmg_struct)} structs with dist', result=max_min_dist)
        return pmg_struct[best_idx]


def energy_evaluation_and_get_samples(pmg_struct, csp, idx=None):
    struct = pymatgen_struct_to_array(pmg_struct)
    try:
        r_struct, energy = csp.relax_and_evaluate_energy(
            struct, sample_num=csp.iter_sample_num, i=idx)
        if (energy is None) or (None in energy) or (r_struct is None) or (None in r_struct):
            csp.logger.record_new_struct_fail()
            r_struct, energy = [pmg_struct] * csp.fail_num, [csp.e_sampling_threshold] * csp.fail_num
    except Exception:
        print_or_record(csp.logger, "Raised Exception leads to relaxation failure!")
        csp.logger.record_new_struct_fail()
        r_struct, energy = [pmg_struct] * csp.fail_num, [csp.e_sampling_threshold] * csp.fail_num
    return r_struct, energy


def filter_struct_wrt_pair_dist(csp, pmg_struct_list, energy_list=None):
    """ Originally designed for output of energy evaluation & relaxation function,
        in which energies have been estimated. """
    pass_list = []
    for i, p_struct in enumerate(pmg_struct_list):
        if not csp.dist_filter(p_struct, test_lattice=False):
            pass_list.append(False)
            if energy_list is not None:
                if energy_list[i] < 0:
                    energy_list[i] = csp.e_sampling_threshold
        else:
            pass_list.append(True)
    csp.logger.record_anything7(f'Finish filtering {len(pmg_struct_list)} structs with passed', result=sum(pass_list))
    if energy_list is not None:
        return pmg_struct_list, energy_list
    else:
        return pmg_struct_list, pass_list


# def energy_evaluation_and_get_samples(pmg_struct, csp):
#     struct = pymatgen_struct_to_array(pmg_struct)
#     r_struct, energy = csp.relax_and_evaluate_energy(
#         struct, sample_num=csp.iter_sample_num)
#     if (energy is None) or (None in energy):
#         csp.logger.record_new_struct_fail()
#         r_struct, energy = [None], [999]
#     return r_struct, energy


class OptBase:
    def __init__(self, csp, sol):
        self.csp = csp
        self.sol = sol
        self.sol.get_logger(self.csp.logger)
        return

    def _build_solver(self):
        raise NotImplementedError

    def _init_model(self):
        start_time = time.process_time()
        print("Start building model")
        self.csp.init_model()
        self._build_solver()
        if hasattr(self.sol, 'timeout'):
            self.csp.logger.record_line(f"Current number of bits: {self.sol.num_var}; "
                                        f"timeout for BQM solver: {self.sol.timeout}")
        elif hasattr(self.sol, 'max_step'):
            self.csp.logger.record_line(f"Current number of vars: {self.sol.num_var}; "
                                        f"max trial steps for classical solver: {self.sol.max_step}")
        self.csp.build_simulator(self.csp.encode_x())
        self.csp.logger.record_modelling1_time(operation='Initialization',
                                               delta_t=time.process_time() - start_time)
        return

    def _model_learn(self):
        self.csp.model_learn()
        return

    def _model_decode(self, new_x):
        return self.csp.model_decode(new_x)

    def _model_relaxed_encode(self, r_pmg_struct, new_x):
        return self.csp.model_relaxed_encode(r_pmg_struct, new_x)

    def _model_add_points(self, r_new_x, energy):
        self.csp.model.add_data(np.vstack(r_new_x), energy)
        return

    def learn_and_solve(self):
        raise NotImplementedError

    def _unsolvable_reload(self):
        return

    def _prepare_for_new_step(self, *args):
        return

    def run(self):
        self._init_model()
        print("Start running")

        while self.csp.logger.new_struct_suc < self.csp.num_steps:
            # 1. fitting & generate solver based on the equation
            # 2. solve it
            new_x, solved_value, start_time = self.learn_and_solve()

            # judge whether the problem can be solved
            if new_x is None:
                self.csp.logger.record_new_struct_fail()
                self._unsolvable_reload()
                print("The 'no feasible solution error' is raised. "
                      "The next iteration will be conducted without constraints.")
                self.csp.logger.record_anything10(operation='No feasible solution', result='fail')
                continue

            pmg_struct, new_x = self._model_decode(new_x)
            self.csp.logger.record_line("Finish decoding the material")
            # if pmg_struct is None:  # no wp available for current spg in CONBQA
            #     self.csp.logger.record_new_struct_fail()
            #     continue
            self.csp.logger.record_modelling4_time(operation='New structure generating trails',
                                                   delta_t=time.process_time() - start_time)

            # 2.1. filter the generated structure
            if self.csp.filter_struct:
                pmg_struct = find_struct_wrt_pair_dist(pmg_struct, csp=self.csp)

            # 3. save the generated initial structure information
            self.csp.logger4csp.save_init_struct(pmg_struct, idx=self.csp.logger.new_struct_suc, idx_off=0)

            # 4. try to relax the initial structure & save it
            r_struct, energy = energy_evaluation_and_get_samples(pmg_struct, csp=self.csp,
                                                                 idx=self.csp.logger.new_struct_gen_call)

            # 4.1 assign high energies to structures whose atoms are too close but energies < 0, as a remedy of MLP
            if self.csp.filter_struct:
                r_struct, energy = filter_struct_wrt_pair_dist(csp=self.csp, pmg_struct_list=r_struct,
                                                               energy_list=energy)

            self.csp.success_step(self.csp.logger.new_struct_suc, r_struct, energy, (new_x, solved_value))

            # 5. add new data point for fitting
            r_new_x = self._model_relaxed_encode(r_struct, new_x)
            self._model_add_points(r_new_x, energy)

            # 0. if there is anything left to do
            print('Start preparing for the next step')
            self._prepare_for_new_step(r_new_x, energy)

        return

    def __call__(self):
        self.run()
        return self.csp.logger4csp.finish_logging()


class BQM4CSP(OptBase):
    def __init__(self, csp, sol):
        super().__init__(csp=csp, sol=sol)
        self.rebuild_sol_flag = False
        return

    def sol_unwrap(self):
        self.sol_real = self.sol.unwrap()  # instantiate the Solver class
        return

    def _build_bqm_basic(self, model, num_var, learning_model, interaction_terms):
        if self.sol_real.name == SolverName.simulated_annealing()[0]:
            q = model.convert_maximization_of_learned_function_into_qubo()
            bqm = self.sol_real.init_solver(q)
        else:
            w = model.get_weight()
            bqm = self.sol_real.gen_solver(w, from_sklearn=True, learning_model=learning_model,
                                           interaction_terms=interaction_terms)
            # bqm = self.sol_real.init_solver(w, num_var=num_var, from_sklearn=True, learning_model=learning_model)
        return bqm

    def _build_bqm(self):
        raise NotImplementedError

    def _solve_bqm(self, bqm):
        sol = self.sol if not self.sol.proxy else self.sol_real  # self.sol_real = self.sol.unwrap()
        new_x, solved_value = sol.solve(bqm, seed=None)
        return new_x, solved_value

    def learn_and_solve(self):
        start_time = time.process_time()
        self._model_learn()
        self.csp.logger.record_line("Finish learning")
        bqm = self._build_bqm()
        self.csp.logger.record_line("Finish building bqm")
        self.csp.logger.record_modelling5_time(operation='Learning & encoding',
                                               delta_t=time.process_time() - start_time)
        start_time = time.process_time()
        new_x, solved_value = self._solve_bqm(bqm)
        self.csp.logger.record_line("Finish solving bqm")
        return new_x, solved_value, start_time

    def _unsolvable_reload(self):
        self.sol.remove_constraints()
        self.sol.get_num_var(self.csp.ohp.num_var)
        self.sol_real = self.sol.unwrap()
        self.rebuild_sol_flag = True
        return

    def _reload_constrained_solver(self):
        if self.rebuild_sol_flag:
            self._build_solver()
            self.rebuild_sol_flag = False
            print("Recover constraints")
        return

    def _prepare_for_new_step(self, *args):
        self._reload_constrained_solver()
        return


class Classical4CSP(OptBase):
    def __init__(self, csp, sol):
        super().__init__(csp=csp, sol=sol)
        return

    def _build_objective_function(self):
        raise NotImplementedError

    def learn_and_solve(self):
        start_time = time.process_time()
        self._model_learn()
        self.csp.logger.record_modelling5_time(operation='Learning & encoding',
                                               delta_t=time.process_time() - start_time)

        start_time = time.process_time()
        self.sol.get_q(q=self._build_objective_function())
        new_x, solved_value = self.sol.solve()
        return new_x, solved_value, start_time


class RandomGenerate4MS(CSP):
    def __init__(self, system_name, e_cal, num_steps, struct_file_path, init_pos_num, seed,
                 filter_gen_struct, follow_dir):
        super().__init__(model_name=OptimizerName.random_search()[0],
                         real_model_name=OptimizerName.random_search()[1],
                         system_name=system_name, num_steps=num_steps,
                         init_pos_num=init_pos_num, init_sample_rate=1, iter_sample_num=1,
                         struct_file_path=struct_file_path, seed=seed, e_cal=e_cal,
                         sol_name=None, real_solver_name=None, filter_gen_struct=filter_gen_struct,
                         follow_dir=follow_dir, fail_num=1)
        return

    def _read_input_structure2(self, load_cache=None):
        self._read_input_structure()
        return

    def run(self):
        # generate new structures
        if self.init_struct_num < self.num_steps:
            start_time = time.process_time()
            new_struct = self._generate(id_offset=self.init_struct_num,
                                        dire=None,
                                        nstruct=self.num_steps - self.init_struct_num,
                                        filter_dist=self.filter_struct,
                                        seed=self.seed)
            self.logger4csp.pmg_struct_list += new_struct[0]
            self.logger.record_modelling1_time(operation='Additional structures generation',
                                               delta_t=time.process_time() - start_time)
        print('Finish generating new structures')

        # read all structure and compute energies
        print("Start running")
        for i in range(self.init_struct_num, self.num_steps):
            self.logger4csp.save_init_struct(self.logger4csp.pmg_struct_list[i], idx=i, idx_off=0)
            r_struct, energy = energy_evaluation_and_get_samples(self.logger4csp.pmg_struct_list[i], csp=self, idx=i)
            self.success_step(i, r_struct, energy, None)
        return

    def __call__(self):
        self.run()
        return self.logger4csp.finish_logging()


class ContinuousModel:
    def __init__(self, atom_num, species, num_species_atom, init_wp_list, init_spg_list,
                 struct_list, follow_dir, logger=None):
        self.lat_dim = 3
        self.ang_dim = 3
        self.pos_dim = atom_num * 3
        self.spg_l, self.spg_u = 2, 230
        self.species = species
        self.logger = logger

        # self.lat_l = 5.0
        # self.ang_l, self.ang_u = 30, 150
        llc = LatticeLengthController(num_atom=atom_num, logger=self.logger)
        self.lat_l = llc.min()
        self.ang_l, self.ang_u = default_ang_lb, default_ang_ub

        self.wyckoffs_dict, self.wyckoffs_max = get_all_wyckoff_combination(
            sg_list=sg_list, atom_num=num_species_atom)
        self.num_var = self.lat_dim + self.ang_dim + self.pos_dim + 2
        self.lat_scale = 0.1

        if follow_dir is None:
            self.point_list, self.lat_max = self.generate_struct_point(
                init_wp_list=init_wp_list, init_spg_list=init_spg_list, struct_list=struct_list)
        else:
            self.point_list, self.point_list_y = load_training_set(follow_dir)
            self.lat_max = np.max(self.point_list[:self.lat_dim]) * (1 + self.lat_scale)
            print(f"Loading the previous training set with {len(self.point_list)} samples")
        return

    def generate_struct_point(self, struct_list, init_wp_list, init_spg_list):
        point_list, lat_list = [], []
        # init_wp = np.linspace(0, self.wyckoffs_max - 1, len(self.struct_list)).astype(int)
        # hard to get wp from generated structures
        for i, struct in enumerate(struct_list):
            # lat_vec = struct.lattice_param  # a, b, c, alpha, beta, gamma
            # pos_vec = struct.pos.flatten()
            # spg = struct.space_group
            lat_vec = np.array(list(struct.lattice.abc) + list(struct.lattice.angles))
            pos_vec = struct.frac_coords.flatten()
            # spg = int(SPG.struct_spg_estimate(struct)[1])
            spg = init_spg_list[i]
            wp = init_wp_list[i]
            point = np.concatenate([lat_vec, pos_vec, np.array([spg, wp])])
            lat_list.append(np.expand_dims(lat_vec[:self.lat_dim], axis=0))
            point_list.append(point)
        lat_list = np.concatenate(lat_list, axis=0)
        lat_max = np.max(lat_list) * (1 + self.lat_scale)
        point_list = np.vstack(point_list)
        return point_list, lat_max

    def vec2struct(self, lat_para, pos, spg, wp):
        spg = shift_spg(spg, wyckoffs_dict=self.wyckoffs_dict,
                        spg_u=self.spg_u, spg_l=self.spg_l, logger=self.logger)
        if len(self.wyckoffs_dict[spg]) == 0:
            print(f'No available Wycokff positions exist for spg {spg}')
            return None
        r2w = Random2Wyckoff(wyckoffs_dict=self.wyckoffs_dict,
                             wyckoffs_max=self.wyckoffs_max,
                             species_list=self.species,
                             lat_para=lat_para, pos=pos,
                             spg=spg, wp=wp, logger=self.logger)
        pmg_struct = r2w.get_pmg_struct()
        _, real_spg_n = SPG.struct_spg_estimate(pmg_struct)
        if self.logger is not None:
            self.logger.record_anything3(operation='Generated Space Group', result=real_spg_n)
        return pmg_struct

    def get_struct_from_vec(self, new_x):
        lat_param_dim = self.lat_dim + self.ang_dim
        lat_para = new_x[:lat_param_dim]
        pos = np.reshape(new_x[lat_param_dim: lat_param_dim + self.pos_dim], (-1, 3))
        spg, wp = int(new_x[-2]), int(new_x[-1])
        # spg, wp = 124, 501945  # this will lead to CHGNet Exception
        # spg, wp = 163, 186564
        pmg_struct = self.vec2struct(lat_para=lat_para, pos=pos, spg=spg, wp=wp)
        if self.logger is not None:
            self.logger.record_anything1(operation='Decoded Space Group', result=spg)
            self.logger.record_anything2(operation='Decoded Wyckoff Position', result=wp)
        return pmg_struct, new_x

    def get_vec_from_struct(self, r_pmg_struct, new_x):
        struct_list = []
        spg = new_x[-1]
        wp = new_x[-1]
        for rp_s in r_pmg_struct:
            # spg = SPG.struct_spg_estimate(rp_s)[-1]
            r_struct_point = np.array(list(rp_s.lattice.abc) +
                                      list(rp_s.lattice.angles) +
                                      list(rp_s.frac_coords.flatten()) +
                                      [spg, wp])
            struct_list.append(r_struct_point)
        return struct_list

    def lower_bound(self):
        ll = [self.lat_l] * self.lat_dim + [self.ang_l] * self.ang_dim \
             + [0] * self.pos_dim + [self.spg_l, 0]
        return ll

    def upper_bound(self, lat_max):
        ul = [lat_max] * self.lat_dim + [self.ang_u] * self.ang_dim \
             + [1] * self.pos_dim + [self.spg_u, self.wyckoffs_max]
        return ul
