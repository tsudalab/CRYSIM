import pickle
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
import numpy as np
import pandas as pd
import time
from logzero import logfile, logger
import platform
import torch
import re
from copy import deepcopy

from src.struct_utils import LatticeInfo, SPG, DistFilter, unique_elements, \
    array_to_pmg_for_training_and_recording, save_pymatgen_struct


def print_or_record(recorder, line):
    if recorder is None:
        print(line)
    else:
        recorder.record_line(line)
    return


def load_training_set(follow_dir):
    with open(follow_dir + '/' + 'training_set.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y']


class Logger:
    def __init__(self, model_name, system_name, n_steps, log_dir=None, seed=None,
                 follow_dir=None, require_time=True):
        if require_time:
            time_stamp = time.strftime('%Y-%m-%d-%H-%M-%S',
                                       time.localtime(int(round(time.time() * 1000)) / 1000))
            self.file_name = f'{time_stamp}_{model_name}_{system_name}_s{seed}_n{n_steps}'
        else:
            self.file_name = f'{model_name}_{system_name}_s{seed}_n{n_steps}'

        if log_dir is None:
            log_dir = f'./results/{self.file_name}'
        else:
            log_dir = f'{log_dir}/results/{self.file_name}'
        self.log_dir = log_dir
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        logfile(self.log_dir + '/' + 'log')
        self.log_model_dir = self.log_dir + '/' + 'model'
        Path(self.log_model_dir).mkdir(parents=True, exist_ok=True)
        self.log_status_file = 'log_status.pkl'

        follow_dir = f'./results/{follow_dir}' if follow_dir is not None else follow_dir
        self.follow_dir = follow_dir

        logger.info(f'System: {platform.system()}')
        logger.info(f'Platform: {platform.platform()}')
        logger.info(f'Torch: {torch.__version__}, cuda: {torch.version.cuda}')
        logger.info(f'Model name: {model_name}')
        logger.info(f'System: {system_name}')

        # config = ConfigParser()
        # config.read('./cryspy.in')
        # logger.info(f'Cryspy generation settings:')
        # for option in config.options('structure'):
        #     logger.info(option + ': ' + config.get('structure', option))

        # recording call times
        if self.follow_dir is None:
            self.energy_call = 0
            self.modelling1_call, self.modelling2_call, self.modelling3_call, \
                self.modelling4_call, self.modelling5_call = 0, 0, 0, 0, 0
            self.operation1_call, self.operation2_call, self.operation3_call, \
                self.operation4_call, self.operation5_call, self.operation6_call, \
                self.operation7_call, self.operation8_call, self.operation9_call, \
                self.operation10_call = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            self.new_struct_gen_call = 0
            self.new_struct_suc = 0
        else:
            with open(self.follow_dir + '/' + self.log_status_file, 'rb') as f:
                log_heritage = pickle.load(f)
            self.energy_call, \
                self.modelling1_call, self.modelling2_call, self.modelling3_call, \
                self.modelling4_call, self.modelling5_call, \
                self.operation1_call, self.operation2_call, self.operation3_call, \
                self.operation4_call, self.operation5_call, self.operation6_call, \
                self.operation7_call, self.operation8_call, self.operation9_call, \
                self.operation10_call = \
                log_heritage['energy_call'], \
                    log_heritage['modelling1_call'], log_heritage['modelling2_call'], \
                    log_heritage['modelling3_call'], log_heritage['modelling4_call'], \
                    log_heritage['modelling5_call'], \
                    log_heritage['operation1_call'], log_heritage['operation2_call'], \
                    log_heritage['operation3_call'], log_heritage['operation4_call'], \
                    log_heritage['operation5_call'], log_heritage['operation6_call'], \
                    log_heritage['operation7_call'], log_heritage['operation8_call'], \
                    log_heritage['operation9_call'], log_heritage['operation10_call']
            self.new_struct_gen_call = log_heritage['new_struct_gen_call']
            self.new_struct_suc = log_heritage['new_struct_suc']
            logger.info(f"Inheriting from previous training experiment:")
            logger.info(f"{self.follow_dir}")
            logger.info(f"Beginning from the {self.new_struct_suc}-th iteration")
        return

    def record_dataset_settings(self, init_training_size, iter_sample_num, fail_num):
        self.init_struct_num = init_training_size
        logger.info(f'Initial number of structures: {init_training_size}')
        logger.info(f'Number of added structures for each iteration: {iter_sample_num}')
        logger.info(f'Failed structures will be duplicated {fail_num} times in the dataset to enhance its weight')
        return

    def record_learning_settings(self, epochs, batch_size, lr, lr_s_name, warmup_steps, weight_decay,
                                 alpha, beta, ema_momentum):
        logger.info(f'Training epochs: {epochs}')
        logger.info(f'Training batch size: {batch_size}')
        logger.info(f'Starting learning rate: {lr}')
        logger.info(f'Learning rate decay strategy: {lr_s_name}')
        logger.info(f'Warming up steps before decay: {warmup_steps}')
        logger.info(f'Weight decay for Adam: {weight_decay}')
        logger.info(f'Weight of MSE and PCC in the loss function: {alpha}, {beta}')
        logger.info(f'EMA momentum (for model): {ema_momentum}')
        return

    @staticmethod
    def record_pes_metric(rmse, pcc):
        logger.info(f'RMSE of trained PES: {rmse}')
        logger.info(f'PCC of trained PES: {pcc}')
        return

    def record_random(self, operation, result):
        self._record_anything(operation=operation, call_time=0, result=result)
        return

    def record_line(self, line):
        logger.info(f'{line}')
        return

    @staticmethod
    def _record_anything(operation, call_time, result):
        logger.info(f'[{operation}] -- {call_time}-th -- {result}')
        return

    def _record_time(self, operation, call_time, delta_t):
        self._record_anything(operation, call_time, result=f'time cost: {delta_t:.5} (s)')
        return

    def record_energy_time(self, delta_t):
        self.energy_call += 1
        self._record_time(operation='Relaxation & energy computation', call_time=self.energy_call,
                          delta_t=delta_t)
        return

    def record_modelling1_time(self, operation, delta_t):
        self.modelling1_call += 1
        self._record_time(operation=operation, call_time=self.modelling1_call, delta_t=delta_t)
        return

    def record_modelling2_time(self, operation, delta_t):
        self.modelling2_call += 1
        self._record_time(operation=operation, call_time=self.modelling2_call, delta_t=delta_t)
        return

    def record_modelling3_time(self, operation, delta_t):
        self.modelling3_call += 1
        self._record_time(operation=operation, call_time=self.modelling3_call, delta_t=delta_t)
        return

    def record_modelling4_time(self, operation, delta_t):
        self.modelling4_call += 1
        self._record_time(operation=operation, call_time=self.modelling4_call, delta_t=delta_t)
        return

    def record_modelling5_time(self, operation, delta_t):
        self.modelling5_call += 1
        self._record_time(operation=operation, call_time=self.modelling5_call, delta_t=delta_t)
        return

    def record_anything1(self, operation, result):
        self.operation1_call += 1
        self._record_anything(operation=operation, call_time=self.operation1_call, result=result)
        return

    def record_anything2(self, operation, result):
        self.operation2_call += 1
        self._record_anything(operation=operation, call_time=self.operation2_call, result=result)
        return

    def record_anything3(self, operation, result):
        self.operation3_call += 1
        self._record_anything(operation=operation, call_time=self.operation3_call, result=result)
        return

    def record_anything4(self, operation, result):
        self.operation4_call += 1
        self._record_anything(operation=operation, call_time=self.operation4_call, result=result)
        return

    def record_anything5(self, operation, result):
        self.operation5_call += 1
        self._record_anything(operation=operation, call_time=self.operation5_call, result=result)
        return

    def record_anything6(self, operation, result):
        self.operation6_call += 1
        self._record_anything(operation=operation, call_time=self.operation6_call, result=result)
        return

    def record_anything7(self, operation, result):
        self.operation7_call += 1
        self._record_anything(operation=operation, call_time=self.operation7_call, result=result)
        return

    def record_anything8(self, operation, result):
        self.operation8_call += 1
        self._record_anything(operation=operation, call_time=self.operation8_call, result=result)
        return

    def record_anything9(self, operation, result):
        self.operation9_call += 1
        self._record_anything(operation=operation, call_time=self.operation9_call, result=result)
        return

    def record_anything10(self, operation, result):
        self.operation10_call += 1
        self._record_anything(operation=operation, call_time=self.operation10_call, result=result)
        return

    def record_new_struct_fail(self):
        self.new_struct_gen_call += 1
        logger.info(f'[New structure generation fail] -- {self.new_struct_gen_call}-th structure')
        return

    def record_new_struct_success(self, energy):
        self.new_struct_gen_call += 1
        self.new_struct_suc += 1
        logger.info(f'[RESULT] Energy {self.new_struct_suc}: {energy} for the '
                    f'{self.new_struct_gen_call}-th running')
        return

    def record_new_struct_success_lattice(self, a, b, c, alpha, beta, gamma):
        logger.info(f'[RESULT] Relaxed lattice {self.new_struct_suc}: '
                    f'a: {a}, b: {b}, c: {c}, alpha: {alpha}, beta: {beta}, gamma: {gamma}')
        return

    def record_new_struct_success_spg(self, spg):
        logger.info(f'[RESULT] Space group {self.new_struct_suc}: {spg}')
        return

    def save_log_status(self):
        data = {'energy_call': self.energy_call,
                'modelling1_call': self.modelling1_call, 'modelling2_call': self.modelling2_call,
                'modelling3_call': self.modelling3_call, 'modelling4_call': self.modelling4_call,
                'modelling5_call': self.modelling5_call,
                'operation1_call': self.operation1_call, 'operation2_call': self.operation2_call,
                'operation3_call': self.operation3_call, 'operation4_call': self.operation4_call,
                'operation5_call': self.operation5_call, 'operation6_call': self.operation6_call,
                'operation7_call': self.operation7_call, 'operation8_call': self.operation8_call,
                'operation9_call': self.operation9_call, 'operation10_call': self.operation10_call,
                'new_struct_suc': self.new_struct_suc,
                'new_struct_gen_call': self.new_struct_gen_call}
        with open(self.log_dir + '/' + self.log_status_file, 'wb') as f:
            pickle.dump(data, f)
        return

    def save_training_set(self, X, y):
        data = {'X': X, 'y': y}
        with open(self.log_dir + '/' + 'training_set.pkl', 'wb') as f:
            pickle.dump(data, f)
        return

    def record_results(self, e_list, save=True, split=False):
        tol = 0.02
        e_min3, aver3, _, lowest_first3, lowest_num3, _, _, ratio3 = \
            Analyzer(e_round=3, tolerance=tol).analyze(e_list)
        e_min4, aver4, _, lowest_first4, lowest_num4, _, _, ratio4 = \
            Analyzer(e_round=4, tolerance=tol).analyze(e_list)

        logger.info('\n')
        logger.info('#######################################')
        logger.info('############### Summary ###############')
        logger.info('#######################################')
        logger.info('\n')
        logger.info(f'After {self.new_struct_gen_call} times structure generations, '
                    f'{self.new_struct_suc} structures are generated successfully.')
        # logger.info(f'(including {self.init_struct_num} initially generated input structures)')
        # rate = ((self.new_struct_suc - self.init_struct_num) /
        #         (self.new_struct_gen_call - self.init_struct_num))
        rate = self.new_struct_suc / self.new_struct_gen_call
        logger.info(f'Success rate: {rate}')
        logger.info('Among these structures, ')
        logger.info(f'The lowest energy: {e_min4}')
        logger.info(f'The mean energy: {aver4}')
        logger.info(f'Ratio within {tol} eV higher than the lowest: {ratio4}')

        def _screen_first_lowest(lowest_first, lowest_num):
            init_first = []
            for i in range(self.init_struct_num):
                if i in lowest_first:
                    init_first.append(i)
                    lowest_first.remove(i)
                    lowest_num -= 1
            if lowest_num > 0:
                return init_first, lowest_first[0], lowest_num
            else:
                return init_first, -1, lowest_num

        init_first3, lowest_first3, l_n_3 = _screen_first_lowest(lowest_first3, lowest_num3)
        init_first4, lowest_first4, l_n_4 = _screen_first_lowest(lowest_first4, lowest_num4)
        logger.info(f'First time reaching the lowest (3 decimal places): {init_first3}, {lowest_first3}')
        logger.info(f'Total numbers of reaching the lowest (3 decimal places): '
                    f'[{lowest_num3 - l_n_3}], {l_n_3}')
        logger.info(f'First time reaching the lowest (4 decimal places): {init_first4}, {lowest_first4}')
        logger.info(f'Total numbers of reaching the lowest (4 decimal places): '
                    f'[{lowest_num4 - l_n_4}], {l_n_4}')

        if save:
            np.savez(self.log_dir + '/stat.npz',
                     struct_all=self.new_struct_gen_call,
                     struct_success=self.new_struct_suc,
                     struct_success_rate=rate,
                     lowest_energy=e_min4, average_energy=aver4, thre_ratio=ratio4,
                     init_lowest_3=init_first3, new_lowest_first_3=lowest_first3,
                     init_lowest_num_3=lowest_num3 - l_n_3, new_lowest_num_3=l_n_3,
                     init_lowest_4=init_first4, new_lowest_first_4=lowest_first4,
                     init_lowest_num_4=lowest_num4 - l_n_4, new_lowest_num_4=l_n_4)

        if split:
            oa = OutputAnalysis(self.log_dir)
            oa.split_pos_init_file()
            oa.split_pos_relaxed_file()
        return


class Logger4CSP:
    def __init__(self, logger_obj):
        self.logger = logger_obj

        # lists for recording results
        self.solved_value_list = []
        self.pmg_struct_list = []
        self.pmg_struct_relax_list = []
        self.e_list = []
        self.lat_list = []
        self.spg_dict = {'spg': [], 'spg_number': []}
        return

    def record_success_step(self, i, r_struct, energy, solved_value):
        self.save_relaxed_struct(r_struct, idx=i, idx_off=0)
        self.e_list.append(energy)
        self.logger.record_new_struct_success(energy)
        self.solved_value_list.append(solved_value)
        lat = r_struct.lattice
        a, b, c, alpha, beta, gamma = lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma
        self.logger.record_new_struct_success_lattice(a, b, c, alpha, beta, gamma)
        spg, spg_n = SPG.struct_spg_estimate(r_struct)
        self.logger.record_new_struct_success_spg(spg)
        lat_dict = {'a': a, 'b': b, 'c': c, 'alpha': alpha, 'beta': beta, 'gamma': gamma}
        self.lat_list.append(lat_dict)
        self.spg_dict['spg'].append(spg), self.spg_dict['spg_number'].append(spg_n)
        self.save_result_data()
        return

    def save_init_struct(self, struct, idx, idx_off):
        struct = array_to_pmg_for_training_and_recording(struct)
        save_pymatgen_struct(struct, idx, idx_off, f'{self.logger.log_dir}/POS_init')
        self.pmg_struct_list.append(struct)
        print(f"Successfully save init structure {idx_off + idx}")
        return

    def save_relaxed_struct(self, struct, idx, idx_off):
        struct = array_to_pmg_for_training_and_recording(struct)
        save_pymatgen_struct(struct, idx, idx_off, f'{self.logger.log_dir}/POS_relaxed')
        # not adding the following line will lead to _pickle.PicklingError
        struct = Structure(lattice=struct.lattice, species=struct.species,
                           coords=struct.frac_coords, coords_are_cartesian=False)
        self.pmg_struct_relax_list.append(struct)
        print(f"Successfully save relaxed structure {idx_off + idx}")
        return

    def save_result_data(self):
        data = {'energy': self.e_list, 'lattice': self.lat_list,
                'solved_value': self.solved_value_list,
                'spg': self.spg_dict['spg'], 'spg_n': self.spg_dict['spg_number'],
                'init_struct': self.pmg_struct_list,
                'relaxed_struct': self.pmg_struct_relax_list}
        with open(f'{self.logger.log_dir}/accumulated-results.pkl', 'wb') as f:
            pickle.dump(data, f)
        self.logger.save_log_status()
        return

    def draw(self, name):
        plt.figure(figsize=(8, 3), dpi=90)
        y = self.e_list
        plt.subplot(1, 2, 1)
        plt.plot(pd.DataFrame(y).cummin())
        plt.plot(y, '.')
        plt.ylim(np.min(y) * 1.05, np.max(y) * 0.95)  # y < 0
        plt.ylabel("Predicted Relaxed Energy (eV)")
        plt.xlabel("Sampling No.")
        plt.show()
        plt.savefig(f'{self.logger.log_dir}/{name}.png', bbox_inches='tight')
        plt.tight_layout()
        return

    def finish_logging(self):
        self.draw('stat')
        self.logger.record_results(e_list=self.e_list, save=True, split=False)
        return self.e_list, self.pmg_struct_list, self.pmg_struct_relax_list


class Analyzer:
    def __init__(self, e_round=2, threshold=None, tolerance=0.02):
        self.e_round = e_round
        self.threshold = threshold
        self.tolerance = tolerance
        return

    def analyze(self, result_e, n_init=0, n_steps=None):
        result_e = np.array(result_e) if isinstance(result_e, list) else result_e
        result_e = result_e[n_init:]
        result_e = np.array([e for e in result_e if e is not None])
        if n_steps is not None:
            result_e = result_e[:n_steps]
        aver = np.mean(result_e).round(self.e_round)
        e_min = np.min(result_e).round(self.e_round)
        result_e = result_e.round(self.e_round)
        num = len(result_e)
        threshold = e_min + self.tolerance if self.threshold is None else self.threshold + self.tolerance
        threshold_first = np.where(result_e <= threshold)[0].tolist()
        try:
            n_th = len(threshold_first)
            threshold_first = threshold_first[0]
        except IndexError:
            pass
        threshold_num = np.sum(np.where(result_e < threshold, 1, 0))
        threshold_ratio = np.round(threshold_num / num, 4)
        lowest_first = np.where(result_e == e_min)[0].tolist()
        lowest_num = np.sum(np.where(result_e == e_min, 1, 0))
        return e_min, aver, num, lowest_first, lowest_num, threshold_first, threshold_num, threshold_ratio

    def analyze_report(self, output):
        lowest = 'lowest' if self.threshold is None else self.threshold
        e_min, aver, num, lowest_first, lowest_num, threshold_first, threshold_num, threshold_ratio = output
        print(f'Total numbers of generated structures: {num}\n'
              f'Mean: {aver} eV\n'
              f'Lowest: {e_min} eV ({lowest_first}-th)\n'
              f'Lowest reaching number: {lowest_num}\n'
              f'Within {lowest} + {self.tolerance} threshold: {threshold_num} '
              f'(first reaching: {threshold_first}-th)\n'
              f'Within {lowest} + {self.tolerance} threshold ratio: {threshold_ratio}'
              )
        return


class OutputAnalysis:
    def __init__(self, log_dir,
                 log_file=None, pos_init_file=None, pos_relaxed_file=None, e_list_file=None):
        self.log_dir = log_dir
        self.log_file = 'log' if log_file is None else log_file
        self.pos_init_file = 'POS_init' if pos_init_file is None else pos_init_file
        self.pos_relaxed_file = 'POS_relaxed' if pos_relaxed_file is None else pos_relaxed_file
        self.results_list_file = 'accumulated-results.pkl' if e_list_file is None else e_list_file
        return

    def _read_files(self, file_name, binary, label=None):
        if not binary:
            if isinstance(self.log_dir, list):
                file = []
                for log_d in self.log_dir:
                    with open(f'{log_d}/{file_name}', 'r') as f:
                        lines = f.readlines()
                        file += lines
            else:
                with open(f'{self.log_dir}/{file_name}', 'r') as f:
                    file = f.readlines()
        else:
            if label is not None:
                if isinstance(self.log_dir, list):
                    file = []
                    for log_d in self.log_dir:
                        with open(f'{log_d}/{file_name}', 'rb') as f:
                            data = pickle.load(f)
                            file += data[label]
                else:
                    with open(f'{self.log_dir}/{file_name}', 'rb') as f:
                        data = pickle.load(f)
                        file = data[label]
            else:
                if isinstance(self.log_dir, list):
                    file = {}
                    for log_d in self.log_dir:
                        with open(f'{log_d}/{file_name}', 'rb') as f:
                            data = pickle.load(f)
                            for key in data.keys():
                                if key in file.keys():
                                    file[key] += data[key]
                                else:
                                    file[key] = data[key]
                else:
                    with open(f'{self.log_dir}/{file_name}', 'rb') as f:
                        file = pickle.load(f)
        return file

    def load_log(self):
        self.log_list = self._read_files(file_name=self.log_file, binary=False)
        return

    def load_pos_relaxed(self):
        self.pos_relaxed = self._read_files(file_name=self.pos_relaxed_file, binary=False)
        self.pos_relaxed_id_list = [i for i, line in enumerate(self.pos_relaxed) if "ID_" in line]
        self.pos_relaxed_list = self._read_pos_file(self.pos_relaxed, id_list=self.pos_relaxed_id_list)
        return

    def load_pos_init(self):
        self.pos_init = self._read_files(file_name=self.pos_init_file, binary=False)
        self.pos_init_id_list = [i for i, line in enumerate(self.pos_init) if "ID_" in line]
        self.pos_init_list = self._read_pos_file(self.pos_init, id_list=self.pos_init_id_list)
        return

    def load_e_list(self, filter_struct=False):
        self.e_list = self._read_files(file_name=self.results_list_file, binary=True, label='energy')
        self.e_list = np.array(self.e_list)
        self.e_list = [e for e in self.e_list if e is not None]
        self.e_list = np.array(self.e_list)
        if filter_struct:
            filter_num = self._filter_unreasonable_struct_for_energy()
            return filter_num
        else:
            return None

    def load_result(self):
        self.results_list = self._read_files(file_name=self.results_list_file, binary=True)
        return

    def load(self, filter_struct=False):
        self.load_result()
        return self.load_e_list(filter_struct=filter_struct)

    def _filter_unreasonable_struct_for_energy(self):
        if not hasattr(self, "results_list"):
            self.load_result()
        pos_relaxed_list = self.results_list['relaxed_struct']
        species_list = pos_relaxed_list[0].species
        elements, num_species_atom = unique_elements(species_list)
        dist_filter = DistFilter(atom_num_list=num_species_atom, dist=1.0)
        filter_num = 0
        for i in range(len(self.e_list)):
            if not dist_filter(pos_relaxed_list[i], test_lattice=False):
                self.e_list[i] = 999999
                filter_num += 1
        print(f"{filter_num} structures are filtered out deal to too small pair-wise distances")
        return filter_num

    @staticmethod
    def _read_pos_file(cum_pos_file, id_list):
        pos_document = []
        pos_file_len = id_list[1] - id_list[0]
        for i, idx in enumerate(id_list):
            line_idx = idx
            pos_file = []
            while line_idx < idx + pos_file_len:
                pos_line = []
                content = re.findall(r'[\d.-]+', cum_pos_file[line_idx])
                for word in content:
                    try:
                        num = np.float64(word)
                        pos_line.append(num)
                    except ValueError:
                        continue
                pos_file.append(pos_line)
                line_idx += 1
            lat_param = pos_file[2: 5]
            for lv in range(len(lat_param)):  # deal with nan
                if len(lat_param[lv]) != 3:
                    lat_param[lv].append(np.nan)
            species = []
            for s in range(len(pos_file[6])):
                species += [s + 1] * int(pos_file[6][s])
            dire = pos_file[8: 8 + pos_file_len]
            pos_document.append((np.array(lat_param), np.array(dire), np.array(species)))
        return pos_document

    @staticmethod
    def _split_cum_pos_file(cum_pos_file, output_file_name, id_list, idx_in_need=None):
        pos_file_len = id_list[1] - id_list[0]
        idx_in_need = id_list if idx_in_need is None else idx_in_need
        for idx in idx_in_need:
            line_idx = idx
            with open(output_file_name + f'_{id_list.index(idx) + 1}', 'w') as f:
                while line_idx < idx + pos_file_len:
                    f.write(cum_pos_file[line_idx])
                    line_idx += 1
        return

    def split_pos_init_file(self, output_file_name=None):
        self.load_pos_init()
        output_file_name = 'POS_init' if output_file_name is None else output_file_name
        self._split_cum_pos_file(cum_pos_file=self.pos_init,
                                 output_file_name=self.log_dir + '/' + output_file_name,
                                 id_list=self.pos_init_id_list)
        print("Finish splitting init")
        return

    def split_pos_relaxed_file(self, output_file_name=None):
        self.load_pos_relaxed()
        output_file_name = 'POS_relaxed' if output_file_name is None else output_file_name
        self._split_cum_pos_file(cum_pos_file=self.pos_relaxed,
                                 output_file_name=self.log_dir + '/' + output_file_name,
                                 id_list=self.pos_relaxed_id_list)
        print("Finish splitting relaxed")
        return

    def split_stablest_struct(self, num_struct, filter_struct, decimal=None, e_threshold=None):
        if isinstance(self.log_dir, str):
            log_dir = self.log_dir + '/filter' if filter_struct else self.log_dir
        elif isinstance(self.log_dir, list):
            log_dir = self.log_dir[-1] + '/filter' if filter_struct else self.log_dir[-1]
        else:
            raise NotImplementedError
        if filter_struct:
            if Path(log_dir).exists():
                shutil.rmtree(log_dir)
            Path(log_dir).mkdir(parents=False, exist_ok=False)

        e_list = deepcopy(self.e_list)
        pos_init_list = self.results_list['init_struct']
        pos_relaxed_list = self.results_list['relaxed_struct']
        if e_threshold is not None:
            e_list = np.where(e_list < e_threshold, 0, e_list)
        e_list = e_list.round(decimal) if decimal is not None else e_list
        struct_ids = np.argsort(e_list)  # ascending
        for i in range(num_struct):
            idx = struct_ids[i]
            energy = e_list[idx]

            pos_init = pos_init_list[idx]
            pos_relaxed = pos_relaxed_list[idx]
            ini_output_file_name = f'POSCAR_e{decimal}_{energy}-{idx}'
            ini_output_file_name = ini_output_file_name.replace('.', 'dot')
            re_output_file_name = f'POSCAR_relaxed_e{decimal}_{energy}-{idx}'
            re_output_file_name = re_output_file_name.replace('.', 'dot')
            pos_init.to(fmt='poscar', filename=log_dir + '/' + ini_output_file_name)
            pos_relaxed.to(fmt='poscar', filename=log_dir + '/' + re_output_file_name)
        print("Finish sampling")
        return

    def split_energy_struct(self, num_struct, energy=None, decimal=None, output_file_name=None):
        self.load_e_list()
        self.load_pos_relaxed()

        e_list = self.e_list.round(decimal) if decimal is not None else self.e_list
        np.random.seed(0)
        energy = np.min(e_list) if energy is None else energy
        output_file_name = f'POSCAR_relaxed_e{decimal}_{energy}' \
            if output_file_name is None else output_file_name
        # output_file_name = f'POSCAR_init_e{decimal}_{energy}' \
        #     if output_file_name is None else output_file_name
        output_file_name = output_file_name.replace('.', 'dot')

        lowest_e_idx = np.argwhere(e_list == energy).flatten()
        try:
            e_num = len(lowest_e_idx)
            if e_num == 0:
                print(f'No relaxed structures have the energy {energy}.')
                return
        except TypeError:
            lowest_e_idx = [lowest_e_idx]
            e_num = 1
        if num_struct < e_num:
            lowest_e_idx = np.random.choice(lowest_e_idx, num_struct)
        idx_in_need = [self.pos_relaxed_id_list[i] for i in lowest_e_idx]

        self._split_cum_pos_file(cum_pos_file=self.pos_relaxed,
                                 output_file_name=self.log_dir + '/' + output_file_name,
                                 id_list=self.pos_relaxed_id_list,
                                 idx_in_need=idx_in_need)
        print("Finish sampling")
        return

    def split_specified_file(self, relaxed, idx_in_need, output_file_name):
        if relaxed:
            self.load_pos_relaxed()
            pos = self.pos_relaxed
            pos_id_list = self.pos_relaxed_id_list
        else:
            self.load_pos_init()
            pos = self.pos_init
            pos_id_list = self.pos_init_id_list

        output_file_name = 'POSCAR' if output_file_name is None else output_file_name
        idx_in_need = [pos_id_list[i] for i in idx_in_need]
        self._split_cum_pos_file(cum_pos_file=pos,
                                 output_file_name=self.log_dir + '/' + output_file_name,
                                 id_list=pos_id_list,
                                 idx_in_need=idx_in_need)
        print("Finish obtaining")
        return

    def collect_lattice(self):
        lat_list = []
        for pos in self.pos_relaxed_list:
            lat = pos[0]
            a, b, c, alpha, beta, gamma = LatticeInfo(lat)()
            lat_dict = {'a': a, 'b': b, 'c': c, 'alpha': alpha, 'beta': beta, 'gamma': gamma}
            lat_list.append(lat_dict)
        return lat_list

    def final_sym(self, spg_stable_num):
        self.load_result()
        spg_list = self.results_list['spg_n']
        sym_stable_list = np.argwhere(spg_list == spg_stable_num).flatten()
        return sym_stable_list

    def final_energy(self, e_round=1):
        self.load_e_list()
        _, _, _, e_list, _, _, _, _ = Analyzer(e_round=e_round).analyze(self.e_list)
        return e_list

    def compare_structure_with_ground_truth(self, poscar_true_dir, output_name, return_lowest=False):
        e_deci = 2
        e_list = deepcopy(self.e_list)
        e_list = np.round(e_list, e_deci)
        struct_ids_temp = np.argsort(e_list, kind='stable')
        e_ref = e_list[struct_ids_temp[0]]
        struct_ids = []
        for i in struct_ids_temp:
            if np.abs(e_ref - e_list[i]) < 0.02:
                struct_ids.append(i)
            else:
                print(f'Gathered {len(struct_ids)} structures')
                break
        num_e_tol = len(struct_ids)
        e_low_id = 999
        e_low = 999

        pos_relaxed_list = self.results_list['relaxed_struct']
        candidate_list, dist_list = [], []
        smr = StructureMatcher(stol=0.5, ltol=0.3, angle_tol=10.0)
        true_struct = Structure.from_file(poscar_true_dir)
        for curr_idx in struct_ids:
            if e_list[curr_idx] >= 0:
                break
            # distance
            dist = smr.get_rms_dist(pos_relaxed_list[curr_idx], true_struct)
            if dist is not None:
                candidate_list.append(curr_idx)
                dist_list.append(np.round(dist[0], 5))
            # energy
            if (e_low > e_list[curr_idx]) or ((e_low == e_list[curr_idx]) and (e_low_id > curr_idx)):
                e_low = e_list[curr_idx]
                e_low_id = curr_idx

        if not return_lowest:
            if len(candidate_list) == 0:
                print(f'Fail for {output_name}')
                return num_e_tol, [struct_ids[0]], [999], \
                    [pos_relaxed_list[struct_ids[0]]], [np.round(e_list[struct_ids[0]], e_deci)]
            else:
                print(f'Succeed for {output_name}')
                return num_e_tol, candidate_list, dist_list, \
                    [pos_relaxed_list[j] for j in candidate_list], [np.round(e_list[j], e_deci) for j in candidate_list]
        else:
            if len(candidate_list) == 0:
                print(f'Fail for {output_name}')
                return num_e_tol, [struct_ids[0]], [999], \
                    [pos_relaxed_list[struct_ids[0]]], [np.round(e_list[struct_ids[0]], e_deci)], \
                    e_low_id, e_low
            else:
                print(f'Succeed for {output_name}')
                return num_e_tol, candidate_list, dist_list, \
                    [pos_relaxed_list[j] for j in candidate_list], [np.round(e_list[j], e_deci) for j in candidate_list], \
                    e_low_id, e_low


class CalypsoOutputAnalysis(OutputAnalysis):
    def __init__(self, log_dir, iter_num, generate_file_num):
        super().__init__(log_dir=log_dir)
        self.log_dir = log_dir
        self.log_file = None
        self.pos_init_file = None
        self.pos_relaxed_file = None
        self.results_list_file = f'run/calypso_energy_{iter_num}_{generate_file_num}.pkl'
        return
