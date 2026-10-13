# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
#  2021/03/17 下午 08:57
#                 _____   _   _   _____   __   _   _____  
#                /  ___| | | | | | ____| |  \ | | /  ___| 
#                | |     | |_| | | |__   |   \| | | |     
#                | |     |  _  | |  __|  | |\   | | |  _  
#                | |___  | | | | | |___  | | \  | | |_| | 
#                \_____| |_| |_| |_____| |_|  \_| \_____/ 
# ------------------------------------------------------------------------

import copy
import itertools
import functools
import os
import random
import time
import pickle
import pandas as pd
from logzero import logger, logfile

import sys
sys.path.append('../..')

from crysim.src.registry import FileLoader
from crysim.src import SYM_ROOT
from pyxtal.symmetry import Wyckoff_position


class WPDict(FileLoader):
    def __init__(self, dim, load_cache=True):
        self.dim = dim
        current_path = SYM_ROOT
        file_name = f'wp_list-{dim}.pkl'
        file_dire = os.path.join(current_path, file_name)
        super().__init__(file_dire=file_dire, load_cache=load_cache)
        return

    def generate(self):
        wp_dict = {}
        for spg in range(1, 231):
            wp_dict[spg] = []
            i = 0
            try:
                while True:
                    try:
                        wp_obj = Wyckoff_position.from_group_and_index(spg, i, dim=self.dim)
                    except IndexError:
                        break
                    wp_dict[spg].append((wp_obj.multiplicity, wp_obj.letter, wp_obj.get_dof()))
                    i += 1
            except ValueError:  # this SPG is incompatible with the dimension
                del wp_dict[spg]
                break
        self.file = wp_dict
        return


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_data_bin(path, filename, data=None):
    if path is None:
        path = os.path.split(filename)[0]
        check_path(path)
        file_path = filename
    else:
        check_path(path)
        file_path = os.path.join(path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)


def read_data_bin(path, filename):
    if path is None:
        file_path = filename
    else:
        file_path = os.path.join(path, filename)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        # data = pickle.load(f, encoding='iso-8859-1')
    return data


def print_run_info(info):
    def wrapper(func):
        def _wrapper(*args, **kargs):
            print(info + '! Running ...')
            res = func(*args, **kargs)
            print(info + ' OK!')
            print('')
            return res

        return _wrapper

    return wrapper


def custom_sort_key(wp_list):
    constant_flag = 0
    for wps in wp_list:
        for wp in wps:
            if not (('x' in wp) or ('y' in wp) or ('z' in wp)):
                constant_flag = 1
                break
    return constant_flag


def weighted_shuffle(items, weights):
    """
    reference:
    https://stackoverflow.com/questions/29972712/algorithm-to-shuffle-an-array-randomly-based-on-different-weights
    Weighted Random Sampling (2005; Efraimidis, Spirakis)
    https://utopia.duth.gr/~pefraimi/research/data/2007EncOfAlg.pdf
    """
    random.seed(42)
    order = sorted(range(len(items)), key=lambda i: -random.random() ** (1.0 / weights[i]))
    # order = sorted(range(len(items)), key=lambda i: -random.random() * np.sin(np.pi / 2 * weights[i] / sum(weights)))
    return [items[i] for i in order]


class GetWyckoffPosition:
    def __init__(self,
                 sg, atom_num, mode, dim,
                 spg_wpc_l,
                 shuffle=False,
                 # max_count=1,
                 verbose=False,
                 save_path='.'
                 ):
        start_time = time.time()
        self.save_path = save_path
        self.mode = mode
        self.dim = dim

        self.sg = sg
        self.spg_wpc_l = spg_wpc_l

        # self.pool = Pool(8)

        self.wyckoffs = []

        self.get_wyckoffs(atom_num, shuffle=shuffle)

        if verbose:
            end_time = time.time()
            print(sg, 'OK! time used:', end_time - start_time, 's', 'Count:', len(self.wyckoffs))

    def get_wyckoffs(self, atom_num, shuffle):
        wyckoff_position = eval(self.spg_wpc_l) if self.mode == 'gn-oa' else self.spg_wpc_l
        self.wyckoffs = self.combination_wp_all(wyckoff_position, atom_num, mode=self.mode, dim=self.dim, shuffle=shuffle)

        # if is_random:
        #     self.wyckoffs = self.combination_wp_random(wyckoff_position, atom_num, mode=self.mode, is_shuffle=True)
        # else:
        #     # self.wyckoffs = self.combination_wp_all(wyckoff_position, atom_num, mode=self.mode, shuffle=True)
        #     # if len(self.wyckoffs) != 0 and self.wyckoffs[0] is None:
        #     #     self.wyckoffs = self.combination_wp_all(wyckoff_position, atom_num, mode=self.mode, shuffle=False)
        #     self.wyckoffs = self.combination_wp_all(wyckoff_position, atom_num, mode=self.mode, dim=self.dim, shuffle=False)

    def combination_wp_all(self,
                           wyckoff_position: list,
                           atom_num: list,
                           mode: str,
                           dim: int,
                           is_fast: bool = True,
                           max_count: int = 100e4,
                           # max_count: int = 10000,
                           max_trial: int = 1e9,  # max trial between two adjacent successful combinations
                           # shuffle: bool = True,
                           shuffle: bool = False
                           ):
        """
        By default, without shuffling, WPs of elements are sorted to make WPCs including more general WPs be in the
            front of the list before combining into WPCs of the system.

        is_fast == False: Try and collect all successful WPCs;
        is_fast == True & shuffle == False: Try and collect successful WPCs, until max_count or max_trial
            reaches; tricks are added in case of there are too many possible WPCs to include more WPCs based on
            special WPs;
        is_fast == True & shuffle == True: After shuffling, try and collect successful WPCs, until max_count or
            max_trial reaches.
        """
        wp_part_list = []
        wp_bin_name = 'part_wp' if mode == 'gn-oa' else 'part_wp-alphabet'
        wp_bin_name += f'-{dim}'
        for an in atom_num:
            part_wp_an_path = os.path.join(self.save_path, wp_bin_name, str(an) + '_' + str(self.sg))
            if os.path.isfile(part_wp_an_path):
                wp_part = read_data_bin(None, part_wp_an_path)
            else:
                wp_part = self.combination_wp_part(wyckoff_position, an, mode=mode)
                save_data_bin(None, part_wp_an_path, data=wp_part)
            if not wp_part:
                return []

            # make the wp_parts that do not contain static positions be in the front of the list
            # if self.mode == 'gn-oa':
            #     wp_part = sorted(wp_part, key=custom_sort_key)
            # else:
            #     wp_part = sorted(wp_part)[::-1]
            # wp_part has already been sorted, since the wp dicts have been sorted

            wp_part_list.append(wp_part)

        is_use_fast = False
        is_use_reverse = False
        wp_part_len_list = [len(wpp) for wpp in wp_part_list]
        wp_all_count = functools.reduce(lambda x, y: x * y, wp_part_len_list)
        # print(wp_all_count)
        if is_fast:
            if wp_all_count > max_count:
                is_use_fast = True
                if shuffle:
                    wp_part_list_tmp = []
                    for wpp in wp_part_list:
                        random.shuffle(wpp)
                        # wpp = weighted_shuffle(wpp, weights=[i**10 for i in list(range(1, len(wpp) + 1))[::-1]])
                        # let more general WPs have higher possibility to be in the front in shuffling
                        wp_part_list_tmp.append(wpp)
                    wp_part_list = wp_part_list_tmp

            if (not shuffle) and (wp_all_count > 100 * max_count):
                is_use_fast = True
                is_use_reverse = True
                wp_part_reverse_list = []
                for i, wpp in enumerate(wp_part_list):
                    if i == 0:
                        wpp = wpp[::-1]
                    wp_part_reverse_list.append(wpp)

        def _combine_wp(_wp_part_list, _is_use_fast, _max_count, _max_trial, _max_total_explore):
            wp_all_list = []
            wp_product = itertools.product(*_wp_part_list)
            _space = 1
            for _part in _wp_part_list:
                _space *= len(_part)
            logger.info(f'Start with {_space} solutions, {_max_total_explore} total trials')
            num_trial, num_explore = 0, 0
            for p in wp_product:
                if _max_total_explore is not None:
                    if num_explore >= _max_total_explore:
                        logger.info(f'Finish with {len(wp_all_list)} valid WPCs, after maximum {_max_total_explore} trials for the SG')
                        return wp_all_list

                num_trial += 1
                num_explore += 1
                if self.mode == 'gn-oa':
                    pp = [set(k for j in i for k in j) for i in p]
                else:
                    pp = [set(i) for i in p]

                # res = list(pp[0].intersection(*pp[1:]))
                # original implementation, getting intersection of wp combinations for all elements
                # however, what we need is the union of all pair-wise intersections
                pp_pairs = itertools.combinations(pp, 2)
                intersections = [set(a).intersection(b) for a, b in pp_pairs]
                res = set().union(*intersections)

                for ri in res:
                    stat_crit = (not ('x' in ri or 'y' in ri or 'z' in ri)) if self.mode == 'gn-oa' else (ri[2] == 0)
                    if stat_crit:  # static coordinates conflict
                        p = []
                        break
                if p:
                    wp_all_list.append(p)
                    num_trial = 0
                    if _is_use_fast and len(wp_all_list) >= _max_count:
                        logger.info(f'Finish with {len(wp_all_list)} valid WPCs since max_count is reached')
                        return wp_all_list

                if (num_trial % 1e6 == 0) and (num_trial != 0):
                    logger.info(f'{num_trial} trials without any valid WPC, current valid WPCs {len(wp_all_list)}')

                if (num_trial >= _max_trial) & (_max_trial > 0):
                    # if len(wp_all_list) > int(max_count * 0.01):
                    #     return wp_all_list
                    # else:
                    #     return [None]  # if generated number with shuffling is small, try restarting from general WPs
                    logger.info(f'Finish with {len(wp_all_list)} valid WPCs since max_trial is reached')
                    return wp_all_list

            logger.info(f'Finish with {len(wp_all_list)} valid WPCs, the complete list of WPCs for the SG')
            return wp_all_list

        if is_use_fast:
            logger.info("Combine and try until max_count or max_trial is reached")
        else:
            logger.info("Combine and try all WPCs")
        if not is_use_reverse:
            wp_all_list_output = _combine_wp(_wp_part_list=wp_part_list, _is_use_fast=is_use_fast,
                                             _max_count=max_count, _max_trial=max_trial, _max_total_explore=None)
        else:
            wp_all_list_output = []
            max_explore_time = int(len(wp_part_list[0]) / 2)
            for part in wp_part_list[1:]:
                max_explore_time *= len(part)
            wp_all_list_output += _combine_wp(_wp_part_list=wp_part_list, _is_use_fast=is_use_fast,
                                              _max_count=int(max_count / 2), _max_trial=max_trial, _max_total_explore=max_explore_time)
            logger.info(f'Finish general part, start special part')
            wp_special = _combine_wp(_wp_part_list=wp_part_reverse_list, _is_use_fast=is_use_fast,
                                     _max_count=int(max_count / 2), _max_trial=max_trial, _max_total_explore=max_explore_time)

            if len(wp_special) == 0 and len(wp_all_list_output) >= int(max_count / 10):
                logger.info(f'Fail in special part, rerun with general WPCs')
                wp_all_list_output = _combine_wp(_wp_part_list=wp_part_list, _is_use_fast=is_use_fast,
                                                 _max_count=max_count, _max_trial=max_trial, _max_total_explore=None)
            else:
                # wp_all_list_output += wp_special[::-1]
                wp_all_list_output += wp_special

        return wp_all_list_output

    def combination_wp_random(self,
                              wyckoff_position: list,
                              atom_num: list,
                              mode: str,
                              max_count: int = 1,
                              is_shuffle: bool = True):
        """
        max_count only one now
        """

        if max_count and is_shuffle:
            random.shuffle(wyckoff_position)

        wp_part_list = []
        for an in atom_num:
            # print(len(wyckoff_position))
            wp_part = self.combination_wp_part(wyckoff_position, an, mode=mode, max_count=max_count)
            # wp_part = wp_part[0]

            if not wp_part:
                return []

            wp_part_list.append(wp_part)

            for wpp in wp_part:
                for wp in wpp:
                    stat_crit = (not ('x' in '_'.join(wp) or 'y' in '_'.join(wp) or 'z' in '_'.join(wp))) \
                        if self.mode == 'gn-oa' else (wp[2] == 0)
                    if stat_crit:  # dof == 0
                        wyckoff_position.remove(wp)

        wp_all_list = list(itertools.product(*wp_part_list))

        return wp_all_list

    @staticmethod
    def combination_wp_part(wyckoff_position: list,
                            atom_num_part: int,
                            mode: str,
                            max_count: int = 100000
                            # max_count: int = -1
                            ) -> list:
        def dfs(target: int,
                index: int,
                temp: list,
                temp_num: list):
            if sum(temp_num) == de:
                temp.sort()
                if temp not in result:
                    result.append(temp)
                if len(result) == max_count:  # 达到max_count个组合数，引发异常，结束搜索
                    raise Exception()

            for i in range(index, len(wp_num)):
                check = (wp[i] in temp and not ('x' in ','.join(wp[i]) or 'y' in ','.join(wp[i]) or 'z' in ','.join(wp[i]))) \
                    if mode == 'gn-oa' else (wp[i] in temp and wp[i][2] == 0)
                if check:
                    continue

                if target > wp_num[i]:
                    dfs(target - wp_num[i], i, temp + [wp[i]], temp_num + [wp_num[i]])

                elif target == wp_num[i]:
                    dfs(target - wp_num[i], i, temp + [wp[i]], temp_num + [wp_num[i]])

                elif target < wp_num[i]:
                    continue

        result = []
        _index = 0
        _temp = []
        _temp_num = []
        de = atom_num_part
        wp = copy.deepcopy(wyckoff_position)
        wp_num = [len(i) for i in wp] if mode == 'gn-oa' else [i[0] for i in wp]
        try:
            dfs(atom_num_part, _index, _temp, _temp_num)
        except:
            pass
        # print(result)

        return result


@print_run_info("Get the Wyckoff position combinations")
def get_all_wyckoff_combination(sg_list, atom_num, mode, dim):
    current_path = SYM_ROOT
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z']
    wyckoff_combination_type = ''.join([alphabet[i] + str(atom_num[i]) for i in range(len(atom_num))])

    wyckoffs_dict = {}

    if mode == 'pyx':
        spg_wyckoff_dict = WPDict(dim=dim)()  # (multiplicity, label, dof)
    elif mode == 'gn-oa':
        wyckoff_df = pd.read_csv(current_path + "/wyckoff_list.csv")
        spg_wyckoff_dict = wyckoff_df["0"]
    else:
        raise Exception("Do not support the WPC computation mode")

    max_wyckoffs_count = 0
    wpc_bin_name = 'wp_sg' if mode == 'gn-oa' else 'wp_sg-alphabet'
    wpc_bin_name += f'-{dim}'
    for sg_i in sg_list:
        sg_i_path = os.path.join(current_path, wpc_bin_name, wyckoff_combination_type + '_' + str(sg_i))
        if os.path.isfile(sg_i_path):
            wp = read_data_bin(None, sg_i_path)
        else:
            logger.info(f"Starting the space group {sg_i}")
            wp = GetWyckoffPosition(sg_i, atom_num, mode=mode, dim=dim, spg_wpc_l=spg_wyckoff_dict[sg_i],
                                    shuffle=False, verbose=False, save_path=current_path).wyckoffs
            save_data_bin(None, sg_i_path, data=wp)
        if None in wp:
            wp = []
        wyckoffs_dict[sg_i] = wp
        max_wyckoffs_count = max(len(wp), max_wyckoffs_count)
        logger.info(f"Get spg {sg_i} with {len(wp)} wps")

    return wyckoffs_dict, max_wyckoffs_count


if __name__ == '__main__':
    # _sg_list = list(range(2, 230+1))
    _sg_list = list(range(2, 80+1))
    # _sg_list = [10]
    _atom_num = [2, 4]
    # _atom_num = [2, 17]
    # _atom_num = [6, 51]
    # _atom_num = [4, 34]
    # _atom_num = [12, 8, 12, 48]
    # _atom_num = [4, 2, 2, 4, 16]
    # _atom_num = [28]
    # _atom_num = [12, 102]
    # _atom_num = [4, 4, 8]
    _a, _b = get_all_wyckoff_combination(sg_list=_sg_list, atom_num=_atom_num, mode='pyx', dim=2)
    # _wyckoff_df = pd.read_csv(SYM_ROOT + "/wyckoff_list.csv")

    # from crysim.src.spg.spg_utils import pyx_wpc_to_str
    # for k, v in _a.items():
    #     wpc_str = [pyx_wpc_to_str(k, wpc, 2) for wpc in v]
    #     _a[k] = wpc_str

    from crysim.src.spg.spg_utils import pyx_wp_to_str
    import numpy as np
    _w = WPDict(2)()
    for k, v in _w.items():
        wp_str = [pyx_wp_to_str(k, wp, 2) for wp in v]
        _w[k] = wp_str
    _wz = {}
    for k, v in _w.items():
        wz = []
        for wp in v:
            wpz = [w.split(',')[-1][1:] for w in wp]
            wz.append(wpz)
        _wz[k] = wz
    
    # from functools import reduce
    # import operator
    # _ff = []
    # for v in _w.values():
    #     _ff += reduce(operator.concat, v)
    # _z = np.unique([i.split(',')[-1][1:] for i in _ff])
    
    _wpz_type = {}
    for k, v in _wz.items():
        wpt = []
        for wp in v:
            if '0' in wp and 'z' not in wp and '-z' not in wp:
                wpt.append(0)
            elif 'z' in wp and '-z' in wp and '0' not in wp:
                wpt.append(1)
            elif 'z' in wp and '-z' not in wp and '0' not in wp:
                wpt.append(2)
            else:
                print(k, wp)
                raise Exception
        _wpz_type[k] = wpt
