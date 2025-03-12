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
from logzero import logger


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
    order = sorted(range(len(items)), key=lambda i: -random.random() ** (1.0 / weights[i]))
    # order = sorted(range(len(items)), key=lambda i: -random.random() * np.sin(np.pi / 2 * weights[i] / sum(weights)))
    return [items[i] for i in order]


class GetWyckoffPosition:
    def __init__(self,
                 sg, atom_num,
                 is_random=False,
                 # is_random=True,
                 # max_count=1,
                 verbose=False,
                 save_path='.'
                 ):
        start_time = time.time()
        self.save_path = save_path

        self.sg = sg
        self.wyckoff_df = pd.read_csv(os.path.split(__file__)[0] + "/wyckoff_list.csv")

        # self.pool = Pool(8)

        self.wyckoffs = []

        self.get_wyckoffs(atom_num, is_random=is_random)

        if verbose:
            end_time = time.time()
            print(sg, 'OK! time used:', end_time - start_time, 's', 'Count:', len(self.wyckoffs))

    def get_wyckoffs(self, atom_num, is_random):
        wyckoff_position = eval(self.wyckoff_df["0"][self.sg])

        if is_random:
            self.wyckoffs = self.combination_wp_random(wyckoff_position, atom_num, is_shuffle=True)
        else:
            # self.wyckoffs = self.combination_wp_all(wyckoff_position, atom_num)
            # if len(self.wyckoffs) != 0 and self.wyckoffs[0] is None:
            #     self.wyckoffs = self.combination_wp_all(wyckoff_position, atom_num, shuffle=False)
            self.wyckoffs = self.combination_wp_all(wyckoff_position, atom_num, shuffle=False)

    def combination_wp_all(self,
                           wyckoff_position: list,
                           atom_num: list,
                           is_fast: bool = True,
                           max_count: int = 100e4,
                           # max_count: int = 10000,
                           max_trial: int = 1e8,
                           # shuffle: bool = True,
                           shuffle: bool = False
                           ):
        wp_part_list = []
        for an in atom_num:
            part_wp_an_path = os.path.join(self.save_path, 'part_wp', str(an) + '_' + str(self.sg))
            if os.path.isfile(part_wp_an_path):
                wp_part = read_data_bin(None, part_wp_an_path)
            else:
                wp_part = self.combination_wp_part(wyckoff_position, an)
                save_data_bin(None, part_wp_an_path, data=wp_part)
            if not wp_part:
                return []

            # make the wp_parts that do not contain static positions to be in the front of the list
            wp_part = sorted(wp_part, key=custom_sort_key)

            wp_part_list.append(wp_part)

        is_use_fast = False
        if is_fast:
            wp_part_len_list = [len(wpp) for wpp in wp_part_list]
            wp_all_count = functools.reduce(lambda x, y: x * y, wp_part_len_list)
            if wp_all_count > max_count:
                if shuffle:
                    wp_part_list_tmp = []
                    for wpp in wp_part_list:
                        random.shuffle(wpp)
                        # wpp = weighted_shuffle(wpp, weights=list(range(1, len(wpp) + 1))[::-1])
                        wp_part_list_tmp.append(wpp)
                    wp_part_list = wp_part_list_tmp
                is_use_fast = True
            # print(wp_all_count)

        wp_all_list = []
        wp_product = itertools.product(*wp_part_list)
        num_trial = 0
        for p in wp_product:
            num_trial += 1
            pp = [set(k for j in i for k in j) for i in p]

            # res = list(pp[0].intersection(*pp[1:]))
            # original implementation, getting intersection of wp combinations for all elements
            # however, what we need is the union of all pair-wise intersections
            pp_pairs = itertools.combinations(pp, 2)
            intersections = [set(a).intersection(b) for a, b in pp_pairs]
            res = set().union(*intersections)

            for ri in res:
                if not ('x' in ri or 'y' in ri or 'z' in ri):
                    p = []
                    break
            if p:
                wp_all_list.append(p)
                num_trial = 0
                if is_use_fast and len(wp_all_list) >= max_count:
                    return wp_all_list

            if (num_trial % 1000000 == 0) and (num_trial != 0):
                logger.info(f'Finish {num_trial} trials, current number of valid wp {len(wp_all_list)}')

            if (num_trial >= max_trial) & (max_trial > 0):
                # if len(wp_all_list) > int(max_count * 0.01):
                #     return wp_all_list
                # else:
                #     return [None]  # if generated number with shuffling is small, try restarting from general WPs
                return wp_all_list

        return wp_all_list

    def combination_wp_random(self,
                              wyckoff_position: list,
                              atom_num: list,
                              max_count: int = 1,
                              is_shuffle: bool = True):
        """
        max_count only one now
        :param wyckoff_position:
        :param atom_num:
        :param max_count:
        :param is_shuffle:
        :return:
        """

        if max_count and is_shuffle:
            random.shuffle(wyckoff_position)

        wp_part_list = []
        for an in atom_num:
            # print(len(wyckoff_position))
            wp_part = self.combination_wp_part(wyckoff_position, an, max_count=max_count)
            # wp_part = wp_part[0]

            if not wp_part:
                return []

            wp_part_list.append(wp_part)

            for wpp in wp_part:
                for wp in wpp:
                    if not ('x' in '_'.join(wp) or 'y' in '_'.join(wp) or 'z' in '_'.join(wp)):
                        wyckoff_position.remove(wp)

        wp_all_list = list(itertools.product(*wp_part_list))

        return wp_all_list

    @staticmethod
    def combination_wp_part(wyckoff_position: list,
                            atom_num_part: int,
                            max_count: int = 100000) -> list:
        # max_count: int = -1) -> list:
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
                if wp[i] in temp and not ('x' in ','.join(wp[i]) or 'y' in ','.join(wp[i]) or 'z' in ','.join(wp[i])):
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
        wp_num = [len(i) for i in wp]
        try:
            dfs(atom_num_part, _index, _temp, _temp_num)
        except:
            pass
        # print(result)

        return result


@print_run_info("Get the Wyckoff position combinations")
def get_all_wyckoff_combination(sg_list, atom_num):
    current_path = os.path.split(__file__)[0]

    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z']
    wyckoff_combination_type = ''.join([alphabet[i] + str(atom_num[i]) for i in range(len(atom_num))])

    wyckoffs_dict = {}
    max_wyckoffs_count = 0
    for sg_i in sg_list:
        sg_i_path = os.path.join(current_path, 'wp_sg', wyckoff_combination_type + '_' + str(sg_i))
        if os.path.isfile(sg_i_path):
            wp = read_data_bin(None, sg_i_path)
        else:
            logger.info(f"Starting the space group {sg_i}")
            wp = GetWyckoffPosition(sg_i, atom_num, is_random=False, verbose=False, save_path=current_path).wyckoffs
            save_data_bin(None, sg_i_path, data=wp)
        if None in wp:
            wp = []
        wyckoffs_dict[sg_i] = wp
        max_wyckoffs_count = max(len(wp), max_wyckoffs_count)
        logger.info(f"Get spg {sg_i} with {len(wp)} wps")

    return wyckoffs_dict, max_wyckoffs_count


if __name__ == '__main__':
    # _sg_list = list(range(2, 230+1))
    _sg_list = [10]
    # _atom_num = [2, 17]
    # _atom_num = [6, 51]
    # _atom_num = [4, 34]
    # _atom_num = [12, 8, 12, 48]
    _atom_num = [24, 16, 24, 96]
    # _atom_num = [12, 102]
    # _atom_num = [4, 4, 8]
    _a, _b = get_all_wyckoff_combination(sg_list=_sg_list, atom_num=_atom_num)
    _wyckoff_df = pd.read_csv(os.path.split(__file__)[0] + "/wyckoff_list.csv")
