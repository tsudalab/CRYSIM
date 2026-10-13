import pickle
import numpy as np


results_root_dir = '../BQM4CSP-results/'


def get_init_set():
    spg_list = {spg: [] for spg in range(2, 231)}
    for file in [
        'B28_M3GNet_num1000-s0.pkl',
        'B28_M3GNet_num1000-s1.pkl',
        'B28_M3GNet_num1000-s2.pkl',
        'B28_M3GNet_num1000-s3.pkl',
    ]:
        with open(f'../dataset/{file}', 'rb') as f:
            d = pickle.load(f)
            for i, s in enumerate(d['spg_gen']):
                spg_list[s].append(d['struct_e_list'][i])
    return spg_list


def test_training_set_decode(pres_list, spg_list, dim, training_set_dir):
    from crysim.src.onehot.oh_cs_spg import decode_cs_based_on_spg, decode_spg_for_cs, calculate_spg_precision
    from crysim.src.spg.get_wyckoff_position import get_all_wyckoff_combination
    from crysim.src.spg.spg_utils import get_spg_list
    from crysim.src.struct_utils import get_cs_list
    lat_len, lat_ang, num_cs, num_spg, wpc, coord = pres_list
    # wyckoffs_dict, wyckoffs_max = get_all_wyckoff_combination(
    #     sg_list=list(range(2, 231)), atom_num=[24, 16, 24, 96])
    # spg_list = get_spg_list(wyckoffs_dict)
    cs_list = get_cs_list(spg_list, dim=dim)
    spg_e_list = []
    with open(f'{training_set_dir}/training_set.pkl', 'rb') as f:
        data = pickle.load(f)
    for i in range(len(data['X'])):
        num_ang = 3 if dim == 3 else 1
        sym_vec = data['X'][i][lat_len * 3 + lat_ang * num_ang: lat_len * 3 + lat_ang * num_ang + num_cs + num_spg]
        cs, spg = np.where(sym_vec)[0]
        cs = decode_cs_based_on_spg(cs, cs_list)
        spg = decode_spg_for_cs(spg, cs, spg_list, spg_precision=calculate_spg_precision(spg_list, dim), dim=dim)
        spg_e_list.append((spg, data['y'][i]))
        print(f'obtain {spg}, {data["y"][i]}')
    return spg_e_list


def generate_from_spg(system_name, dim):
    from crysim.src.optimizer.rand_gen import random_generate
    from crysim.src.struct_utils import get_species_list
    species = get_species_list(system_name)
    struct_list = random_generate(dim, species, nstruc=50, id_offset=0, init_pos_path=None, seed=0,
                                  filter_dist=True, logger=None, random=False)
    return struct_list


def _exe_generated_struct_file(file_name):
    from crysim.src.struct_utils import analyze_pyx_struct
    import bz2
    with bz2.open('data_' + file_name, 'rb') as f:
        struct_list = pickle.load(f)
    pmg_struct_list, used_atoms_list, spg_list, wp_list = [], [], [], []
    for struc in struct_list:
        struc, wp_gen = struc
        pmg_struct, used_atoms, spg_gen, _ = analyze_pyx_struct(struc)
        pmg_struct_list.append(pmg_struct)
        used_atoms_list.append(used_atoms)
        spg_list.append(spg_gen)
        wp_list.append(wp_gen)
    with bz2.open(file_name, 'wb') as f:
        pickle.dump((pmg_struct_list, used_atoms_list, spg_list, wp_list), f)
    return


def _exe_gen_wpc():
    from crysim.src.spg.get_wyckoff_position import get_all_wyckoff_combination
    from crysim.src.spg.spg_utils import sg_list
    import argparse
    parser = argparse.ArgumentParser(description='gen')
    parser.add_argument('-n', '--num_species_atom', type=int, action='append')
    parser.add_argument('-m', '--mode', type=str)
    parser.add_argument('-d', '--dim', type=int)
    args = parser.parse_args()
    wyckoffs_dict, wyckoffs_max = get_all_wyckoff_combination(
        sg_list=sg_list(args.dim), atom_num=args.num_species_atom, mode=args.mode, dim=args.dim)
    return


def _test_wpc(num_species_atom):
    from crysim.src.spg.get_wyckoff_position import get_all_wyckoff_combination
    from crysim.src.spg.spg_utils import dof_compute
    wyckoffs_dict, _ = get_all_wyckoff_combination(
        sg_list=list(range(2, 231)), atom_num=num_species_atom, mode='gn-oa')
    wyckoffs_dict_a, _ = get_all_wyckoff_combination(
        sg_list=list(range(2, 231)), atom_num=num_species_atom, mode='pyx')
    dof_dict = {}
    for k in wyckoffs_dict.keys():
        dof, dof_a = [], []
        wpc_list = wyckoffs_dict[k]
        wpc_list_a = wyckoffs_dict_a[k]
        for i in range(len(wpc_list)):
            wpc, wpc_a = wpc_list[i], wpc_list_a[i]
            for j in range(len(wpc)):
                wpc_ele, wpc_ele_a = wpc[j], wpc_a[j]
                wpc_dof, wpc_mul, wpc_dof_a, wpc_mul_a = [], [], [], []
                for ll in range(len(wpc_ele)):
                    wp_str, wp = wpc_ele[ll], wpc_ele_a[ll]
                    wpc_dof.append(dof_compute(wp_str))
                    wpc_mul.append(len(wp_str))
                    wpc_dof_a.append(wp[-1])
                    wpc_mul_a.append(wp[0])
                dof += sorted(wpc_dof)
                dof += sorted(wpc_mul)
                dof_a += sorted(wpc_dof_a)
                dof_a += sorted(wpc_mul_a)
        print(k, all(np.array(dof) == np.array(dof_a)))
        dof_dict[k] = [np.array(dof), np.array(dof_a)]
    return wyckoffs_dict, wyckoffs_dict_a, dof_dict


def _test_wpc_num():
    from crysim.src.spg.get_wyckoff_position import get_all_wyckoff_combination
    from crysim.src.spg.spg_utils import get_spg_list, get_spg_list2, sg_list
    import argparse
    parser = argparse.ArgumentParser(description='gen')
    parser.add_argument('-n', '--num_species_atom', type=int, action='append')
    parser.add_argument('-d', '--dim', type=int)
    args = parser.parse_args()
    wyckoffs_dict_a, _ = get_all_wyckoff_combination(
        sg_list=sg_list(args.dim), atom_num=args.num_species_atom, mode='pyx', dim=args.dim)
    spg_list = set(get_spg_list(wyckoffs_dict_a))
    spg_list_pyx = set(get_spg_list2(args.num_species_atom, dim=args.dim))
    print(len(spg_list), len(spg_list_pyx))
    print(spg_list_pyx - spg_list)
    return


def _reverse_wpc(log_file):
    from crysim.src.spg.get_wyckoff_position import save_data_bin, get_all_wyckoff_combination
    import re
    from collections import defaultdict
    import os
    from pathlib import Path

    def find_reversed_spg(log_file_path):
        # 定义正则表达式模式
        start_group_pattern = re.compile(r'Starting the space group (\d+)')
        solutions_pattern = re.compile(r'Start with (\d+) solutions, (\d+) total trials')

        # 存储结果
        results = defaultdict(list)
        current_group = None

        with open(log_file_path, 'r') as file:
            for line in file:
                # 检查是否是"Starting the space group"行
                group_match = start_group_pattern.search(line)
                if group_match:
                    current_group = int(group_match.group(1))
                    continue

                # 检查是否是"Start with X solutions, Y total trials"行
                solutions_match = solutions_pattern.search(line)
                if solutions_match and current_group is not None:
                    a = int(solutions_match.group(1))
                    b = int(solutions_match.group(2))
                    results[current_group].append((a, b))

        return results

    space_group_data = find_reversed_spg(log_file)
    sg_list = list(space_group_data.keys())
    for mode in ['gn-oa', 'pyx']:
        wpc_dict, _ = get_all_wyckoff_combination(sg_list, [24, 16, 24, 96], mode=mode)
        for sg in sg_list:
            reversed_len = int(len(wpc_dict[sg]) / 2)
            print(f'{reversed_len} are found for {sg}')
            reversed_part = wpc_dict[sg][reversed_len:]
            new_wpc = wpc_dict[sg][:reversed_len] + reversed_part[::-1]
            print(f"{len(new_wpc)} is saved for {sg}")
            wpc_bin_name = 'wp_sg' if mode == 'gn-oa' else 'wp_sg-alphabet'
            Path(wpc_bin_name).mkdir(exist_ok=True)
            sg_i_path = os.path.join('.', wpc_bin_name, 'A24B16C24D96' + '_' + str(sg))
            save_data_bin(None, sg_i_path, new_wpc)
    return


def _exe_pyx_uni_gen():
    import argparse
    from crysim.src.optimizer.rand_gen import RandGenDirector
    from crysim.src.registry import OptimizerName
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-sys', '--system', type=str)
    parser.add_argument('-n', '--n-struct', type=int)
    parser.add_argument('-spg', '--spg', type=int)
    parser.add_argument('-s', '--seed', type=int, action='append')
    parser.add_argument('-d', '--dim', type=int)
    args = parser.parse_args()
    system_name = args.system
    n_struct = args.n_struct
    spg = args.spg
    seed = args.seed
    for s in seed:
        rg = RandGenDirector(dim=args.dim, system_name=system_name, gen_tool=OptimizerName.wpc_l_rs_uni()[0],
                             n_struct=n_struct, spg_list=[spg], seed=s)
        rg()
    return


def _exe_pyx_gen():
    import argparse
    from crysim.src.optimizer.rand_gen import RandGenDirector
    from crysim.src.registry import OptimizerName
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-sys', '--system', type=str)
    parser.add_argument('-n', '--n-struct', type=int)
    parser.add_argument('-s', '--seed', type=int, action='append')
    parser.add_argument('-d', '--dim', type=int)
    args = parser.parse_args()
    system_name = args.system
    n_struct = args.n_struct
    seed = args.seed
    for s in seed:
        rg = RandGenDirector(dim=args.dim, system_name=system_name, 
                             gen_tool=OptimizerName.wpc_l_rs()[0],
                            #  gen_tool=OptimizerName.pyxtal_rs()[0],
                             filter_dist=False, len_lbs=None, len_ubs=None, ang_lbs=None, ang_ubs=None,
                             n_struct=n_struct, seed=s)
        rg()
    return


def _exe_dataset_e_eval():
    import argparse
    from crysim.src.optimizer.rand_gen import RandGenDirector, RGEnergyDirector
    from crysim.src.registry import OptimizerName, EnergyEvaluatorName, rlx_type
    from crysim.src.evaluator.loader import load_energy_evaluator
    from crysim.src.evaluator.e_eval import _steps
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('-sys', '--system', type=str)
    parser.add_argument('-n', '--n-struct', type=int)
    parser.add_argument('-s', '--seed', type=int, action='append')
    parser.add_argument('-d', '--dim', type=int)
    parser.add_argument('-r', '--rlx-steps', type=int, default=_steps)
    args = parser.parse_args()
    system_name = args.system
    n_struct = args.n_struct
    seed = args.seed
    rlx_steps = args.rlx_steps
    e_cal = load_energy_evaluator(EnergyEvaluatorName.orb(), dim=args.dim)
    for s in seed:
        rg = RandGenDirector(dim=args.dim, system_name=system_name, 
                             gen_tool=OptimizerName.wpc_l_rs()[0],
                            #  gen_tool=OptimizerName.pyxtal_rs()[0],
                             filter_dist=False, len_lbs=None, len_ubs=None, ang_lbs=None, ang_ubs=None,
                             n_struct=n_struct, seed=s, init_spg_list=None, logger=None, load_cache=True)
        rg()
        rge = RGEnergyDirector(dim=args.dim, system_name=system_name, gen_tool=OptimizerName.wpc_l_rs()[0],
                               len_lbs=None, len_ubs=None, ang_lbs=None, ang_ubs=None, use_rlx_type=rlx_type[1],
                               n_struct=n_struct, seed=s, e_cal=e_cal, init_spg_list=None, rlx_steps=rlx_steps,
                               logger=None, load_cache=True)
        rge()
    return


if __name__ == '__main__':
    # test_training_set_decode(
    #     [90, 40, 7, 68, 300, 1728],
    #     [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    #      30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    #      56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
    #      82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
    #      107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    #      128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 147, 149, 150, 156, 157,
    #      158, 159, 162, 163, 164, 165, 168, 173, 174, 175, 176, 177, 182, 183, 184, 185, 186, 187, 188, 189, 190,
    #      191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 204, 205, 206, 207, 208, 209, 211, 212, 213,
    #      214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 229, 230],
    #     'results/ca24al16si24o96/filter/2024-12-14-16-20-45_oh-12-None-100-CSSPG-opt_all_2-fm16_filtering_amp_c-tNone_M3GNet_Ca24Al16Si24O96_s0_n300'
    # )

    # _a = test_training_set_decode(
    #     [112, 40, 7, 66, 300, 3375],
    #     # [127, 40, 7, 66, 300, 3375],
    #     [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    #      32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    #      61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
    #      91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115,
    #      116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137,
    #      138, 139, 140, 141, 143, 147, 149, 150, 156, 157, 158, 159, 162, 163, 164, 165, 168, 173, 174, 175, 176, 177,
    #      182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 204,
    #      205, 207, 208, 209, 211, 212, 213, 214, 215, 216, 217, 218, 220, 221, 222, 223, 224, 225, 229],
    #     results_root_dir + 'intern/results/benchmarks/2025-07-10-17-25-59_oh-15-None-100-CSSPG_2-fm16_filtering_grb_c-tNone_M3GNet_B28_s0_n300_64cpu-428GB'
    # )

    # _a = generate_from_spg2('B28')

    # _exe_generated_struct_file('B28-pyx-wpc-uni_num50-s0.pkl.bz2')

    # _exe_gen_wpc('gn-oa')
    # _exe_gen_wpc('pyx')
    # _exe_gen_wpc()
    # _a, _b, _c = _test_wpc([6, 51])

    # _reverse_wpc('src/spg/wpclog-gn-oa_24-16-24-96.log')

    # _exe_pyx_uni_gen()
    # _exe_pyx_gen()
    _exe_dataset_e_eval()

    # _test_wpc_num()
