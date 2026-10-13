import numpy as np
from logzero import logfile
from tqdm import tqdm
from copy import deepcopy
from functools import partial
from pyxtal import pyxtal

from crysim.src import DATASET_ROOT, DEBUG
from crysim.src.spg.get_wyckoff_position import get_all_wyckoff_combination
from crysim.src.spg.spg_utils import Random2Wyckoff, get_spg_list2, analyze_pyx_struct
from crysim.src.struct_utils import unique_elements, DistFilter, get_system_name, \
    get_species_list, get_savable_structures, get_system_size, _mid_scale_lb, _mid_scale_ub, \
    LatticeLengthController, default_ang_lb, default_ang_ub, dim2pbc, add_vacuum_to_pmg, _vacuum_thickness
from crysim.src.evaluator.e_eval import average_force, _steps
from crysim.src.registry import OptimizerName, FileLoader, init_set_list, EnergyEvaluatorName, rlx_type, alpha
from logzero import logger as log_printer

from crysim.src.registry import mixing


def arrange_lat_bounds(input_lbs, input_ubs, default_lb, default_ub):
    input_lbs = np.array([input_lbs, input_lbs, input_lbs]) if np.array([input_lbs]).ndim == 1 else np.array(input_lbs)
    input_ubs = np.array([input_ubs, input_ubs, input_ubs]) if np.array([input_ubs]).ndim == 1 else np.array(input_ubs)
    input_lbs = np.where(input_lbs == None, default_lb, input_lbs).astype(float)
    input_ubs = np.where(input_ubs == None, default_ub, input_ubs).astype(float)
    return input_lbs, input_ubs
    

def get_lat_bounds(system_size, len_lbs, len_ubs, ang_lbs, ang_ubs):
    llc = LatticeLengthController(system_size)
    default_len_lb, default_len_ub = llc.min(), llc.max()
    len_lbs, len_ubs = arrange_lat_bounds(len_lbs, len_ubs, default_len_lb, default_len_ub)
    ang_lbs, ang_ubs = arrange_lat_bounds(ang_lbs, ang_ubs, default_ang_lb, default_ang_ub)
    return len_lbs, len_ubs, ang_lbs, ang_ubs


class RandomGenerator:
    def __init__(self, dim, wyckoffs_dict, species_list, spg_num, wp_num,
                 len_lbs, len_ubs, ang_lbs, ang_ubs, add_vacuum_to_pmg_fn=None):
        self.dim = dim
        self.species_list = species_list
        self.spg_num = spg_num
        self.wp_num = wp_num
        self.add_vacuum_to_pmg_fn = add_vacuum_to_pmg_fn
        self.elements, self.num_species_atom = unique_elements(species_list)
        self.wyckoffs_dict = wyckoffs_dict
        self.len_lbs, self.len_ubs, self.ang_lbs, self.ang_ubs = \
            get_lat_bounds(len(self.species_list), len_lbs, len_ubs, ang_lbs, ang_ubs)
        return

    def _generate(self):
        a, b, c = np.random.uniform(self.len_lbs, self.len_ubs)
        alpha, beta, gamma = np.random.uniform(self.ang_lbs, self.ang_ubs)
        pos = np.random.uniform(0, 1, size=(len(self.species_list), 3))
        return (a, b, c, alpha, beta, gamma), pos

    def generate(self):
        lat_para, pos = self._generate()
        r2w = Random2Wyckoff(dim=self.dim, wyckoffs_dict=self.wyckoffs_dict,
                             species_list=self.species_list, lat_para=lat_para, pos=pos, spg=self.spg_num, 
                             wp=self.wp_num, add_vacuum_to_pmg_fn=self.add_vacuum_to_pmg_fn)
        pmg_struct = r2w.get_pmg_struct()
        return pmg_struct, r2w.used_decode


def out_poscar(struc, cid, fpath, spg_num, spg_sym):
    # ---------- poscar format
    pos = struc.to(fmt='poscar')
    pos = pos.split('\n')
    blank_indx = pos.index('')  # cut unnecessary parts
    pos = pos[:blank_indx]
    pos[0] = 'ID_{}, {}, {}'.format(cid, spg_num, spg_sym)  # replace with ID
    lines = [line + '\n' for line in pos]

    # ---------- append POSCAR
    with open(fpath, 'a+') as f:
        for line in lines:
            f.write(line)


def get_struct_file_name(system_name, dim, len_lbs, len_ubs, ang_lbs, ang_ubs, 
                         gen_tool, n_struct, seed, init_spg_list, custom_name):
    lat_scl = f"{''.join([str(np.round(lb, 1)) for lb in len_lbs])}-{''.join([str(np.round(ub, 1)) for ub in len_ubs])}"
    lat_scl += f"-{''.join([str(ang) for ang in ang_lbs])}-{''.join([str(ang) for ang in ang_ubs])}"
    base_name = f'{system_name}-{dim}_{lat_scl}-{gen_tool}_num{n_struct}-s{seed}'.replace('.', 'p')
    if custom_name is not None:
        base_name += f'_{custom_name}'
    if init_spg_list is None:
        return f'{base_name}'
    else:
        spg = '-'.join([str(s) for s in init_spg_list])
        spg = spg[:20]
        return f'{base_name}_{spg}'


def get_energy_file_name(system_name, dim, len_lbs, len_ubs, ang_lbs, ang_ubs, 
                         gen_tool, use_rlx_type, n_struct, seed, e_cal_name, init_spg_list, custom_name, rlx_steps=_steps):
    struct_file = get_struct_file_name(system_name=system_name, dim=dim, gen_tool=gen_tool, 
                                       len_lbs=len_lbs, len_ubs=len_ubs, ang_lbs=ang_lbs, ang_ubs=ang_ubs, 
                                       n_struct=n_struct, seed=seed, init_spg_list=init_spg_list, custom_name=custom_name)
    base_name = struct_file + f'_{e_cal_name}'
    if use_rlx_type != rlx_type[0]:
        base_name += f'-{use_rlx_type}'
        if use_rlx_type in [rlx_type[1], rlx_type[2]] and rlx_steps != _steps:
            base_name += f'{rlx_steps}'
    return base_name


class RandGenFileManager(FileLoader):
    def __init__(self, dim, system_name, len_lbs, len_ubs, ang_lbs, ang_ubs, n_struct, gen_tool, seed, 
                 file_name, struct_file_root, load_cache, zip_file, init_spg_list, logger, custom_name):
        self.dim = dim
        self.seed = seed
        self.n_struct = n_struct
        self.gen_tool = gen_tool
        self.system_name = system_name
        self.species_list = get_species_list(sys_name=system_name)
        self.elements, self.num_species_atom = unique_elements(self.species_list)
        self.len_lbs, self.len_ubs, self.ang_lbs, self.ang_ubs = len_lbs, len_ubs, ang_lbs, ang_ubs
        
        struct_file_name = get_struct_file_name(system_name=system_name, dim=dim, 
                                                len_lbs=self.len_lbs, len_ubs=self.len_ubs, 
                                                ang_lbs=self.ang_lbs, ang_ubs=self.ang_ubs, 
                                                gen_tool=gen_tool, n_struct=n_struct, 
                                                seed=seed, init_spg_list=init_spg_list, custom_name=custom_name)
        if logger is None:
            logfile(f'{struct_file_root}/{struct_file_name}.log')
        if file_name is None:
            file_name = f'{struct_file_name}.pkl'
        
        self.custom_name = custom_name
        self.logger = logger
        self.file_name = file_name
        self.struct_file_root = struct_file_root
        super().__init__(file_dire=struct_file_root + '/' + file_name, load_cache=load_cache, zip_file=zip_file)
        self._process_loaded_gen()
        return

    def _process_loaded_gen(self):
        # load generated file into init lists for further processing
        raise NotImplementedError

    def _preprocess_for_gen(self):
        # preprocess before generation if the loaded dataset is still incomplete
        raise NotImplementedError

    def save_gen(self, *args):
        # save new data into lists and save them to the disk
        raise NotImplementedError

    def save_process(self, curr_struct):
        if curr_struct % 1 == 0:
            self.save()
            return
        if curr_struct + 1 == self.n_struct:
            self.save()
            return


class RandGen(RandGenFileManager):
    def __init__(self, dim, system_name, filter_dist, len_lbs, len_ubs, ang_lbs, ang_ubs, n_struct, gen_tool, seed, struct_file_root,
                 load_cache, zip_file, logger, init_spg_list, filtering_fn, custom_name):
        super().__init__(dim=dim, system_name=system_name, 
                         len_lbs=len_lbs, len_ubs=len_ubs, ang_lbs=ang_lbs, ang_ubs=ang_ubs, 
                         n_struct=n_struct, gen_tool=gen_tool, seed=seed,
                         file_name=None, struct_file_root=struct_file_root, load_cache=load_cache,
                         init_spg_list=init_spg_list, zip_file=zip_file, logger=logger, custom_name=custom_name)
        self.compat_sg_list = get_spg_list2(self.num_species_atom, dim=dim, include_spg_list=init_spg_list)
        self.min_mid_scale, self.max_mid_scale = _mid_scale_lb, _mid_scale_ub
        self.filter_dist = filter_dist
        self.random = False if OptimizerName.uniform_gen()[0] in self.gen_tool else True
        self.filtering_fn = filtering_fn if filtering_fn is not None else lambda x: True
        consider_pbc_in_unit = False if custom_name == mixing else True
        self.add_vacuum_to_pmg_fn = partial(add_vacuum_to_pmg, consider_pbc_in_unit=consider_pbc_in_unit)
        return
    
    def filter_lattice(self, pmg_struct, after_add_vac):
        lat = pmg_struct.lattice
        lat_len, lat_ang = np.array(lat.abc), np.array(lat.angles)
        pbc = dim2pbc(self.dim)
        lat_len -= after_add_vac * np.abs(np.array(pbc) - 1) * _vacuum_thickness
        if not (np.all(self.len_lbs < lat_len) and np.all(lat_len < self.len_ubs) and \
                np.all(self.ang_lbs < lat_ang) and np.all(lat_ang < self.ang_ubs)):
            return False
        return True

    def _process_loaded_gen(self):
        # after loading
        if not self.load_flag:
            # init
            self.pmg_struct_list, self.used_atoms_list, self.spg_list, self.wp_list = [], [], [], []
            self.file = (self.pmg_struct_list, self.used_atoms_list, self.spg_list, self.wp_list)
        else:
            # receive
            self.pmg_struct_list, self.used_atoms_list, self.spg_list, self.wp_list = self.file
        return

    def save_gen(self, pmg_struct, used_atoms, spg_gen, wp_gen):
        self.pmg_struct_list.append(pmg_struct)
        self.used_atoms_list.append(used_atoms)
        self.spg_list.append(spg_gen)
        self.wp_list.append(wp_gen)
        self.file = (self.pmg_struct_list, self.used_atoms_list, self.spg_list, self.wp_list)
        return

    def __call__(self, *args, **kwargs):
        current_size = len(self.pmg_struct_list)
        self.n_struct_0 = deepcopy(self.n_struct)
        if current_size == 0:
            self._preprocess_for_gen()
            self.generate()
            self.save()
        else:
            if OptimizerName.uniform_gen()[0] not in self.gen_tool:
                if current_size < self.n_struct:
                    self._preprocess_for_gen()
                    self.n_struct = self.n_struct - current_size
                    self.seed += current_size
                    self.pmg_struct_list, self.used_atoms_list, self.spg_list, self.wp_list = self.file
                    log_printer.info(f"Load {current_size} generated samples, "
                                     f"continue to generate remaining {self.n_struct} "
                                     f"with seed {self.seed} in the same file")
                    self.generate()
                    self.save()
            else:
                total_num = self.n_struct * len(self.compat_sg_list)
                if current_size < total_num:
                    self._preprocess_for_gen()
                    self.compat_sg_list_0 = deepcopy(self.compat_sg_list)
                    current_spg_idx, current_spg_size = current_size // self.n_struct_0, current_size % self.n_struct
                    if current_spg_size != 0:
                        # finish this SPG
                        self.n_struct = self.n_struct_0 - current_spg_size
                        self.seed += current_size
                        current_spg = self.compat_sg_list_0[current_spg_idx]
                        self.compat_sg_list = [current_spg]
                        self.pmg_struct_list, self.used_atoms_list, self.spg_list, self.wp_list = self.file
                        log_printer.info(f"Load {current_size} generated samples, "
                                         f"continue generate {self.n_struct} for SPG {current_spg} "
                                         f"with seed {self.seed} in the same file")
                        self.generate()
                        self.save()
                        current_spg_idx = current_spg_idx + 1
                    remained_spg = self.compat_sg_list_0[current_spg_idx:]
                    self.n_struct = self.n_struct_0
                    self.seed += current_size
                    self.compat_sg_list = remained_spg
                    self.pmg_struct_list, self.used_atoms_list, self.spg_list, self.wp_list = self.file
                    log_printer.info(f"Load {current_size} generated samples, "
                                     f"continue generate {self.n_struct} for SPG starting from {self.compat_sg_list[0]}"
                                     f" with seed {self.seed} in the same file")
                    self.generate()
                    self.save()
        return self.file


class SimpleRandGen(RandGen):
    def __init__(self, system_name, len_lbs, len_ubs, ang_lbs, ang_ubs, 
                 gen_tool, n_struct, id_offset, init_pos_path, filter_dist, dim, seed, root,
                 load_cache, logger, init_spg_list, filtering_fn, custom_name):
        assert gen_tool in [OptimizerName.random_search()[0], OptimizerName.random_search_uni()[0]]
        super().__init__(dim=dim, system_name=system_name, filter_dist=filter_dist, n_struct=n_struct, gen_tool=gen_tool, 
                         len_lbs=len_lbs, len_ubs=len_ubs, ang_lbs=ang_lbs, ang_ubs=ang_ubs, 
                         seed=seed, struct_file_root=root, load_cache=load_cache, zip_file=False, 
                         logger=logger, init_spg_list=init_spg_list, filtering_fn=filtering_fn, custom_name=custom_name)
        self.id_offset = id_offset
        self.init_pos_path = init_pos_path
        return

    def _preprocess_for_gen(self):
        self.wyckoffs_dict, self.wyckoffs_max = \
            get_all_wyckoff_combination(sg_list=self.compat_sg_list, atom_num=self.num_species_atom, mode='pyx', dim=self.dim)
        return

    def generate(self):
        np.random.seed(self.seed)
        i, j = 0, 0
        if self.random:
            while i < self.n_struct:
                spg_num = np.random.choice(self.compat_sg_list)
                if len(self.wpc_dict[spg_num]) == 0:
                    continue
                wpc_idx = np.random.choice(range(len(self.wpc_dict[spg_num])))
                j, i = self.generate_one_physical_struct(spg_num, wpc_idx, j, i)
                self.save_process(i)
        else:
            max_try = 1e6
            for spg_num in self.compat_sg_list:
                if len(self.wpc_dict[spg_num]) == 0:
                    continue
                while i < self.n_struct:
                    wpc_idx = np.random.choice(range(len(self.wpc_dict[spg_num])))
                    j, i = self.generate_one_physical_struct(spg_num, wpc_idx, j, i)
                    self.save_process(i)
                    if j >= max_try:
                        log_printer.info(f"Fail on SPG {spg_num} after {max_try} trials")
                        self.save()
                        break
                i = 0
                j = 0
        print(f'Generation success rate: {i / j:.4}')
        return self.pmg_struct_list, self.used_atoms_list, self.spg_list, self.wp_list

    def generate_one_physical_struct(self, spg_num, wyckoff_num, j, i):
        rg = RandomGenerator(dim=self.dim, wyckoffs_dict=self.wyckoffs_dict,
                             species_list=self.species_list, spg_num=spg_num, wp_num=wyckoff_num,
                             len_lb=self.len_lbs, len_ub=self.len_ubs, ang_lb=self.ang_lbs, ang_ub=self.ang_ubs)
        pmg_struct, used_decode = rg.generate()
        if (self.filter_dist and \
            not DistFilter.test(pmg_struct, test_lattice=False, 
                                min_scale=self.min_mid_scale, max_scale=self.max_mid_scale, pbc=dim2pbc(self.dim))) or \
           not self.filtering_fn(pmg_struct) or \
           not self.filter_lattice(pmg_struct, after_add_vac=True):
            log_printer.info(f'structure generation fail on {i}-th material after {j} generation trials')
            j += 1
            return j, i
        log_printer.info(f'success on {i}-th material generation')
        if self.logger is not None:
            self.logger.record_anything1(operation='Space Group Random Selection', result=spg_num)
            self.logger.record_anything2(operation='Wyckoff Positions Random Selection', result=wyckoff_num)
        self.save_gen(pmg_struct=pmg_struct, used_atoms=used_decode, spg_gen=spg_num, wp_gen=wyckoff_num)
        cid = len(self.pmg_struct_list) + self.id_offset
        if self.init_pos_path is not None:
            out_poscar(pmg_struct, cid, self.init_pos_path, spg_num, 'spg_sym')
        i += 1
        j += 1
        return j, i


class RandGenPYX(RandGen):
    def __init__(self, dim, system_name, filter_dist, len_lbs, len_ubs, ang_lbs, ang_ubs, 
                 n_struct, gen_tool, seed, root, load_cache, zip_file, logger, init_spg_list,
                 filtering_fn, custom_name):
        super().__init__(dim=dim, system_name=system_name, filter_dist=filter_dist, n_struct=n_struct, gen_tool=gen_tool, 
                         len_lbs=len_lbs, len_ubs=len_ubs, ang_lbs=ang_lbs, ang_ubs=ang_ubs, 
                         seed=seed, struct_file_root=root, load_cache=load_cache, zip_file=zip_file, logger=logger,
                         init_spg_list=init_spg_list, filtering_fn=filtering_fn, custom_name=custom_name)
        self.llc = LatticeLengthController(num_atom=get_system_size(sys_name=system_name))
        return

    def generate_one(self, sg_num):
        raise NotImplementedError

    def _decode(self, struc):
        struc, wp_gen = struc
        if wp_gen is not None:
            pmg_struct, used_atoms, spg_gen, _ = analyze_pyx_struct(struc, dim=self.dim)
            # wp_gen = {ele: wp for ele, wp in zip(self.elements, wp_gen)}
        else:
            pmg_struct, used_atoms, spg_gen, wp_gen = analyze_pyx_struct(struc, dim=self.dim)
        #     wp_gen = [wp_gen[ele] for ele in self.elements]
        # if isinstance(used_atoms, dict):
        #     used_atoms = [used_atoms[ele] for ele in self.elements]
        return pmg_struct, used_atoms, spg_gen, wp_gen

    def generate(self):
        i, total_try = 0, 0
        np.random.seed(self.seed)

        if self.random:
            max_try = max(int(self.n_struct / 10), 200000)
            for i in tqdm(range(self.n_struct)):
                success = False
                num_try = 0
                while not success and num_try < max_try:
                    if (num_try % 10 == 0) and (num_try != 0):
                        log_printer.info(f"Currently {num_try + 1}-th trial")
                    try:
                        sg_num = np.random.choice(self.compat_sg_list, size=1)[0]
                        struc, wp_gen = self.generate_one(sg_num)
                        if struc is None:
                            num_try += 1
                            total_try += 1
                            continue
                        pmg_struct, used_atoms, spg_gen, wp_gen = self._decode((struc, wp_gen))
                        if (DistFilter.test(pmg_struct, test_lattice=False, 
                                           min_scale=self.min_mid_scale, max_scale=self.max_mid_scale, pbc=None) \
                            and self.filter_lattice(pmg_struct, after_add_vac=False)) or (not self.filter_dist):
                            # continue to generate until pass length check; but unpassing doesn't mean failing
                            if self.dim < 3:
                                pmg_struct = self.add_vacuum_to_pmg_fn(pmg_struct, pbc=dim2pbc(self.dim))
                            if self.filtering_fn(pmg_struct):
                                self.save_gen(pmg_struct, used_atoms, spg_gen, wp_gen)
                                self.save_process(i)
                                success = True
                        else:
                            num_try += 1
                        total_try += 1
                    except RuntimeError:
                        num_try += 1
                        total_try += 1
                    if num_try >= 2 * self.n_struct_0:
                        self.min_mid_scale *= 0.9
                        self.max_mid_scale *= 1.1
                        log_printer.warning(f"mid scaling parameters changed into {self.min_mid_scale}, {self.max_mid_scale} \
                                            to prevent continuous failures")
                if not success:
                    self.save()
                    raise RuntimeError(f"Generation by {self.gen_tool} failed")

        else:
            max_try = max(int(len(self.compat_sg_list) * self.n_struct / 10), 200000)
            for sg_num in self.compat_sg_list:
                log_printer.info(f"Start SPG {sg_num}")
                for _ in tqdm(range(self.n_struct)):
                    num_try = 0
                    while num_try < max_try:
                        if (num_try % 10 == 0) and (num_try != 0):
                            log_printer.info(f"Currently {num_try + 1}-th trial")
                        try:
                            struc, wp_gen = self.generate_one(sg_num)
                            if struc is None:
                                num_try += 1
                                total_try += 1
                                break
                            pmg_struct, used_atoms, spg_gen, wp_gen = self._decode((struc, wp_gen))
                            if (DistFilter.test(pmg_struct, test_lattice=False, 
                                               min_scale=self.min_mid_scale, max_scale=self.max_mid_scale, pbc=None) \
                                and self.filter_lattice(pmg_struct, after_add_vac=False)) or (not self.filter_dist):
                                if self.dim < 3:
                                    pmg_struct = self.add_vacuum_to_pmg_fn(pmg_struct, pbc=dim2pbc(self.dim))
                                if self.filtering_fn(pmg_struct):
                                    self.save_gen(pmg_struct, used_atoms, spg_gen, wp_gen)
                                    i += 1
                                    self.save_process(i)
                            else:
                                num_try += 1
                            total_try += 1
                            break
                        except RuntimeError:
                            num_try += 1
                            total_try += 1
                        if num_try >= 2 * self.n_struct_0:
                            self.min_mid_scale *= 0.9
                            self.max_mid_scale *= 1.1
                            log_printer.warning(f"mid scaling parameters changed into {self.min_mid_scale}, {self.max_mid_scale} \
                                                to prevent continuous failures")
                    if num_try >= max_try or struc is None:
                        self.save()
                        log_printer.info(f"Generation by {self.gen_tool} failed in SPG {sg_num}")
                log_printer.info(f"Finish SPG {sg_num}")

        log_printer.info(f"Generation by {self.gen_tool} succeeded with {i + 1}/{total_try}")
        return self.pmg_struct_list, self.used_atoms_list, self.spg_list, self.wp_list


class RandGenWithPYX(RandGenPYX):
    def __init__(self, system_name, filter_dist, len_lbs, len_ubs, ang_lbs, ang_ubs, 
                 gen_tool, n_struct, dim, seed, root, load_cache, logger, init_spg_list,
                 filtering_fn, custom_name):
        assert gen_tool in [OptimizerName.pyxtal_rs()[0], OptimizerName.pyxtal_rs_uni()[0]]
        super().__init__(dim=dim, system_name=system_name, filter_dist=filter_dist, n_struct=n_struct, gen_tool=gen_tool,
                         len_lbs=len_lbs, len_ubs=len_ubs, ang_lbs=ang_lbs, ang_ubs=ang_ubs, 
                         seed=seed, root=root, load_cache=load_cache, zip_file=True, init_spg_list=init_spg_list, logger=logger,
                         filtering_fn=filtering_fn, custom_name=custom_name)
        return

    def _preprocess_for_gen(self):
        return

    def generate_one(self, sg_num):
        struc = pyxtal()
        # ATTENTION: not using 'thickness' in pyxtal: minimum thickness is set as 3 A
        struc.from_random(dim=self.dim, group=sg_num, species=self.elements, 
                          numIons=self.num_species_atom, seed=self.seed)
        return struc, None


def pyx_calculator(e_cal, pyx_struct):
    pmg_struct = pyx_struct.to_pymatgen()
    e_cal.build_atoms(pmg_struct)
    e_init = e_cal.cal_energy()
    return e_init


def wpc_to_sites(wpc):
    return [[f"{wp[0]}{wp[1]}" for wp in wpc_ele] for wpc_ele in wpc]


def real_nstruct_for_uni(n_struct, num_species_atom, dim):
    spg_list = get_spg_list2(num_species_atom, dim=dim)
    return int(len(spg_list) * n_struct)


class RandGenFromWPCList(RandGenPYX):
    def __init__(self, dim, system_name, filter_dist, len_lbs, len_ubs, ang_lbs, ang_ubs, 
                 gen_tool, n_struct, seed, root, load_cache, logger, init_spg_list,
                 filtering_fn, custom_name, is_insist_wpclist=False):
        assert gen_tool in [OptimizerName.wpc_l_rs()[0], OptimizerName.wpc_l_rs_uni()[0]]
        super().__init__(dim=dim, system_name=system_name, filter_dist=filter_dist, n_struct=n_struct, gen_tool=gen_tool,
                         len_lbs=len_lbs, len_ubs=len_ubs, ang_lbs=ang_lbs, ang_ubs=ang_ubs, 
                         seed=seed, root=root, load_cache=load_cache, zip_file=True, init_spg_list=init_spg_list, logger=logger,
                         filtering_fn=filtering_fn, custom_name=custom_name)
        self.is_insist_wpclist = is_insist_wpclist
        return

    def _preprocess_for_gen(self):
        self.wpc_dict, _ = get_all_wyckoff_combination(sg_list=self.compat_sg_list, atom_num=self.num_species_atom, mode='pyx', dim=self.dim)
        return

    def generate_one(self, sg_num):
        wpc_list = self.wpc_dict[sg_num]
        # rare case: pyxtal.symmetry.Group().check_compatible() == True, but no WPC is successfully discovered
        if len(wpc_list) == 0:
            log_printer.warning(f"SG {sg_num} is not available by WPC list for the stoichiometry")
            if self.is_insist_wpclist:
                log_printer.warning(f"Return None, try another SG")
                return None, None
            else:
                log_printer.warning(f"Try pyxtal without site")
                struc = pyxtal()
                struc.from_random(dim=self.dim, group=sg_num, species=self.elements,
                                  numIons=self.num_species_atom, seed=self.seed)
                return struc, None
        wpc_idx = np.random.choice(range(len(wpc_list)))
        wpc = wpc_list[wpc_idx]
        sites = wpc_to_sites(wpc)
        struc = pyxtal()
        # ATTENTION: not using 'thickness' in pyxtal: minimum thickness is set as 3 A
        struc.from_random(dim=self.dim, group=sg_num, species=self.elements,
                          numIons=self.num_species_atom, sites=sites, seed=self.seed)
        return struc, wpc_idx


class RandGenDirector:
    def __init__(self, dim, system_name, filter_dist, len_lbs, len_ubs, ang_lbs, ang_ubs, gen_tool, n_struct, seed=0,
                 is_insist_wpclist=True, id_offset_for_rg=0, init_spg_list=None,
                 logger=None, file_root=DATASET_ROOT, load_cache=True, filtering_fn=None, custom_name=None):
        self.system_name = system_name
        self.filter_dist = filter_dist
        self.gen_tool = gen_tool
        self.n_struct = n_struct
        self.dim = dim
        self.len_lbs, self.len_ubs, self.ang_lbs, self.ang_ubs = \
            get_lat_bounds(get_system_size(system_name), len_lbs, len_ubs, ang_lbs, ang_ubs)
        self.seed = seed
        self.init_spg_list = init_spg_list
        self.id_offset_for_rg = id_offset_for_rg
        self.logger = logger
        self.file_root = file_root
        self.filtering_fn = filtering_fn
        self.custom_name = custom_name
        self.is_insist_wpclist = is_insist_wpclist
        self.load_cache = load_cache
        return

    def __call__(self):
        if self.gen_tool in [OptimizerName.random_search()[0], OptimizerName.random_search_uni()[0]]:
            rg = SimpleRandGen(dim=self.dim, system_name=self.system_name, 
                               len_lbs=self.len_lbs, len_ubs=self.len_ubs, ang_lbs=self.ang_lbs, ang_ubs=self.ang_ubs, 
                               gen_tool=self.gen_tool, n_struct=self.n_struct, seed=self.seed, filter_dist=self.filter_dist,
                               logger=self.logger, root=self.file_root, load_cache=self.load_cache,
                               init_spg_list=self.init_spg_list, filtering_fn=self.filtering_fn, custom_name=self.custom_name)
        elif self.gen_tool in init_set_list:
            rg = RandGenFromWPCList(dim=self.dim, system_name=self.system_name, 
                                    len_lbs=self.len_lbs, len_ubs=self.len_ubs, ang_lbs=self.ang_lbs, ang_ubs=self.ang_ubs, 
                                    gen_tool=self.gen_tool, n_struct=self.n_struct, seed=self.seed, filter_dist=self.filter_dist,
                                    logger=self.logger, root=self.file_root, load_cache=self.load_cache, 
                                    is_insist_wpclist=self.is_insist_wpclist,
                                    init_spg_list=self.init_spg_list, filtering_fn=self.filtering_fn, custom_name=self.custom_name)
        else:
            rg = RandGenWithPYX(dim=self.dim, system_name=self.system_name, 
                                len_lbs=self.len_lbs, len_ubs=self.len_ubs, ang_lbs=self.ang_lbs, ang_ubs=self.ang_ubs, 
                                gen_tool=self.gen_tool, n_struct=self.n_struct, seed=self.seed, filter_dist=self.filter_dist,
                                logger=self.logger, root=self.file_root, load_cache=self.load_cache,
                                init_spg_list=self.init_spg_list, filtering_fn=self.filtering_fn, custom_name=self.custom_name)
        pmg_struct_list, used_atoms, spg_gen, wp_gen = rg()
        return pmg_struct_list, used_atoms, spg_gen, wp_gen


class RGEnergy(RandGenFileManager):
    """ This is to manage energy estimation for already generated structures. Therefore, this class assumes that
        the structures have been generated and saved. """

    def __init__(self, dim, system_name, len_lbs, len_ubs, ang_lbs, ang_ubs, e_cal, n_struct, gen_tool, seed, init_spg_list,
                 load_cache, logger, use_rlx_type, struct_file_root, custom_name, rlx_steps=_steps):
        energy_file_name = get_energy_file_name(system_name, dim=dim, gen_tool=gen_tool, use_rlx_type=use_rlx_type,
                                                len_lbs=len_lbs, len_ubs=len_ubs, ang_lbs=ang_lbs, ang_ubs=ang_ubs,
                                                rlx_steps=rlx_steps, n_struct=n_struct, seed=seed, e_cal_name=e_cal.name,
                                                init_spg_list=init_spg_list, custom_name=custom_name)
        energy_file_name += f'.pkl'
        super().__init__(dim=dim, system_name=system_name, n_struct=n_struct, gen_tool=gen_tool, 
                         len_lbs=len_lbs, len_ubs=len_ubs, ang_lbs=ang_lbs, ang_ubs=ang_ubs,
                         seed=seed, file_name=energy_file_name, struct_file_root=struct_file_root, load_cache=load_cache,
                         init_spg_list=init_spg_list, zip_file=True, logger=logger, custom_name=custom_name)
        self.e_cal = e_cal
        self.init_spg_list = init_spg_list
        return

    def _preprocess_for_gen(self):
        self.load_struct_file()
        return

    def load_struct_file(self):
        zip_file = True if 'pyx' in self.gen_tool else False
        struct_file = get_struct_file_name(system_name=self.system_name, dim=self.dim, gen_tool=self.gen_tool, n_struct=self.n_struct, 
                                           len_lbs=self.len_lbs, len_ubs=self.len_ubs, ang_lbs=self.ang_lbs, ang_ubs=self.ang_ubs,
                                           init_spg_list=self.init_spg_list, seed=self.seed, custom_name=self.custom_name)
        struct_file += '.pkl'
        loader = FileLoader(file_dire=self.struct_file_root + '/' + struct_file, load_cache=True, zip_file=zip_file)
        assert loader.load_flag, "Please generate structures before calling energy estimator"
        self.prime_pmg_struct_list, self.prime_used_atoms_list, self.prime_spg_list, self.prime_wp_list = loader.file
        return

    def _process_loaded_gen(self):
        # after loading
        if not self.load_flag:
            # init
            self.pmg_struct_list, self.energy_list, self.forces_list = [], [], []
            self.used_atoms_list, self.spg_list, self.wp_list = [], [], []
            self.file = {'struct_list': [], 'struct_e_list': [], 'struct_f_list': [],
                         'used_atoms': [], 'spg_gen': [], 'wp_gen': []}
        else:
            # continue to calculate
            self.pmg_struct_list, self.energy_list = self.file['struct_list'], self.file['struct_e_list']
            self.used_atoms_list, self.spg_list, self.wp_list = self.file['used_atoms'], self.file['spg_gen'], \
                self.file['wp_gen']
            if 'struct_f_list' in self.file and len(self.file['struct_f_list']) == len(self.pmg_struct_list):
                self.forces_list = self.file['struct_f_list']
            else:
                self.forces_list = []
        return

    def save_gen(self, pmg_struct, energy, forces, used_atoms, spg_gen, wp_gen):
        self.used_atoms_list.append(used_atoms)
        self.spg_list.append(spg_gen)
        self.wp_list.append(wp_gen)
        self.pmg_struct_list.append(pmg_struct)
        self.energy_list.append(energy)
        self.forces_list.append(forces)
        self.file = {'struct_list': self.pmg_struct_list, 'struct_e_list': self.energy_list,
                     'struct_f_list': self.forces_list, 'used_atoms': self.used_atoms_list,
                     'spg_gen': self.spg_list, 'wp_gen': self.wp_list}
        return

    def estimate(self, struct, label):
        raise NotImplementedError

    def generate(self):
        current_size = len(self.pmg_struct_list)
        for i in range(current_size, len(self.prime_pmg_struct_list)):
            struct = self.prime_pmg_struct_list[i]
            if self.e_cal.name == EnergyEvaluatorName.vasp():
                self.e_cal.set_output(label=f'vaspout-{i}')
            struct, energy, forces = self.estimate(struct, i)
            struct = get_savable_structures(struct)
            log_printer.info(f'{i}-th generated structure: energy {energy}, forces {average_force(forces)}')
            self.save_gen(pmg_struct=struct, energy=energy, forces=forces, used_atoms=self.prime_used_atoms_list[i],
                          spg_gen=self.prime_spg_list[i], wp_gen=self.prime_wp_list[i])
            self.save_process(i)
        return

    def __call__(self, *args, **kwargs):
        current_size = len(self.pmg_struct_list)
        if OptimizerName.uniform_gen()[0] in self.gen_tool:
            _, num_species_atom = unique_elements(get_species_list(self.system_name))
            n_struct = real_nstruct_for_uni(self.n_struct, num_species_atom, dim=self.dim)
        else:
            n_struct = self.n_struct
        if current_size < n_struct:
            self._preprocess_for_gen()
            log_printer.info(f"Load {current_size} calculated samples, "
                             f"continue to calculate remaining {n_struct - current_size} in the same file")
            self.generate()
            self.save()
        return self.file


class RGEnergyEstimator(RGEnergy):
    def __init__(self, dim, system_name, len_lbs, len_ubs, ang_lbs, ang_ubs, e_cal, n_struct, gen_tool, seed, init_spg_list, 
                 load_cache, logger, custom_name, struct_file_root):
        super().__init__(dim=dim, system_name=system_name, n_struct=n_struct, gen_tool=gen_tool, 
                         len_lbs=len_lbs, len_ubs=len_ubs, ang_lbs=ang_lbs, ang_ubs=ang_ubs,
                         seed=seed, e_cal=e_cal, use_rlx_type=rlx_type[0], init_spg_list=init_spg_list, 
                         struct_file_root=struct_file_root, load_cache=load_cache,
                         logger=logger, custom_name=custom_name)
        return

    def estimate(self, struct, label):
        try:
            self.e_cal.build_atoms(struct)
            energy = self.e_cal.cal_energy()
            forces = self.e_cal.cal_forces()
        except Exception:
            if not DEBUG:
                energy, forces = None, None
            else:
                raise Exception("Calculation failed")
        return struct, energy, forces


class RGRelaxer(RGEnergy):
    def __init__(self, dim, system_name, len_lbs, len_ubs, ang_lbs, ang_ubs, e_cal, n_struct, gen_tool, seed, init_spg_list,
                 load_cache, logger, custom_name, struct_file_root, steps=_steps):
        self.steps = steps
        super().__init__(dim=dim, system_name=system_name, n_struct=n_struct, gen_tool=gen_tool, 
                         len_lbs=len_lbs, len_ubs=len_ubs, ang_lbs=ang_lbs, ang_ubs=ang_ubs,
                         seed=seed, e_cal=e_cal, use_rlx_type=rlx_type[1], init_spg_list=init_spg_list,
                         rlx_steps=steps, struct_file_root=struct_file_root, load_cache=load_cache,
                         logger=logger, custom_name=custom_name)
        return

    def _preprocess_for_gen(self):
        self.load_struct_file()
        self.e_cal.set_traj_logger(logger=None, traj_dir=self.file_name[:-len('.pkl')])
        return

    def estimate(self, struct, label):
        try:
            try:
                self.e_cal.build_atoms(struct)
                self.e_cal.set_traj_label(label)
                struct, energy, forces = self.e_cal.cal_relax(fix_sym=True, steps=self.steps)
            except Exception:
                self.e_cal.build_atoms(struct)
                self.e_cal.set_traj_label(label)
                struct, energy, forces = self.e_cal.cal_relax(fix_sym=False, steps=self.steps)
        except Exception:
            if not DEBUG:
                energy, forces = None, None
            else:
                raise Exception("Calculation failed")
        return struct, energy, forces


class RGEnergyDirector:
    def __init__(self, dim, system_name, len_lbs, len_ubs, ang_lbs, ang_ubs,e_cal, n_struct, gen_tool, use_rlx_type, seed, 
                 init_spg_list, load_cache, logger, rlx_steps=_steps, custom_name=None, struct_file_root=DATASET_ROOT):
        assert use_rlx_type in rlx_type, f"Supported dataset energy type: {rlx_type}"
        self.dim = dim
        self.system_name = system_name
        self.len_lbs, self.len_ubs, self.ang_lbs, self.ang_ubs = \
            get_lat_bounds(get_system_size(system_name), len_lbs, len_ubs, ang_lbs, ang_ubs)
        self.e_cal = e_cal
        self.n_struct = n_struct
        self.gen_tool = gen_tool
        self.use_rlx_type = use_rlx_type
        self.seed = seed
        self.init_spg_list = init_spg_list
        self.load_cache = load_cache
        self.logger = logger
        self.custom_name = custom_name
        self.struct_file_root = struct_file_root
        self.rlx_steps = rlx_steps
        return

    def __call__(self, *args, **kwargs):
        if self.use_rlx_type == rlx_type[0]:
            self.rge = RGEnergyEstimator(dim=self.dim, system_name=self.system_name, 
                                         len_lbs=self.len_lbs, len_ubs=self.len_ubs, ang_lbs=self.ang_lbs, ang_ubs=self.ang_ubs,
                                         gen_tool=self.gen_tool, n_struct=self.n_struct, e_cal=self.e_cal, seed=self.seed,
                                         custom_name=self.custom_name, struct_file_root=self.struct_file_root,
                                         init_spg_list=self.init_spg_list, load_cache=True, logger=self.logger)
        elif self.use_rlx_type in [rlx_type[1], rlx_type[2]]:
            self.rge = RGRelaxer(dim=self.dim, system_name=self.system_name, steps=self.rlx_steps,
                                 len_lbs=self.len_lbs, len_ubs=self.len_ubs, ang_lbs=self.ang_lbs, ang_ubs=self.ang_ubs,
                                 gen_tool=self.gen_tool, n_struct=self.n_struct, e_cal=self.e_cal, seed=self.seed,
                                 custom_name=self.custom_name, struct_file_root=self.struct_file_root,
                                 init_spg_list=self.init_spg_list, load_cache=True, logger=self.logger)
        else:
            raise NotImplementedError("Unknown rlx type")
        self.file = self.rge()
        self.used_atoms_list = self.file['used_atoms']
        self.spg_list = self.file['spg_gen']
        self.wp_list = self.file['wp_gen']
        if self.use_rlx_type == rlx_type[2]:
            self.rge.load_struct_file()
            init_used_atom_0 = np.concatenate(self.rge.prime_used_atoms_list[0])
            used_atom_0 = np.concatenate(self.used_atoms_list[0])
            init_struct_0 = self.rge.prime_pmg_struct_list[0].cart_coords
            struct_0 = self.file['struct_list'][0].cart_coords
            assert np.all(init_used_atom_0 == used_atom_0) and np.any(init_struct_0 != struct_0)
            self.pmg_struct_list = self.rge.prime_pmg_struct_list
            self.file['struct_list'] = self.rge.prime_pmg_struct_list
        else:
            self.pmg_struct_list = self.file['struct_list']
        self.energy_list = self.file['struct_e_list']
        if 'struct_f_list' in self.file:
            self.forces_list = self.file['struct_f_list']
        else:
            self.forces_list = []
        return self.file


def get_training_set(rge, e_sampling_threshold):
    pmg_struct_train_list, energy_train_list, forces_train_list = [], [], []
    used_atoms_train_list, spg_train_list, wp_train_list = [], [], []
    has_force = hasattr(rge, 'forces_list') and len(rge.forces_list) == len(rge.pmg_struct_list)
    for i in range(len(rge.pmg_struct_list)):
        if rge.energy_list[i] is not None:
            if rge.energy_list[i] < e_sampling_threshold:
                # if DistFilter.test(rge.pmg_struct_list[i], test_lattice=True, pbc=dim2pbc(rge.dim)):
                    pmg_struct_train_list.append(rge.pmg_struct_list[i])
                    energy_train_list.append(rge.energy_list[i])
                    used_atoms_train_list.append(rge.used_atoms_list[i])
                    spg_train_list.append(rge.spg_list[i])
                    wp_train_list.append(rge.wp_list[i])
                    if has_force:
                        forces_train_list.append(rge.forces_list[i])
                    else:
                        forces_train_list.append(np.ones(shape=(get_system_size(rge.system_name), 3)))
    return pmg_struct_train_list, energy_train_list, forces_train_list, \
        used_atoms_train_list, spg_train_list, wp_train_list


def analyze_dataset_type(gen_tool):
    # consequence (following def get_energy_file_name): base_tool_type - rlx_type - mixing or not
    use_mixing = False
    if mixing in gen_tool:
        gen_tool = gen_tool[:-len(mixing) - 1]
        use_mixing = True
    rlx_sign_len = len(rlx_type[0])
    use_rlx_type = gen_tool[-rlx_sign_len:]
    gen_tool = gen_tool[:-rlx_sign_len - 1]
    return gen_tool, use_rlx_type, use_mixing
