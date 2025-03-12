import numpy as np
from logzero import logger as log_printer
from src.spg.get_wyckoff_position import get_all_wyckoff_combination
from src.spg.spg_utils import sg_list, Random2Wyckoff, shift_spg, get_spg_list
from src.struct_utils import unique_elements, DistFilter


# average lattice length a, b, c of materials with number of atoms x <= 20, 20 < x <= 50,
# 50 < x <= 80, 80 < x <= 100, and x > 100,
# based on more than 150,000 materials from Material Project

threshold = [20, 50, 80, 100, 'large']
results_aver_a = [5.859180830043813, 7.391743829404799, 9.074192498061459,
                  10.082653839712718, 11.30067570941344]
results_aver_b = [6.125914597335389, 8.169390782111698, 9.637091463252037,
                  10.56358747132569, 10.862278427360812]
results_aver_c = [7.883718134982488, 10.779934874718478, 13.756538741124768,
                  15.800706725323218, 20.107668351531817]

# threshold = [20, 50, 80, 'large']
# results_aver_a = [5.859180830043813, 7.391743829404799, 9.074192498061459, 10.890674465020503]
# results_aver_b = [6.125914597335389, 8.169390782111698, 9.637091463252037, 10.761735352798937]
# results_aver_c = [7.883718134982488, 10.779934874718478, 13.756538741124768, 18.65789173402984]


default_ang_lb = 50
default_ang_ub = 130
default_ang_mean = np.mean([default_ang_lb, default_ang_ub])


class LatticeLengthController:
    def __init__(self, num_atom, logger=None):
        self.num_atom = num_atom
        self.min_scale = 0.8
        self.max_scale = 2
        self.logger = logger
        return

    def _min(self, i):
        a = np.floor(self.min_scale * np.min([results_aver_a[i], results_aver_b[i], results_aver_c[i]]))
        if self.logger is not None:
            self.logger.record_random(operation='Get minimum lattice length', result=a)
        return a

    def min(self):
        for i in range(len(threshold) - 1):
            if self.num_atom <= threshold[i]:
                return self._min(i)
        return self._min(-1)

    def _max(self, i):
        a = self.max_scale * np.ceil(np.mean(
            [results_aver_a[i], results_aver_b[i], results_aver_c[i]]
        ))
        if self.logger is not None:
            self.logger.record_random(operation='Get maximum lattice length', result=a)
        return a

    def max(self):
        for i in range(len(threshold) - 1):
            if self.num_atom < threshold[i]:
                return self._max(i)
        return self._max(-1)

    def mean(self):
        a = np.mean([self.min(), self.max()])
        if self.logger is not None:
            self.logger.record_random(operation='Get default lattice length', result=a)
        return a


class RandomGenerator:
    def __init__(self, wyckoffs_dict, wyckoffs_max, species_list, spg_num, wp_num,
                 lat_lb, lat_ub, ang_lb, ang_ub, logger=None):
        self.species_list = species_list
        self.spg_num = spg_num
        self.wp_num = wp_num
        self.logger = logger
        self.elements, self.num_species_atom = unique_elements(species_list)
        self.wyckoffs_dict, self.wyckoffs_max = wyckoffs_dict, wyckoffs_max
        self.lat_lb, self.lat_ub, self.ang_lb, self.ang_ub = lat_lb, lat_ub, ang_lb, ang_ub
        return

    def _generate(self):
        a, b, c = np.random.uniform(self.lat_lb, self.lat_ub, size=(3,))
        alpha, beta, gamma = np.random.uniform(self.ang_lb, self.ang_ub, size=(3,))
        pos = np.random.uniform(0, 1, size=(len(self.species_list), 3))
        return (a, b, c, alpha, beta, gamma), pos

    def generate(self):
        lat_para, pos = self._generate()
        r2w = Random2Wyckoff(self.wyckoffs_dict, self.wyckoffs_max, self.species_list, lat_para, pos,
                             self.spg_num, self.wp_num, logger=self.logger)
        pmg_struct = r2w.get_pmg_struct()
        return pmg_struct, r2w.used_decode


def random_generate(species_list, nstruc, id_offset, init_pos_path, seed, filter_dist, logger=None):
    np.random.seed(seed)
    init_struc_data, used_atoms, spg_gen, wp_gen = [], [], [], []
    elements, num_species_atom = unique_elements(species_list)
    wyckoffs_dict, wyckoffs_max = get_all_wyckoff_combination(
        sg_list=sg_list, atom_num=num_species_atom)
    spg_list = get_spg_list(wyckoffs_dict)

    llc = LatticeLengthController(len(species_list), logger=logger)
    lat_lb, lat_ub = llc.min(), llc.max()

    i, j = 0, 0
    if filter_dist:
        dist = 1.5 if len(species_list) < 200 else 1.0
        dist_filter = DistFilter(num_species_atom, dist=dist)
    while i < nstruc:
        spg_num = np.random.choice(spg_list)
        wyckoff_num = np.random.choice(list(range(wyckoffs_max)))
        spg_num = shift_spg(spg_num, wyckoffs_dict, 230, 2, logger)
        rg = RandomGenerator(wyckoffs_dict=wyckoffs_dict, wyckoffs_max=wyckoffs_max,
                             species_list=species_list, spg_num=spg_num, wp_num=wyckoff_num,
                             lat_lb=lat_lb, lat_ub=lat_ub, ang_lb=default_ang_lb, ang_ub=default_ang_ub,
                             logger=logger)
        pmg_struct, used_decode = rg.generate()
        if filter_dist:
            if not dist_filter(pmg_struct):
                if j % 1 == 0:
                    log_printer.info(f'structure generation fail on {i}-th material after {j} generation trials')
                j += 1
                continue
        log_printer.info(f'success on {i}-th material generation')
        spg_gen.append(spg_num)
        wp_gen.append(wyckoff_num)
        if logger is not None:
            logger.record_anything1(operation='Space Group Random Selection', result=spg_num)
            logger.record_anything2(operation='Wyckoff Positions Random Selection', result=wyckoff_num)
        init_struc_data.append(pmg_struct)
        used_atoms.append(used_decode)
        cid = len(init_struc_data) + id_offset
        if init_pos_path is not None:
            out_poscar(pmg_struct, cid, init_pos_path, spg_num, 'spg_sym')
        i += 1
        j += 1
    print(f'Generation success rate: {i / j:.4}')
    return init_struc_data, used_atoms, spg_gen, wp_gen


def out_poscar(struc, cid, fpath, spg_num, spg_sym):
    # ---------- poscar format
    pos = struc.to(fmt='poscar')
    pos = pos.split('\n')
    blank_indx = pos.index('')    # cut unnecessary parts
    pos = pos[:blank_indx]
    pos[0] = 'ID_{}, {}, {}'.format(cid, spg_num, spg_sym)    # replace with ID
    lines = [line+'\n' for line in pos]

    # ---------- append POSCAR
    with open(fpath, 'a+') as f:
        for line in lines:
            f.write(line)
