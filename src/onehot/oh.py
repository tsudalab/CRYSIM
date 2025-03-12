import numpy as np
from src.onehot.oh_basic import \
    OneHotProcessor, OneHotLatticeScaleProcessor, \
    OneHotAtomNumberConstraintsBuilder, OneHotAtomDistConstraintsBuilder, \
    OneHotLatticeConstraintsBuilder
from src.onehot.oh_cs import \
    OneHotProcessorCrysSys, OneHotCrysSysConstraintsBuilder, OneHotVecProcessorCrysSys
from src.struct_utils import unique_elements, get_cs_list
from src.onehot.oh_spg import \
    OneHotProcessorSPG, OneHotSPGConstraintsBuilder, OneHotVecProcessorSPG
from src.onehot.oh_cs_spg import \
    OneHotProcessorCSSPG, OneHotCSSPGConstraintsBuilder, OneHotVecProcessorCSSPG
from src.spg.spg_utils import sg_list, get_all_wyckoff_combination, get_spg_list
from src.log_analysis import print_or_record


class OneHotConstraintsType:
    def __init__(self, args):
        control_lat_shape, control_crys_sys, control_spg = self.get_arg(args)
        assert (isinstance(control_lat_shape, bool)) and \
               (isinstance(control_crys_sys, bool)) and \
               (isinstance(control_spg, bool))

        if (not control_lat_shape) and (not control_crys_sys) and (not control_spg):
            self.constraints_scheme = 0
            self.constraints_name = ''
        elif control_lat_shape and (not control_crys_sys) and (not control_spg):
            self.constraints_scheme = 1
            self.constraints_name = 'Lat'
        elif control_crys_sys and (not control_spg):
            self.constraints_scheme = 2
            self.constraints_name = 'CrysSys'
        elif (not control_crys_sys) and control_spg:
            self.constraints_scheme = 3
            self.constraints_name = 'SPG'
        elif control_crys_sys and control_spg:
            self.constraints_scheme = 4
            self.constraints_name = 'CSSPG'
        return

    def get_arg(self, args):
        if (isinstance(args, list)) or (isinstance(args, tuple)):
            return args[0], args[1], args[2]
        elif isinstance(args, dict):
            return args['control_lat_shape'], args['control_crys_sys'], args['control_spg']
        else:
            raise NotImplementedError


class OneHotImplementationType:
    def __init__(self, use_constraints):
        assert isinstance(use_constraints, bool)

        if not use_constraints:
            self.implementation_scheme = 0
            self.implementation_name = 'direct_sample'
        else:
            self.implementation_scheme = 1
            self.implementation_name = 'opt_all'
        return


class OneHotDirector:
    def __init__(self, species_list, splits, lat_ub, dist_min,
                 constraint_type, implement_type, logger,
                 default_crys_sys=None, default_spg=None, default_wp=None):

        self.logger = logger

        self.default_crys_sys = default_crys_sys
        self.default_spg = default_spg
        self.default_wp = default_wp

        self.species_list = species_list
        self.splits = splits
        self.lat_ub = lat_ub
        self.dist_min = dist_min

        if isinstance(constraint_type, OneHotConstraintsType):
            oht = constraint_type
        else:
            oht = OneHotConstraintsType(constraint_type)
        self.constraints_scheme = oht.constraints_scheme
        if isinstance(implement_type, OneHotImplementationType):
            ohi = implement_type
        else:
            ohi = OneHotImplementationType(implement_type)
        self.implementation_scheme = ohi.implementation_scheme

        return

    def unwrap(self):
        if self.implementation_scheme == 0:
            if self.constraints_scheme == 0:
                return OneHotProcessor(species_list=self.species_list,
                                       splits=self.splits,
                                       lat_ub=self.lat_ub,
                                       logger=self.logger)
            elif self.constraints_scheme == 1:
                return OneHotLatticeScaleProcessor(species_list=self.species_list,
                                                   splits=self.splits,
                                                   lat_ub=self.lat_ub,
                                                   logger=self.logger)
            elif self.constraints_scheme == 2:
                return OneHotProcessorCrysSys(species_list=self.species_list,
                                              splits=self.splits,
                                              lat_ub=self.lat_ub,
                                              logger=self.logger,
                                              default_crys_sys=self.default_crys_sys)
            elif self.constraints_scheme == 3:
                _, num_species_atom = unique_elements(self.species_list)
                wyckoffs_dict, wyckoffs_max = get_all_wyckoff_combination(
                    sg_list=sg_list, atom_num=num_species_atom)
                spg_list = get_spg_list(wyckoffs_dict)
                return OneHotProcessorSPG(species_list=self.species_list,
                                          splits=self.splits,
                                          lat_ub=self.lat_ub,
                                          wyckoffs_dict=wyckoffs_dict,
                                          wyckoffs_max=wyckoffs_max,
                                          spg_list=spg_list,
                                          logger=self.logger,
                                          default_spg=self.default_spg,
                                          default_wp=self.default_wp)
            elif self.constraints_scheme == 4:
                _, num_species_atom = unique_elements(self.species_list)
                wyckoffs_dict, wyckoffs_max = get_all_wyckoff_combination(
                    sg_list=sg_list, atom_num=num_species_atom)
                spg_list = get_spg_list(wyckoffs_dict)
                cs_list = get_cs_list(spg_list)
                return OneHotProcessorCSSPG(species_list=self.species_list,
                                            splits=self.splits,
                                            lat_ub=self.lat_ub,
                                            wyckoffs_dict=wyckoffs_dict,
                                            wyckoffs_max=wyckoffs_max,
                                            cs_list=cs_list,
                                            spg_list=spg_list,
                                            logger=self.logger,
                                            default_spg=self.default_spg,
                                            default_wp=self.default_wp)
            else:
                raise Exception('We do not support this constraints scheme')

        elif self.implementation_scheme == 1:
            if self.constraints_scheme == 0:
                return OneHotBasic(species_list=self.species_list,
                                   splits=self.splits,
                                   lat_ub=self.lat_ub,
                                   dist_min=self.dist_min,
                                   control_lat_shape=False,
                                   logger=self.logger)
            elif self.constraints_scheme == 1:
                return OneHotBasic(species_list=self.species_list,
                                   splits=self.splits,
                                   lat_ub=self.lat_ub,
                                   dist_min=self.dist_min,
                                   control_lat_shape=True,
                                   logger=self.logger)
            elif self.constraints_scheme == 2:
                return OneHotCrysSys(species_list=self.species_list,
                                     splits=self.splits,
                                     lat_ub=self.lat_ub,
                                     dist_min=self.dist_min,
                                     logger=self.logger)
            elif self.constraints_scheme == 3:
                return OneHotSPG(species_list=self.species_list,
                                 splits=self.splits,
                                 lat_ub=self.lat_ub,
                                 dist_min=self.dist_min,
                                 logger=self.logger)
            elif self.constraints_scheme == 4:
                return OneHotCSSPG(species_list=self.species_list,
                                   splits=self.splits,
                                   lat_ub=self.lat_ub,
                                   dist_min=self.dist_min,
                                   logger=self.logger)
            else:
                raise Exception('We do not support this constraints scheme')
        else:
            raise Exception('We do not support this implementation scheme')


class OneHotFactoryFrame:
    def __init__(self):
        self._load_ohp()
        return

    def _load_ohp(self):
        self.ohp = None
        return

    def get_ohp(self):
        return self.ohp

    def _load_linear_constraints(self):
        return

    def _get_linear_constraints(self, linear_constraints_list):
        lin_c_mat, lin_c_bias, lin_c_rate = [], [], []
        for lin in linear_constraints_list:
            lin_c_mat, lin_c_bias, lin_c_rate = self.multi_append(
                lin_c_mat, lin_c_bias, lin_c_rate, lin)
        return np.concatenate(lin_c_mat, axis=0), np.array(lin_c_bias), np.array(lin_c_rate)

    def get_constraints(self):
        raise NotImplementedError

    @staticmethod
    def append(_list, _item):
        assert isinstance(_list, list)
        if isinstance(_item, list):
            _list += _item
        else:
            _list.append(_item)
        return _list

    def multi_append(self, list1, list2, list3, item):
        return self.append(list1, item[0]), \
            self.append(list2, item[1]), \
            self.append(list3, item[2])


class OneHotFactory(OneHotFactoryFrame):
    def __init__(self, species_list, splits, lat_ub, dist_min, logger):
        self.species_list = species_list
        self.splits = splits
        self.lat_ub = lat_ub
        self.dist_min = dist_min
        self.logger = logger
        # self.control_lat_shape = control_lat_shape

        super().__init__()
        self._load_constraints_variable_num()
        self._load_linear_constraints()
        self._load_quadratic_constraints()
        return

    def _load_linear_constraints_basic(self):
        self.Atom_n = OneHotAtomNumberConstraintsBuilder(self.ohp)
        self.atom_n = self.Atom_n()['linear']  # atom number
        self.atom_n[0] = self._atom_constraints_supplement(self.atom_n[0])

        self.Lat_n = OneHotLatticeConstraintsBuilder(self.ohp)
        self.lat_n = self.Lat_n()['linear']  # lattice number constraints
        self.lat_n[0] = self._lattice_constraints_supplement(self.lat_n[0])

        # if self.control_lat_shape:
        #     self.Lat_scale = OneHotLatticeConstraintsBuilderExtra(self.ohp)
        #     self.lat_scale = self.Lat_scale()['linear']  # lattice shape constraints
        #     for i in range(len(self.lat_scale[0])):
        #         # lattice length + lattice angle
        #         self.lat_scale[0][i] = self._lattice_constraints_supplement(self.lat_scale[0][i])

        return

    def _load_linear_constraints(self):
        self._load_linear_constraints_basic()
        return

    def _load_quadratic_constraints(self):
        if (isinstance(self.dist_min, float)) or (isinstance(self.dist_min, int)):
            self.Atom_d = OneHotAtomDistConstraintsBuilder(self.ohp, self.dist_min)
            self.atom_d = self.Atom_d()['quadratic']  # atom neighboring constraints
            self.atom_d[0] = self._atom_constraints_supplement(self.atom_d[0])
        else:
            self.atom_d = None
        return

    def _load_constraints_variable_num_basic(self):
        self.lattice_precision, num_elements, space_precision = \
            self.ohp.lattice_precision, self.ohp.num_elements, self.ohp.space_precision
        self.atom_precision = num_elements * space_precision
        return

    def _load_constraints_variable_num(self):
        self._load_constraints_variable_num_basic()
        return

    @staticmethod
    def _supplement_left(fill_len, mat):
        sup_left = np.zeros((mat.shape[0], fill_len))
        return np.concatenate([sup_left, mat], axis=1)

    @staticmethod
    def _supplement_right(mat, fill_len):
        sup_right = np.zeros((mat.shape[0], fill_len))
        return np.concatenate([mat, sup_right], axis=1)

    @staticmethod
    def _supplement_both(fill_left, mat, fill_right):
        sup_left = np.zeros((mat.shape[0], fill_left))
        sup_right = np.zeros((mat.shape[0], fill_right))
        return np.concatenate([sup_left, mat, sup_right], axis=1)

    def _atom_constraints_supplement(self, atom_mat):
        return self._supplement_left(self.lattice_precision, atom_mat)

    def _lattice_constraints_supplement(self, lat_mat):
        return self._supplement_right(lat_mat, self.atom_precision)

    def _symmetry_constraints_supplement(self, sym_mat):
        raise NotImplementedError

    def _get_constraints(self, linear_constraints_list):
        return self._get_linear_constraints(linear_constraints_list), self.atom_d

    def get_constraints(self):
        raise NotImplementedError


class OneHotBasic(OneHotFactory):
    def __init__(self, species_list, splits, lat_ub, dist_min, control_lat_shape, logger):
        super().__init__(species_list, splits, lat_ub, dist_min, logger=logger)
        self.control_lat_shape = control_lat_shape
        return

    def _load_ohp(self):
        if not self.control_lat_shape:
            self.ohp = OneHotProcessor(species_list=self.species_list,
                                       splits=self.splits, lat_ub=self.lat_ub,
                                       logger=self.logger)
        else:
            self.ohp = OneHotLatticeScaleProcessor(species_list=self.species_list,
                                                   splits=self.splits, lat_ub=self.lat_ub,
                                                   logger=self.logger)
        return

    def get_constraints(self):
        # if not self.control_lat_shape:
        #     linear_constraints_list = [self.atom_n, self.lat_n]
        # else:
        #     linear_constraints_list = [self.atom_n, self.lat_n, self.lat_scale]
        linear_constraints_list = [self.atom_n, self.lat_n]
        return self._get_constraints(linear_constraints_list)


class OneHotCrysSys(OneHotFactory):
    def __init__(self, species_list, splits, lat_ub, dist_min, logger):
        super().__init__(species_list, splits, lat_ub, dist_min, logger=logger)
        return

    def _load_ohp(self):
        self.ohp = OneHotProcessorCrysSys(species_list=self.species_list,
                                          splits=self.splits, lat_ub=self.lat_ub,
                                          logger=self.logger,
                                          default_crys_sys=None)
        return

    def _load_constraints_variable_num(self):
        self._load_constraints_variable_num_basic()
        self.sym_precision = self.ohp.crys_sys_precision
        return

    def _load_symmetry_constraints(self):
        self.SYM_oh = OneHotCrysSysConstraintsBuilder(self.sym_precision)  # crystal system
        return

    def _load_linear_constraints(self):
        self._load_linear_constraints_basic()
        self._load_symmetry_constraints()
        self.sym_oh = self.SYM_oh()['linear']
        self.sym_oh[0] = self._symmetry_constraints_supplement(self.sym_oh[0])
        return

    def _atom_constraints_supplement(self, atom_mat):
        return self._supplement_left(self.lattice_precision + self.sym_precision, atom_mat)

    def _lattice_constraints_supplement(self, lat_mat):
        return self._supplement_right(lat_mat, self.atom_precision + self.sym_precision)

    def _symmetry_constraints_supplement(self, sym_mat):
        return self._supplement_both(self.lattice_precision, sym_mat, self.atom_precision)

    def get_constraints(self):
        # if not self.control_lat_shape:
        #     linear_constraints_list = [self.atom_n, self.lat_n, self.lat_cs]
        # else:
        #     linear_constraints_list = [self.atom_n, self.lat_n, self.lat_scale, self.lat_cs]
        linear_constraints_list = [self.atom_n, self.lat_n, self.sym_oh]
        return self._get_constraints(linear_constraints_list)


class OneHotSPG(OneHotCrysSys):
    def __init__(self, species_list, splits, lat_ub, dist_min, logger):
        self._load_spg(species_list)
        super().__init__(species_list, splits, lat_ub, dist_min, logger=logger)
        return

    def _load_spg(self, species_list):
        _, self.num_species_atom = unique_elements(species_list)
        self.wyckoffs_dict, self.wyckoffs_max = get_all_wyckoff_combination(
            sg_list=sg_list, atom_num=self.num_species_atom)
        self.spg_list = get_spg_list(self.wyckoffs_dict)
        return

    def _load_ohp(self):
        self.ohp = OneHotProcessorSPG(species_list=self.species_list,
                                      splits=self.splits, lat_ub=self.lat_ub,
                                      wyckoffs_dict=self.wyckoffs_dict,
                                      wyckoffs_max=self.wyckoffs_max,
                                      spg_list=self.spg_list,
                                      logger=self.logger,
                                      default_spg=None,
                                      default_wp=None)
        return

    def _load_constraints_variable_num(self):
        self._load_constraints_variable_num_basic()
        self.spg_precision = self.ohp.spg_precision
        self.wp_precision = self.ohp.wp_precision
        self.sym_precision = self.spg_precision + self.wp_precision
        return

    def _load_symmetry_constraints(self):
        self.SYM_oh = OneHotSPGConstraintsBuilder(self.spg_precision, self.wp_precision)
        return


class OneHotCSSPG(OneHotSPG):
    def __init__(self, species_list, splits, lat_ub, dist_min, logger):
        super().__init__(species_list, splits, lat_ub, dist_min, logger)
        print_or_record(self.logger, f"Possible # of crystal systems: {self.crys_sys_precision}")
        print_or_record(self.logger, f"{self.cs_list}")
        print_or_record(self.logger, f"# of bits for space group representation: {self.spg_precision}")
        return

    def _load_spg(self, species_list):
        _, self.num_species_atom = unique_elements(species_list)
        self.wyckoffs_dict, self.wyckoffs_max = get_all_wyckoff_combination(
            sg_list=sg_list, atom_num=self.num_species_atom)
        self.spg_list = get_spg_list(self.wyckoffs_dict)
        self.cs_list = get_cs_list(spg_list=self.spg_list)
        return

    def _load_ohp(self):
        self.ohp = OneHotProcessorCSSPG(species_list=self.species_list,
                                        splits=self.splits, lat_ub=self.lat_ub,
                                        wyckoffs_dict=self.wyckoffs_dict,
                                        wyckoffs_max=self.wyckoffs_max,
                                        spg_list=self.spg_list,
                                        cs_list=self.cs_list,
                                        logger=self.logger,
                                        default_spg=None,
                                        default_wp=None)
        return

    def _load_constraints_variable_num(self):
        self._load_constraints_variable_num_basic()
        self.crys_sys_precision = self.ohp.crys_sys_precision
        self.spg_precision = self.ohp.spg_precision
        self.wp_precision = self.ohp.wp_precision
        self.sym_precision = self.crys_sys_precision + self.spg_precision + self.wp_precision
        return

    def _load_symmetry_constraints(self):
        self.SYM_oh = OneHotCSSPGConstraintsBuilder(self.crys_sys_precision, self.spg_precision,
                                                    self.wp_precision)
        return


class OneHotCrysSysVec(OneHotFactoryFrame):
    def __init__(self, crys_sys_precision, logger):
        self.crys_sys_precision = crys_sys_precision
        self.logger = logger
        super().__init__()
        self._load_linear_constraints()
        return

    def _load_ohp(self):
        self.ohp = OneHotVecProcessorCrysSys(crys_sys_precision=self.crys_sys_precision,
                                             default_crys_sys=None,
                                             logger=self.logger)
        return

    def _load_linear_constraints(self):
        self.CS_t = OneHotCrysSysConstraintsBuilder(self.crys_sys_precision)
        self.cs_t = self.CS_t()['linear']
        return

    def get_constraints(self):
        linear_constraints_list = [self.cs_t]
        return self._get_linear_constraints(linear_constraints_list)


class OneHotSPGVec(OneHotFactoryFrame):
    def __init__(self, spg_precision, wp_precision, wyckoffs_dict, wyckoffs_max, spg_list, logger):
        self.spg_precision = spg_precision
        self.wp_precision = wp_precision
        self.wyckoffs_dict = wyckoffs_dict
        self.wyckoffs_max = wyckoffs_max
        self.spg_list = spg_list
        self.logger = logger
        super().__init__()
        self._load_linear_constraints()
        return

    def _load_ohp(self):
        self.ohp = OneHotVecProcessorSPG(spg_precision=self.spg_precision,
                                         wp_precision=self.wp_precision,
                                         default_spg=None, default_wp=None,
                                         wycokffs_dict=self.wyckoffs_dict,
                                         wyckoffs_max=self.wyckoffs_max,
                                         spg_list=self.spg_list,
                                         logger=self.logger)
        return

    def _load_linear_constraints(self):
        self.SPG_t = OneHotSPGConstraintsBuilder(self.spg_precision, self.wp_precision)
        self.spg_t = self.SPG_t()['linear']
        return

    def get_constraints(self):
        linear_constraints_list = [self.spg_t]
        return self._get_linear_constraints(linear_constraints_list)


class OneHotCSSPGVec(OneHotFactoryFrame):
    def __init__(self, cs_precision, spg_precision, wp_precision,
                 wyckoffs_dict, wyckoffs_max, spg_list, cs_list, logger):
        self.cs_precision = cs_precision
        self.spg_precision = spg_precision
        self.wp_precision = wp_precision
        self.wyckoffs_dict = wyckoffs_dict
        self.wyckoffs_max = wyckoffs_max
        self.spg_list = spg_list
        self.cs_list = cs_list
        self.logger = logger
        super().__init__()
        self._load_linear_constraints()
        return

    def _load_ohp(self):
        self.ohp = OneHotVecProcessorCSSPG(cs_precision=self.cs_precision,
                                           spg_precision=self.spg_precision,
                                           wp_precision=self.wp_precision,
                                           default_spg=None, default_wp=None,
                                           wycokffs_dict=self.wyckoffs_dict,
                                           wyckoffs_max=self.wyckoffs_max,
                                           spg_list=self.spg_list,
                                           cs_list=self.cs_list,
                                           logger=self.logger)
        return

    def _load_linear_constraints(self):
        self.CSSPG_t = OneHotCSSPGConstraintsBuilder(self.cs_precision, self.spg_precision,
                                                     self.wp_precision)
        self.csspg_t = self.CSSPG_t()['linear']
        return

    def get_constraints(self):
        linear_constraints_list = [self.csspg_t]
        return self._get_linear_constraints(linear_constraints_list)
