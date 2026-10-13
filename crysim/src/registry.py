import pickle
from pathlib import Path
import bz2


class Component:
    def __init__(self, name, real_name):
        self.name = name
        self.real_name = real_name
        return


class SolverName:
    """ solvers """

    """ classical methods """

    @classmethod
    def simulated_annealing(cls):
        return ['sa', 'Simulated Annealing']

    @classmethod
    def bayesian_optimization(cls):
        return ['bo', 'Bayesian Optimization']

    @classmethod
    def particle_swarm_optimization(cls):
        return ['pso', 'Particle Swarm Optimization']

    """ qubo solvers """

    @classmethod
    def amplify(cls):
        return ['amp', 'Amplify']

    @classmethod
    def dwave(cls):
        return ['dw', 'DWave']

    @classmethod
    def leap_hybrid(cls):
        return ['leap-h', 'LeapHybrid']

    @classmethod
    def amplify_dwave(cls):
        return [f'{cls.amplify()[0]}-{cls.dwave()[0]}',
                f'{cls.amplify()[1]}-{cls.dwave()[1]}']

    @classmethod
    def amplify_leap_hybrid(cls):
        return [f'{cls.amplify()[0]}-{cls.leap_hybrid()[0]}',
                f'{cls.amplify()[1]}-{cls.leap_hybrid()[1]}']

    @classmethod
    def cpsat_ortools(cls):
        return ['cpsat', 'CP-SAT-ORTools']

    """ instantiation of classical solvers """

    @classmethod
    def hyperopt(cls):
        return ['hy', 'hyperopt']

    @classmethod
    def hyperopt_bayesian_optimization(cls):
        return [f'{SolverName.hyperopt()[0]}-{cls.bayesian_optimization()[0]}',
                f'{SolverName.hyperopt()[1]}-{cls.bayesian_optimization()[1]}']

    @classmethod
    def hyperopt_simulated_annealing(cls):
        return [f'{SolverName.hyperopt()[0]}-{cls.simulated_annealing()[0]}',
                f'{SolverName.hyperopt()[1]}-{cls.simulated_annealing()[1]}']

    @classmethod
    def hyperopt_onehot_bayesian_optimization(cls):
        return [f'{SolverName.hyperopt()[0]}4oh-{cls.bayesian_optimization()[0]}',
                f'{SolverName.hyperopt()[1]}4oh-{cls.bayesian_optimization()[1]}']

    @classmethod
    def hyperopt_onehot_simulated_annealing(cls):
        return [f'{SolverName.hyperopt()[0]}4oh-{cls.simulated_annealing()[0]}',
                f'{SolverName.hyperopt()[1]}4oh-{cls.simulated_annealing()[1]}']

    @classmethod
    def mystic(cls):
        return ['mt', 'mystic']

    """ classical solver for infinite solution set """

    @classmethod
    def hyperopt_set_bayesian_optimization(cls):
        return [f'{SolverName.hyperopt()[0]}4set-{cls.bayesian_optimization()[0]}',
                f'{SolverName.hyperopt()[1]}4set-{cls.bayesian_optimization()[1]}']

    @classmethod
    def hyperopt_set_simulated_annealing(cls):
        return [f'{SolverName.hyperopt()[0]}4set-{cls.simulated_annealing()[0]}',
                f'{SolverName.hyperopt()[1]}4set-{cls.simulated_annealing()[1]}']

    """ gurobi solver """

    @classmethod
    def gurobi(cls):
        return ['grb', 'Gurobi']


qubo_list = [SolverName.amplify()[0], SolverName.amplify_dwave()[0], SolverName.amplify_leap_hybrid()[0],
             SolverName.gurobi()[0], SolverName.simulated_annealing()[0], SolverName.cpsat_ortools()[0]]
direct_list = [SolverName.hyperopt_bayesian_optimization()[0], SolverName.hyperopt_simulated_annealing()[0]]
onehot_list = qubo_list + [SolverName.mystic()[0], SolverName.hyperopt_onehot_bayesian_optimization()[0],
                           SolverName.hyperopt_onehot_simulated_annealing()[0]]


class OptimizerName:
    @classmethod
    def random_search(cls):
        return ['rg', 'Random Generation']

    @classmethod
    def uniform_gen(cls):
        return ['uni', 'uniform']

    @classmethod
    def random_search_uni(cls):
        return [f'rg-{OptimizerName.uniform_gen()[0]}', f'Random Generation-{OptimizerName.uniform_gen()[1]}']

    @classmethod
    def pyxtal_rs(cls):
        return ['pyx', 'PyX']

    @classmethod
    def pyxtal_rs_uni(cls):
        return [f'pyx-{OptimizerName.uniform_gen()[0]}', f'PyX-{OptimizerName.uniform_gen()[1]}']

    @classmethod
    def wpc_l_rs(cls):
        return ['pyx-wpc', 'PyX-WPC']

    @classmethod
    def wpc_l_rs_uni(cls):
        return [f'pyx-wpc-{OptimizerName.uniform_gen()[0]}', f'PyX-WPC-{OptimizerName.uniform_gen()[1]}']

    @classmethod
    def crystal_params_vector(cls):
        return ['cpv', 'Crystal Parameters']

    @classmethod
    def crystal_params_vector_opt(cls):
        return [f'{OptimizerName.crystal_params_vector()[0]}-fit',
                f'{OptimizerName.crystal_params_vector()[1]}-Fitting']

    @classmethod
    def cryspy_rs(cls):
        return ['cryspy_rs', 'Cryspy_RS']

    @classmethod
    def calypso(cls):
        return ['calypso', 'CALYPSO']

    @classmethod
    def onehot(cls):
        return ['oh', 'Onehot']
    
    @classmethod
    def onehot_site(cls):
        return ['ohs', 'Onehot-site']
    
    @classmethod
    def frac_coords(cls):
        return ['fc', 'FracCoords']


init_set_list = [OptimizerName.random_search()[0], OptimizerName.random_search_uni()[0], OptimizerName.wpc_l_rs()[0],
                 OptimizerName.wpc_l_rs_uni()[0]]
rs_list = init_set_list + [OptimizerName.pyxtal_rs()[0], OptimizerName.pyxtal_rs_uni()[0]]
alpha = 'alpha'
mixing = 'mixfil'
short_trj_step = 50
rlx_type = ['ini', 'rlx', 'i2r']

coords_emb = ['latdis', 'sitedis', 'sitecont']
oh_coords_emb = ['latdis', 'sitedis']
sym_emb = ['oh-csspg', 'oh-spg', 'oh-repr']
enc_obj = ['struct', 'used_atoms']


class RegressionName:
    @classmethod
    def quad(cls):
        return ['quad', 'Quadratic']

    @classmethod
    def fm(cls):
        return ['fm', 'FacMach']

    @classmethod
    def output(cls, model_order, model_name, real_model_name, k_fm):
        return [f'{model_order}-{model_name}{k_fm}',
                f'{model_order}-Model{real_model_name}K{k_fm}']

    @classmethod
    def simple_model(cls, model_order):
        return cls.output(model_order, cls.quad()[0], cls.quad()[1], None)

    @classmethod
    def factorization_machine(cls, model_order, k_fm):
        return cls.output(model_order, cls.fm()[0], cls.fm()[1], k_fm)
    
    @classmethod
    def bocs(cls):
        return ['bocs', 'BO4CS']
    
    @classmethod
    def bhs(cls):
        return ['bhs', 'BayesHS']


direct_posterior = [RegressionName.fm()[0], RegressionName.quad()[0]]


class EnergyEvaluatorName:
    @classmethod
    def m3gnet(cls):
        return 'M3GNet'

    @classmethod
    def chgnet(cls):
        return 'CHGNet'

    @classmethod
    def orb(cls):
        return 'OrbFF'

    @classmethod
    def uma(cls):
        return 'UMA'

    @classmethod
    def uma_s(cls):
        return f'{cls.uma()}-S'

    @classmethod
    def uma_m(cls):
        return f'{cls.uma()}-M'

    @classmethod
    def mattersim(cls):
        return 'MatterSim'

    @classmethod
    def base_simulator(cls):
        return 'Base'

    @classmethod
    def vasp(cls):
        return 'VASP'


class FileLoader:
    def __init__(self, file_dire, load_cache=True, zip_file=False):
        self.file_dire = file_dire
        self.file = []
        self.zip_file = zip_file
        self.load_flag = False
        if load_cache:
            self.load()
        return

    def _zip_file_name(self, file_name):
        if self.zip_file:
            if file_name[-4] != '.bz2':
                file_name += '.bz2'
        return file_name

    def _load(self, file_dire):
        file = []
        if Path(self._zip_file_name(file_dire)).exists():
            file = self._load_file(file_dire, self.zip_file)
            self.load_flag = True
        elif self.zip_file:  # zip but no .bz2, only .pkl
            file_name = self._zip_file_name(file_dire)[:-4]
            if Path(file_name).exists():
                file = self._load_file(file_name, zip_file=False)
                with bz2.open(file_name + '.bz2', 'wb') as f:
                    pickle.dump(file, f)
                self.load_flag = True
        elif not self.zip_file:  # not zip but no .pkl, only .bz2
            file_name = self._zip_file_name(file_dire)
            if Path(file_name).exists():
                file = self._load_file(file_name, zip_file=True)
                with open(file_name[:-4], 'wb') as f:
                    pickle.dump(file, f)
                self.load_flag = True
        return file

    def load(self):
        if not isinstance(self.file_dire, list):
            self.file = self._load(self.file_dire)
        else:
            self.file_dire_needed = []
            for f_d in self.file_dire:
                file = self._load(f_d)
                self.file += file
                self.file_dire_needed.append(f_d)
                self.load_flag = True
        return

    def generate(self):
        raise NotImplementedError

    def save(self, file=None):
        file = self.file if file is None else file
        if not isinstance(self.file_dire, list):
            self._dump_file(file, self.file_dire, self.zip_file)
        else:
            f_d = self.file_dire.pop(0)
            self._dump_file(file, f_d, self.zip_file)
        return self.file

    @staticmethod
    def _dump_file(file, file_name, zip_file):
        if not zip_file:
            with open(file_name, 'wb') as f:
                pickle.dump(file, f)
        else:
            with open(file_name, 'wb') as f:
                pickle.dump(file, f)
            file_name = file_name + '.bz2' if file_name[-4:] != '.bz2' else file_name
            with bz2.open(file_name, 'wb') as f:
                pickle.dump(file, f)
        print(f'Finish saving {file_name} with {zip_file} zipping')
        return

    @staticmethod
    def _load_file(file_name, zip_file):
        if not zip_file:
            with open(file_name, 'rb') as f:
                file = pickle.load(f)
        else:
            try:
                file_name = file_name + '.bz2' if file_name[-4:] != '.bz2' else file_name
                with bz2.open(file_name, 'rb') as f:
                    file = pickle.load(f)
            except (EOFError or OSError):
                file_name = file_name[:-4] if file_name[-4:] == '.bz2' else file_name
                with open(file_name, 'rb') as f:
                    file = pickle.load(f)
                with bz2.open(file_name + '.bz2', 'wb') as f:
                    pickle.dump(file, f)
        print(f'Finish loading {file_name} with {zip_file} zipping')
        return file

    def load_and_zip(self):
        if not isinstance(self.file_dire, list):
            if not Path(self.file_dire + '.bz2').exists():
                data = self._load_file(self.file_dire, zip_file=False)
                self._dump_file(data, self.file_dire, zip_file=True)
        else:
            for f_d in self.file_dire:
                if Path(f_d).exists():
                    if not Path(f_d + '.bz2').exists():
                        data = self._load_file(f_d, zip_file=False)
                        self._dump_file(data, f_d, zip_file=True)
        print('Finish zipping')
        return

    def __call__(self, *args, **kwargs):
        if len(self.file) == 0:
            self.generate()
            self.save()
        return self.file
