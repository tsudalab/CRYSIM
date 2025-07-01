import numpy as np
try:
    import hyperopt as hy
except ModuleNotFoundError:
    pass

from src.registry import SolverName, RegressionName
from src.solver.qubo_base import SolverComponent
from src.regressor.poly_sim import collect_poly_terms


class ClassicalSolver(SolverComponent):
    def __init__(self, sol_name, real_sol_name):
        super().__init__(sol_name=sol_name, real_sol_name=real_sol_name, use_constraints=False,
                         proxy=False, qubo=False)
        self.num_var = None
        self.learning_model = None
        self.q = None
        self.seed = None
        self.q_from_dict = None
        self.logger = None
        return

    def get_logger(self, logger):
        self.logger = logger
        return

    def _get_params(self, num_var, learning_model, q_from_dict, interaction_terms, seed):
        self.num_var = num_var
        self.learning_model = learning_model
        self.seed = seed
        self.q_from_dict = q_from_dict
        self.interaction_terms = interaction_terms
        return

    def get_params(self, *args):
        raise NotImplementedError

    def get_q(self, q):
        """ Get coefficients of the objective function """
        self.q = q
        return

    def solve(self, *args):
        raise NotImplementedError


class HyperSolver(ClassicalSolver):
    def __init__(self, algo, sol_name, real_sol_name):
        assert algo in [SolverName.simulated_annealing()[0],
                        SolverName.bayesian_optimization()[0]]
        self.algo = algo
        self.max_step = None
        self.n_init = 200
        self.q = None
        self._load_solver()
        super().__init__(sol_name=sol_name, real_sol_name=real_sol_name)
        return

    def _gen_var(self, *args):
        self.space, self.var = None, None
        return

    def _get_large_max_step(self):
        self.max_step = 320
        return

    def _get_middle_max_step(self):
        self.max_step = 64
        return

    def _get_small_max_step(self):
        self.max_step = 16
        return

    def get_timeout_or_read_limit(self, max_step):
        if isinstance(max_step, str):
            if max_step == 'small':
                self._get_small_max_step()
            elif max_step == 'middle':
                self._get_middle_max_step()
            elif max_step == 'large':
                self._get_large_max_step()
            else:
                raise NotImplementedError
        else:
            self.max_step = max_step
        return

    def _load_solver(self):
        """ Looking at implementations from http://www.comates.group/links?software=gn_oa """
        if self.algo == SolverName.simulated_annealing()[0]:
            self.algo = hy.partial(hy.anneal.suggest)  # simulated annealing
        else:
            self.algo = hy.partial(hy.tpe.suggest, n_startup_jobs=self.n_init)  # bayesian optimization
        return

    def _non_dict_of(self, x):
        f = 0
        poly_terms = collect_poly_terms(var=self.var, order=self.learning_model,
                                        interaction_terms=self.interaction_terms)
        for i, term in enumerate(poly_terms):
            var = np.prod([x[var_label] for var_label in term])
            f += self.q[i] * var
        return f

    def _dict_of(self, x):
        f = 0
        for term, coef in self.q.items():
            var = np.prod([x[self.var[var_label]] for var_label in term])
            f += coef * var
        return f

    def fm_objective_function(self, x):
        w_lin, w_inter = self.q
        lin_terms = sum([w_lin[i] * x[self.var[i]] for i in range(w_lin.shape[0])])
        inter_terms = sum([
            (sum([w_inter[i][k] * x[self.var[i]] for i in range(w_inter.shape[0])])) ** 2
            - sum([w_inter[i][k] ** 2 * x[self.var[i]] ** 2 for i in range(w_inter.shape[0])])
            for k in range(w_inter.shape[1])
        ]) / 2
        return lin_terms + inter_terms

    def objective_function(self, x):
        if self.learning_model == RegressionName.fm()[0]:
            f = self.fm_objective_function(x)
            # Test results: f == f2
            # from src.regressor.poly_sim import fm_objective_function
            # self.q = fm_objective_function(self.q, as_dict=True)
            # f2 = self._dict_of(x)
        else:
            if not self.q_from_dict:
                f = self._non_dict_of(x)
            else:
                f = self._dict_of(x)
        return {'loss': f, 'status': hy.STATUS_OK}

    def solve(self, objective_function=None):
        trials = hy.Trials()
        objective_function = self.objective_function if objective_function is None \
            else objective_function
        best = hy.fmin(fn=objective_function,
                       space=self.space,
                       algo=self.algo,
                       max_evals=self.max_step,
                       trials=trials,
                       rstate=np.random.default_rng(self.seed))
        best_x = hy.space_eval(self.space, best)
        best_x = np.array([best_x[var] for var in self.var])
        return best_x, best


class HyperOneHotSolver(HyperSolver):
    def __init__(self, algo):
        if algo == SolverName.simulated_annealing()[0]:
            name = SolverName.hyperopt_onehot_simulated_annealing()[0]
            real_name = SolverName.hyperopt_onehot_simulated_annealing()[1]
        else:
            name = SolverName.hyperopt_onehot_bayesian_optimization()[0]
            real_name = SolverName.hyperopt_onehot_bayesian_optimization()[1]
        super().__init__(algo=algo, sol_name=name, real_sol_name=real_name)
        return

    def _gen_var(self, *args):
        self.var = np.array([f'x{i}' for i in range(self.num_var)])
        print(f'{self.name} is dealing with {self.num_var} variables')
        self.space = {f'x{i}': hy.hp.randint(f'x{i}', 2)
                      for i in range(self.num_var)}
        return

    def get_params(self, num_var, learning_model, q_from_dict, interaction_terms, seed):
        self._get_params(num_var=num_var, learning_model=learning_model,
                         q_from_dict=q_from_dict, interaction_terms=interaction_terms, seed=seed)
        self._gen_var()
        return


class HyperContinuousSolver(HyperSolver):
    def __init__(self, algo):
        if algo == SolverName.simulated_annealing()[0]:
            name = SolverName.hyperopt_simulated_annealing()[0]
            real_name = SolverName.hyperopt_simulated_annealing()[1]
        else:
            name = SolverName.hyperopt_bayesian_optimization()[0]
            real_name = SolverName.hyperopt_bayesian_optimization()[1]
        super().__init__(algo, sol_name=name, real_sol_name=real_name)
        self.lat_dim, self.ang_dim, self.pos_dim = None, None, None
        self.num_var = None
        self.spg_list = None
        self.e_cal = None
        return

    def _gen_var(self, ll, ul):
        self.var = [f'l{i}' for i in range(self.lat_dim)] + \
                   [f'a{i}' for i in range(self.ang_dim)] + \
                   [f'x{i}' for i in range(self.pos_dim)] + \
                   ['spg', 'wp']
        self.space = {self.var[i]: hy.hp.uniform(self.var[i], ll[i], ul[i])
                      for i in range(self.num_var - 2)}
        if self.spg_list is None:
            self.space.update(
                {self.var[i]: hy.hp.randint(self.var[i], ll[i], ul[i] + 1)
                 for i in range(self.num_var - 2, self.num_var)})
        else:
            self.space.update({'spg': hy.hp.choice('spg', self.spg_list)})
            self.space.update({'wp': hy.hp.randint('wp', ll[-1], ul[-1] + 1)})
        return

    def get_params(self, lat_dim, ang_dim, pos_dim, seed, ll, ul,
                   learning_model=None, q_from_dict=None, spg_list=None, interaction_terms=None):
        self.lat_dim, self.ang_dim, self.pos_dim = lat_dim, ang_dim, pos_dim
        self.spg_list = spg_list
        self._get_params(num_var=self.lat_dim + self.ang_dim + self.pos_dim + 2,
                         learning_model=learning_model, seed=seed, q_from_dict=q_from_dict,
                         interaction_terms=interaction_terms)
        self._gen_var(ll=ll, ul=ul)
        return


class HyperSetSolver(HyperSolver):
    def __init__(self, algo):
        if algo == SolverName.simulated_annealing()[0]:
            name = SolverName.hyperopt_set_simulated_annealing()[0]
            real_name = SolverName.hyperopt_set_simulated_annealing()[1]
        else:
            name = SolverName.hyperopt_set_bayesian_optimization()[0]
            real_name = SolverName.hyperopt_set_bayesian_optimization()[1]
        super().__init__(algo, sol_name=name, real_sol_name=real_name)
        self.solution_set = None
        return

    def _gen_var(self):
        self.var = ['solution']
        self.space = {self.var[0]: hy.hp.choice(self.var[0], self.solution_set)}
        return

    def get_params(self, solution_set, learning_model, q_from_dict, interaction_terms, seed):
        self.solution_set = solution_set
        self._get_params(num_var=1, learning_model=learning_model,
                         q_from_dict=q_from_dict, seed=seed,
                         interaction_terms=interaction_terms)
        self._gen_var()
        return

    def _non_dict_of(self, x):
        f = 0
        x = x['solution']
        bits = np.arange(len(x))
        poly_terms = collect_poly_terms(bits, order=self.learning_model,
                                        interaction_terms=self.interaction_terms)
        for i, term in enumerate(poly_terms):
            var = np.prod([x[var_label] for var_label in term])
            f += self.q[i] * var
        return f
