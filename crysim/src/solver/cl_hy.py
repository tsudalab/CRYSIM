try:
    import hyperopt as hy
except ModuleNotFoundError:
    pass

import numpy as np
from functools import partial
from logzero import logger as log_printer

from crysim.src.registry import SolverName, RegressionName
from crysim.src.solver.sol_base import SolverComponent, QuadSolver, SoftConstraintsBuilder, SolverProxy
from crysim.src.regressor.poly_reg_nn import collect_poly_terms


def load_hy_solver(algo, n_init):
    """ Looking at implementations from http://www.comates.group/links?software=gn_oa """
    if algo == SolverName.simulated_annealing()[0]:
        algo = hy.partial(hy.anneal.suggest)  # simulated annealing
    else:
        algo = hy.partial(hy.tpe.suggest, n_startup_jobs=n_init)  # bayesian optimization
    return algo


def hy_solve(objective_function, var_list, space, algo, max_evals):
    best = hy.fmin(fn=objective_function,
                   space=space,
                   algo=algo,
                   max_evals=max_evals)
    best_x = hy.space_eval(space, best)
    best_x = np.array([best_x[var] for var in var_list])
    return best_x, best


# def _load_timeout_or_read_limit(self):
#     if self.max_step == 'small':
#         return 16
#     elif self.max_step == 'middle':
#         return 64
#     else:
#         return 320


class HyperOptQuadSolver(QuadSolver):
    def __init__(self, algo, init_const_num_var, use_constraints=False, mat_l=None, mat_q=None, lmbda_l=1.0, lmbda_q=1.0):
        assert algo in [SolverName.simulated_annealing()[0], SolverName.bayesian_optimization()[0]]
        if algo == SolverName.simulated_annealing()[0]:
            name = SolverName.hyperopt_onehot_simulated_annealing()[0]
            real_name = SolverName.hyperopt_onehot_simulated_annealing()[1]
        else:
            name = SolverName.hyperopt_onehot_bayesian_optimization()[0]
            real_name = SolverName.hyperopt_onehot_bayesian_optimization()[1]
        super().__init__(init_const_num_var=init_const_num_var, sol_name=name, real_sol_name=real_name, 
                         use_constraints=use_constraints, mat_l=mat_l, mat_q=mat_q, qubo=False)
        self.algo = algo
        self.lmbda_l, self.lmbda_q = lmbda_l, lmbda_q
        return

    def _gen_var(self, num_var):
        # only define keys for variables in hy, which are constructed as a dict
        self.num_var = num_var
        self.var = np.array([f'x{i}' for i in range(num_var)])
        self.space = {f'x{i}': hy.hp.randint(f'x{i}', 2) for i in range(num_var)}
        return

    def _load_timeout_or_read_limit(self):
        if self.num_var <= 5000:
            timelimit = 300
        elif 5000 < self.num_var <= 8000:
            timelimit = 500
        else:
            timelimit = 1000
        log_printer.info("Since max eval is not given, decide it based on # of variables")
        log_printer.info(f"Based on current # of variables: {self.num_var}, max eval is {timelimit}")
        return timelimit
    
    def load_solver(self):
        self.solver = load_hy_solver(algo=self.algo, n_init=max([200, 10 * self.num_var]))
        return
    
    def _build_q(self, x):
        """ General usage of q_dict, with items as (term_i1, term_i2): coef """
        w, b = self.w
        bqm = self._custom_sum(x[self.var[term[0]]] * x[self.var[term[1]]] * coef for term, coef in w.items()) + b
        return bqm

    def _build_q_from_sklearn(self, x):
        """ General usage of building obj from np.array """
        if isinstance(self.w, int) or isinstance(self.w, float):
            log_printer.info("Return void model")
            return 0
        w, b = self.w
        if (self.learning_model == "linear") or (self.learning_model == 1):
            obj = self._custom_sum(x[self.var[i]] * w[i] for i in range(len(w))) + b
            log_printer.info("Return linear model")
            return obj
        elif (self.learning_model == "quadratic") or (self.learning_model == 2):
            poly_terms = collect_poly_terms(x.values(), 2, self.interaction_terms)
            obj = self._custom_sum(np.prod(term) * w[i] for i, term in enumerate(poly_terms)) + b
            log_printer.info("Return simple quadratic model")
            return obj
        
        elif self.learning_model == RegressionName.fm()[0]:
            w_lin, w_inter = w
            lin_terms = sum([w_lin[i] * x[self.var[i]] for i in range(w_lin.shape[0])])
            inter_terms = sum([
                (sum([w_inter[i][k] * x[self.var[i]] for i in range(w_inter.shape[0])])) ** 2
                - sum([w_inter[i][k] ** 2 * x[self.var[i]] ** 2 for i in range(w_inter.shape[0])])
                for k in range(w_inter.shape[1])
            ]) / 2
            obj = lin_terms + inter_terms + b
            log_printer.info("Return factorization machine model")
            return obj

        elif isinstance(self.learning_model, int):
            poly_terms = collect_poly_terms(x.values(), self.learning_model, self.interaction_terms)
            obj = self._custom_sum(np.prod(term) * w[i] for i, term in enumerate(poly_terms)) + b
            log_printer.info("Return high order model")
            return obj
        else:
            raise Exception("Unexpected learning_model")
    
    def gen_solver(self):
        """ This method is used only after variables have been generated. """
        assert self.num_var is not None
        if self.from_sklearn:
            assert self.learning_model is not None
            f = self._build_q_from_sklearn
        else:
            f = self._build_q
        log_printer.info("Get bqm")
        if self.use_constraints:
            f = partial(self._build_constrained_model(objective=f, constraints=self.g))
        else:
            f = self._build_model(f)
        log_printer.info("Finish adding constraints")
        return f
    
    def _build_constraints(self, var_list):
        constraints_builder = SoftConstraintsBuilder(
            var_list=var_list, mat_l=self.coef_mat_l, mat_q=self.coef_mat_q, lmbda_l=self.lmbda_l, lmbda_q=self.lmbda_q)
        self.g = constraints_builder.build()
        return self.g
    
    def _build_constrained_model(self, x, objective, constraints):
        f = objective(x)
        for c in constraints:
            f += c
        return f
    
    def solve(self, f):
        return hy_solve(objective_function=f, var_list=self.var, space=self.space, algo=self.solver, max_evals=self.timeout)


class HyperOptDirectSolver(SolverComponent):
    def __init__(self, algo, init_const_num_var):
        assert algo in [SolverName.simulated_annealing()[0], SolverName.bayesian_optimization()[0]]
        if algo == SolverName.simulated_annealing()[0]:
            name = SolverName.hyperopt_simulated_annealing()[0]
            real_name = SolverName.hyperopt_simulated_annealing()[1]
        else:
            name = SolverName.hyperopt_bayesian_optimization()[0]
            real_name = SolverName.hyperopt_bayesian_optimization()[1]
        self.algo = algo
        self.num_var = init_const_num_var
        super().__init__(sol_name=name, real_sol_name=real_name, use_constraints=False, proxy=False, qubo=False)
        return
    
    def get_space(self, len_lbs, len_ubs, ang_lbs, ang_ubs, compat_spg, max_wpc_idx, num_species_atom):
        assert len(len_lbs) == len(len_ubs) == len(ang_lbs) == len(ang_ubs) == 3
        self.len_lbs, self.len_ubs, self.ang_lbs, self.ang_ubs = len_lbs, len_ubs, ang_lbs, ang_ubs
        self.compat_spg, self.max_wpc_idx = compat_spg, max_wpc_idx
        self.num_species_atom = num_species_atom
        self._gen_var()
        return

    def _gen_var(self):
        self.var = [f'l{i}' for i in range(len(self.len_ubs))] + \
                   [f'a{i}' for i in range(len(self.ang_ubs))] + \
                   ['spg', 'wpc']
        atom_var = []
        for i, num_s in enumerate(self.num_species_atom):
            for j in range(num_s):
                atom_var += [f'e{i}-{j}x', f'e{i}-{j}y', f'e{i}-{j}z']
        self.var += atom_var
        self.space = {f'l{i}': hy.hp.uniform(f'l{i}', self.len_lbs[i], self.len_ubs[i]) for i in range(len(self.len_ubs))}
        self.space.update({f'a{i}': hy.hp.uniform(f'a{i}', self.ang_lbs[i], self.ang_ubs[i]) for i in range(len(self.ang_ubs))})
        self.space.update({'spg': hy.hp.choice('spg', self.compat_spg)})
        self.space.update({'wpc': hy.hp.randint('wpc', 0, self.max_wpc_idx)})
        self.space.update({a_var: hy.hp.uniform(a_var, 0, 1) for a_var in atom_var})
        return

    def load_solver(self):
        self.solver = load_hy_solver(algo=self.algo, n_init=max([200, 10 * self.num_var]))
        return

    def solve(self, f):
        return hy_solve(objective_function=f, var_list=self.var, space=self.space, algo=self.solver, max_evals=self.timeout)


class HyperOptProxy(SolverProxy):
    def __init__(self, algo, is_var_bit, num_var=None, use_constraints=False, 
                 mat_l=None, mat_q=None, lmbda_l=100, lmbda_q=100):
        assert algo in [SolverName.simulated_annealing()[0], SolverName.bayesian_optimization()[0]]
        if is_var_bit:
            if algo == SolverName.simulated_annealing()[0]:
                sol_name, real_sol_name = SolverName.hyperopt_onehot_simulated_annealing()
            else:
                sol_name, real_sol_name = SolverName.hyperopt_onehot_bayesian_optimization()
        else:
            if algo == SolverName.simulated_annealing()[0]:
                sol_name, real_sol_name = SolverName.hyperopt_simulated_annealing()
            else:
                sol_name, real_sol_name = SolverName.hyperopt_bayesian_optimization()
        super().__init__(sol_name=sol_name, real_sol_name=real_sol_name, num_var=num_var, 
                         use_constraints=use_constraints, mat_l=mat_l, mat_q=mat_q, qubo=False)
        self.algo, self.is_var_bit = algo, is_var_bit
        self.lmbda_l, self.lmbda_q = lmbda_l, lmbda_q
        return
    
    def _load(self):
        if self.is_var_bit:
            hy_sol = HyperOptQuadSolver(algo=self.algo, init_const_num_var=self.num_var,
                                        use_constraints=self.use_constraints,
                                        mat_l=self.coef_mat_l, mat_q=self.coef_mat_q,
                                        lmbda_l=self.lmbda_l, lmbda_q=self.lmbda_q)
        else:
            if self.use_constraints:
                log_printer.warning("Direct optimizers do not accept constraints")
            hy_sol = HyperOptDirectSolver(algo=self.algo, init_const_num_var=self.num_var)
        return hy_sol
