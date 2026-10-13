try:
    from ortools.sat.python import cp_model
except ModuleNotFoundError:
    pass


import numpy as np
from itertools import combinations

from crysim.src.solver.sol_base import QuadSolver, ConstraintsBuilder, SolverProxy
from crysim.src.regressor.poly_reg_nn import assemble_fm_weight
from crysim.src.registry import SolverName, RegressionName
from logzero import logger as log_printer


class CPSATSolver(QuadSolver):
    def __init__(self, init_const_num_var=None, use_constraints=False, mat_l=None, mat_q=None):
        super().__init__(sol_name=SolverName.cpsat_ortools()[0], real_sol_name=SolverName.cpsat_ortools()[1],
                         init_const_num_var=init_const_num_var,
                         use_constraints=use_constraints,
                         mat_l=mat_l, mat_q=mat_q, qubo=True)
        self.timeout = None
        return

    def _gen_var(self, num_var):
        self.model = cp_model.CpModel()
        self.var = [self.model.new_int_var(lb=0, ub=1, name=f'x{i}') for i in range(num_var)]
        return

    def _load_timeout_or_read_limit(self):
        num_var = len(self.var)
        if num_var <= 5000:
            timeout = 1500.0
        elif 5000 < num_var <= 8000:
            timeout = 2500.0
        else:
            timeout = 5000.0
        log_printer.info("Since time limit is not given, decide it based on # of bits")
        log_printer.info(f"Based on current # of bits: {num_var}, time limit is {timeout}")
        return timeout

    def _build_q(self, q):
        """ General usage of q_dict, with items as (term_i1, term_i2): coef """
        self.var_quad = []
        q, b = q
        for term1, term2 in q.keys():
            var_quad_term = self.model.new_int_var(lb=0, ub=1, name=f'{self.var[term1].name}{self.var[term2].name}')
            self.var_quad.append(var_quad_term)
            self.model.AddMultiplicationEquality(var_quad_term, [term1, term2])
        log_printer.info("Finish adding quadratic variables")
        bqm = np.array(self.var_quad) @ np.array(list(q.values())) + b
        return bqm

    def _build_q_from_sklearn(self, w, num_var, learning_model, interaction_terms):
        """ General usage of building obj from np.array """
        if isinstance(w, int) or isinstance(w, float):
            log_printer.info("Return void model")
            return 0
        w, b = w
        if (learning_model == "linear") or (learning_model == 1):
            obj = np.array(self.var) @ w + b
            log_printer.info("Return linear model")
            return obj

        else:
            self.var_quad = []
            for term1, term2 in combinations(self.var, r=2):
                var_quad_term = self.model.new_int_var(lb=0, ub=1, name=f'{term1.name}{term2.name}')
                self.var_quad.append(var_quad_term)
                self.model.AddMultiplicationEquality(var_quad_term, [term1, term2])
            log_printer.info("Finish adding quadratic variables")
            if (learning_model == "quadratic") or (learning_model == 2):
                obj = np.array(self.var_quad) @ w[num_var:] + np.array(self.var) @ w[:num_var]
                log_printer.info("Return simple quadratic model")
                return obj + b

            elif learning_model == RegressionName.fm()[0]:
                w_lin, w_inter = w
                w_inter = assemble_fm_weight(w_inter)
                obj = np.array(self.var_quad) @ w_inter + np.array(self.var) @ w_lin
                log_printer.info("Return factorization machine model")
                return obj + b

            else:
                raise Exception("Unexpected learning_model")

    def load_solver(self):
        self.solver = cp_model.CpSolver()
        self.solver.parameters.max_time_in_seconds = self.timeout
        return

    def _build_model(self, f):
        self.model.minimize(f)
        return self.model

    def _build_constraints(self, var_list):
        constraints_builder = CPSATConstraintsBuilder(
            var_list=var_list, mat_l=self.coef_mat_l, mat_q=self.coef_mat_q)
        self.g = constraints_builder.build()
        return self.g

    def _build_constrained_model(self, objective, constraints):
        for c in constraints:
            self.model.add(c)
        self.model.minimize(objective)
        return self.model

    def solve(self, bqm):
        self.solver.solve(bqm)
        q_values = np.array([self.solver.value(i) for i in self.var])
        return q_values, self.solver.objective_value


class CPSATConstraintsBuilder(ConstraintsBuilder):
    def __init__(self, var_list, mat_l, mat_q):
        super().__init__(var_list, mat_l, mat_q)
        return

    def _build_equation(self, mat, bias, weight):
        return mat == bias


class CPSATProxy(SolverProxy):
    def __init__(self, num_var=None, use_constraints=False,
                 mat_l=None, mat_q=None):
        super().__init__(sol_name=SolverName.cpsat_ortools()[0],
                         real_sol_name=SolverName.cpsat_ortools()[1],
                         num_var=num_var, use_constraints=use_constraints,
                         mat_l=mat_l, mat_q=mat_q)
        return

    def _load(self):
        cp_sol = CPSATSolver(init_const_num_var=self.num_var,
                             use_constraints=self.use_constraints,
                             mat_l=self.coef_mat_l, mat_q=self.coef_mat_q)
        return cp_sol
