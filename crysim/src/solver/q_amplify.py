import numpy as np
import yaml

import os
os.environ["DWAVE_CACHE_ENABLED"] = "false"

try:
    import amplify
    from amplify import equal_to, sum_poly, VariableGenerator
except ModuleNotFoundError:
    pass

from logzero import logger as log_printer
from crysim.src.registry import SolverName, RegressionName
from crysim.src.regressor.poly_reg_nn import collect_poly_terms, quad_list_to_q_mat
from crysim.src.solver.sol_base import QuadSolver, ConstraintsBuilder, SolverProxy

with open("token.yml", "r") as tk_dict:
    params = yaml.safe_load(tk_dict)
fae_tk = params["fae_token"]
dw_tk = params["dw_token"]


def build_fm_based_on_amplify(fm_weight_list, amp_var):
    w_lin, w_inter = fm_weight_list
    lin_terms = sum_poly(w_lin.shape[0], lambda ii: w_lin[ii] * amp_var[ii])
    inter_terms = sum_poly(
        w_inter.shape[1],
        lambda k: (
                (sum_poly(w_inter.shape[0], lambda ii: w_inter[ii][k] * amp_var[ii])) ** 2
                - sum_poly(w_inter.shape[0], lambda ii: w_inter[ii][k] ** 2 * amp_var[ii] ** 2)
        )
    ) / 2
    return lin_terms, inter_terms


def fm_objective_function(fm_weight_list, as_dict):
    w_lin, w_inter = fm_weight_list
    var = VariableGenerator()
    var = var.array("Binary", w_lin.shape[0])
    lin_terms, inter_terms = build_fm_based_on_amplify(fm_weight_list, var)
    if as_dict:
        return (lin_terms + inter_terms).as_dict()
    else:
        def sort_values_based_on_keys(_dict):
            _sorted_keys = sorted(_dict.keys())
            return [_dict[k] for k in _sorted_keys]
        w = np.hstack([np.array(sort_values_based_on_keys(lin_terms.as_dict())),
                       np.array(sort_values_based_on_keys(inter_terms.as_dict()))])
        return w


class AmplifySolver(QuadSolver):
    def __init__(self, sol_name=SolverName.amplify()[0], real_sol_name=SolverName.amplify()[1],
                 init_const_num_var=None, use_constraints=False, mat_l=None, mat_q=None):
        """ We provide two types of solver initialization approaches. One is to initialize the
            solver with a provided number of variables (gen_solver), and another is to decide
            the number of variables based on the provided weights / quadratic model (init_solver). """
        super().__init__(sol_name=sol_name, real_sol_name=real_sol_name,
                         init_const_num_var=init_const_num_var,
                         use_constraints=use_constraints,
                         mat_l=mat_l, mat_q=mat_q, qubo=True)
        self.timeout = None
        return

    def _gen_var(self, num_var):
        gen = amplify.VariableGenerator()
        self.var = gen.matrix('Binary', num_var)
        return

    def get_var(self):
        return self.var.variable_array

    def _load_timeout_or_read_limit(self):
        num_var = len(self.get_var())
        if num_var <= 5000:
            timeout = 30000
        elif 5000 < num_var <= 8000:
            # if not self.use_constraints:
            timeout = 50000
            # else:
            #     timeout = 80000
        else:
            # if not self.use_constraints:
            timeout = 80000
            # else:
            #     timeout = 120000
        log_printer.info( "Since annealing time is not given, decide it based on # of bits")
        log_printer.info( f"Based on current # of bits: {num_var}, annealing time is {timeout}")
        return timeout

    def load_solver(self):
        self.client = amplify.FixstarsClient()
        fae_token = fae_tk
        assert fae_tk != "-", "Please provide token for Amplify"
        self.client.token = fae_token
        self.client.parameters.timeout = self.timeout
        return

    def _build_q(self, q):
        q, b = q
        for term, coef in q.items():
            self.var.quadratic[term[0], term[1]] = coef
        return self.var.to_poly() + b

    def _build_q_from_sklearn(self, w, num_var, learning_model, interaction_terms):
        if isinstance(w, int) or isinstance(w, float):
            log_printer.info("Return void model")
            return amplify.Model()
        w, b = w
        if (learning_model == "linear") or (learning_model == 1):
            self.var.linear = w
            log_printer.info("Return linear model")
            return self.var.to_poly() + b
        elif (learning_model == "quadratic") or (learning_model == 2):
            self.var.linear = w[:num_var]
            self.var.quadratic = quad_list_to_q_mat(w[num_var:], num_var=num_var, interaction_terms=interaction_terms)
            log_printer.info("Return simple quadratic model")
            return self.var.to_poly() + b

        elif learning_model == RegressionName.fm()[0]:
            var = self.var.variable_array
            lin_terms, inter_terms = build_fm_based_on_amplify(fm_weight_list=w, amp_var=var)
            log_printer.info("Return factorization machine model")
            return lin_terms + inter_terms + b

        else:
            assert isinstance(learning_model, int)
            var = self.var.variable_array
            poly_terms = collect_poly_terms(var, learning_model, interaction_terms)
            f = sum_poly(np.prod(term) * w[i] for i, term in enumerate(poly_terms)) + b
            # for i, term in enumerate(poly_terms):
            #     f += np.prod(term) * w[i]
            log_printer.info("Return high order model")
            return f

    def _build_model(self, f):
        bqm = amplify.Model(f)
        return bqm

    def _build_constraints(self, var_list):
        constraints_builder = AmpConstraintsBuilder(
            var_list=var_list, mat_l=self.coef_mat_l, mat_q=self.coef_mat_q)
        self.g = constraints_builder.build()
        return self.g
    
    def _build_constrained_model(self, objective, constraints):
        return objective + amplify.sum(constraints)

    def _solve(self, bqm, num_solves):
        solve_flag = False
        while not solve_flag:
            log_printer.info('Try connecting to the solver')
            try:
                result = amplify.solve(model=bqm, client=self.client, num_solves=num_solves)
                solve_flag = True
            except RuntimeError:
                log_printer.info("Solve fail. Try to solve again.")
                solve_flag = False
        return result

    def solve(self, bqm):
        result = self._solve(bqm, num_solves=1)
        try:
            q_values = self.var.variable_array.evaluate(result.best.values)

            # TEST RESULTS: correctly built the BQM
            # energy_values = result.best.objective
            # w_lin = self.var.linear
            # w_q = self.var.quadratic
            # e = 0
            # for i in range(len(w_lin)):
            #     e += q_values[i] * w_lin[i]
            #     for j in range(len(w_lin)):
            #         e += q_values[i] * q_values[j] * w_q[i][j]
            # assert e == energy_values  # pass

        # q_values = self.var.evaluate(result.best.values)
        except RuntimeError:
            log_printer.info('Did not find the solution')
            return None, None
        return q_values, result.best.objective

    def _multi_solve(self, bqm, num_solves):
        result = self._solve(bqm, num_solves)
        q_values = [self.var.variable_array.evaluate(solution.values) for solution in result.solutions]
        return q_values

    def multi_solve(self, bqm, num_solves=100, timeout=500):
        self.client.parameters.timeout = timeout
        return self._multi_solve(bqm, num_solves)


class AmplifyForLeapHybridSolver(AmplifySolver):
    def __init__(self, init_const_num_var=None, use_constraints=False, mat_l=None, mat_q=None):
        super().__init__(sol_name=SolverName.amplify_leap_hybrid()[0],
                         real_sol_name=SolverName.amplify_leap_hybrid()[1],
                         init_const_num_var=init_const_num_var,
                         use_constraints=use_constraints,
                         mat_l=mat_l, mat_q=mat_q)
        return

    def _load_timeout_or_read_limit(self):
        timeout = 5
        return timeout

    def load_solver(self):
        self.client = amplify.LeapHybridCQMSamplerClient()
        self.client.token = dw_tk
        assert dw_tk != "-", "Please provide token for Dwave"
        self.client.parameters.time_limit = self.timeout
        return

    def multi_solve(self, bqm, num_solves=60, timeout=3):
        self.client.parameters.timeout = timeout
        return self._multi_solve(bqm, num_solves)


class AmplifyForDWaveSolver(AmplifySolver):
    def __init__(self, init_const_num_var=None, use_constraints=False, mat_l=None, mat_q=None):
        super().__init__(sol_name=SolverName.amplify_dwave()[0],
                         real_sol_name=SolverName.amplify_dwave()[1],
                         init_const_num_var=init_const_num_var,
                         use_constraints=use_constraints,
                         mat_l=mat_l, mat_q=mat_q)
        return

    def _load_timeout_or_read_limit(self):
        timeout = 1000
        return timeout

    def load_solver(self):
        self.client = amplify.DWaveSamplerClient()
        self.client.token = dw_tk
        assert dw_tk != "-", "Please provide token for Dwave"
        self.client.parameters.num_reads = self.timeout
        return

    def multi_solve(self, bqm, num_solves=60, num_reads=100):
        self.client.parameters.num_reads = num_reads
        return self._multi_solve(bqm, num_solves)


class AmpConstraintsBuilder(ConstraintsBuilder):
    def __init__(self, var_list, mat_l, mat_q):
        super().__init__(var_list, mat_l, mat_q)
        return

    def _load_variables(self, var_list):
        self.var = var_list
        if isinstance(self.var, amplify.Matrix):
            self.var = self.var.variable_array
        self.var = np.array(self.var).reshape(1, -1)
        return

    def _build_equation(self, mat, bias, weight):
        return equal_to(mat, bias) * weight


class AmplifySolverProxy(SolverProxy):
    def __init__(self, num_var=None, use_constraints=False,
                 mat_l=None, mat_q=None):
        super().__init__(sol_name=SolverName.amplify()[0],
                         real_sol_name=SolverName.amplify()[1],
                         num_var=num_var, use_constraints=use_constraints,
                         mat_l=mat_l, mat_q=mat_q)
        return

    def _load(self):
        amp_sol = AmplifySolver(init_const_num_var=self.num_var,
                                use_constraints=self.use_constraints,
                                mat_l=self.coef_mat_l, mat_q=self.coef_mat_q)
        return amp_sol


class AmplifyForLeapSolverProxy(SolverProxy):
    def __init__(self, num_var=None, use_constraints=False,
                 mat_l=None, mat_q=None):
        super().__init__(sol_name=SolverName.amplify_leap_hybrid()[0],
                         real_sol_name=SolverName.amplify_leap_hybrid()[1],
                         num_var=num_var, use_constraints=use_constraints,
                         mat_l=mat_l, mat_q=mat_q)
        return

    def _load(self):
        amp_sol = AmplifyForLeapHybridSolver(init_const_num_var=self.num_var,
                                             use_constraints=self.use_constraints,
                                             mat_l=self.coef_mat_l, mat_q=self.coef_mat_q)
        return amp_sol


class AmplifyForDWaveSolverProxy(SolverProxy):
    def __init__(self, num_var=None, use_constraints=False,
                 mat_l=None, mat_q=None):
        super().__init__(sol_name=SolverName.amplify_dwave()[0],
                         real_sol_name=SolverName.amplify_dwave()[1],
                         num_var=num_var, use_constraints=use_constraints,
                         mat_l=mat_l, mat_q=mat_q)
        return

    def _load(self):
        amp_sol = AmplifyForDWaveSolver(init_const_num_var=self.num_var,
                                        use_constraints=self.use_constraints,
                                        mat_l=self.coef_mat_l, mat_q=self.coef_mat_q)
        return amp_sol
