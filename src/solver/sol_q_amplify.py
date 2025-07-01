import numpy as np
import yaml
try:
    import amplify
    from amplify import equal_to, sum_poly, VariableGenerator
except ModuleNotFoundError:
    pass

from src.log_analysis import print_or_record
from src.registry import SolverName, RegressionName
from src.regressor.poly_sim import collect_poly_terms, use_interaction_terms
from src.solver.qubo_base import ComplexQuadCombinatorialSolver, ConstraintsBuilder, SolverProxy

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


class AmplifySolver(ComplexQuadCombinatorialSolver):
    def __init__(self, sol_name=SolverName.amplify()[0], real_sol_name=SolverName.amplify()[1],
                 init_const_num_var=None, use_constraints=False, mat_l=None, mat_q=None):
        """ We provide two types of solver initialization approaches. One is to initialize the
            solver with a provided number of variables (gen_solver), and another is to decide
            the number of variables based on the provided weights / quadratic model (init_solver). """
        super().__init__(sol_name=sol_name, real_sol_name=real_sol_name,
                         init_const_num_var=init_const_num_var,
                         use_constraints=use_constraints,
                         mat_l=mat_l, mat_q=mat_q)
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
        print_or_record(self.logger, "Since annealing time is not given, decide it based on # of bits")
        print_or_record(self.logger, f"Based on current # of bits: {num_var}, annealing time is {timeout}")
        return timeout

    def load_timeout_or_read_limit(self, timeout):
        if timeout is not None:
            self.timeout = timeout
            print_or_record(self.logger, f"Receive timeout: {timeout}")
        else:
            self.timeout = self._load_timeout_or_read_limit()
        return

    def load_solver(self):
        self.client = amplify.FixstarsClient()
        fae_token = fae_tk
        assert fae_tk != "-", "Please provide token for Amplify"
        self.client.token = fae_token
        self.client.parameters.timeout = self.timeout
        return

    def _build_q(self, q):
        for term, coef in q.items():
            self.var.quadratic[term[0], term[1]] = coef
        return self.var.to_poly()

    def _build_q_from_sklearn(self, w, num_var, learning_model, interaction_terms):
        if isinstance(w, int) or isinstance(w, float):
            print_or_record(self.logger, "Return void model")
            return amplify.Model()
        if (learning_model == "linear") or (learning_model == 1):
            self.var.linear = w
            print_or_record(self.logger, "Return linear model")
            return self.var.to_poly()
        elif (learning_model == "quadratic") or (learning_model == 2):
            q_mat = np.zeros((num_var, num_var))
            self.var.linear = w[:num_var]
            if not use_interaction_terms(interaction_terms):
                q_mat[np.triu_indices(num_var, 1)] = w[num_var:]  # not including diagonal
            else:
                i, j = np.triu_indices(num_var, 1 + num_var - interaction_terms)
                i += num_var - interaction_terms
                q_mat[i, j] = w[num_var:]
            self.var.quadratic = q_mat
            print_or_record(self.logger, "Return simple quadratic model")
            return self.var.to_poly()

        elif learning_model == RegressionName.fm()[0]:
            var = self.var.variable_array
            lin_terms, inter_terms = build_fm_based_on_amplify(fm_weight_list=w, amp_var=var)
            print_or_record(self.logger, "Return factorization machine model")
            return lin_terms + inter_terms

        else:
            assert isinstance(learning_model, int)
            f = 0
            var = self.var.variable_array
            poly_terms = collect_poly_terms(var, learning_model, interaction_terms)
            for i, term in enumerate(poly_terms):
                f += np.prod(term) * w[i]
            print_or_record(self.logger, "Return high order model")
            return f

    def _build_bqm(self, f):
        bqm = amplify.Model(f)
        return bqm

    def _build_constraints_obj(self, var_list):
        constraints_builder = AmpConstraintsBuilder(
            var_list=var_list, mat_l=self.coef_mat_l, mat_q=self.coef_mat_q)
        return constraints_builder

    def _add_constraints(self, objective, constraints):
        return objective + constraints

    def _solve(self, bqm, num_solves):
        solve_flag = False
        while not solve_flag:
            try:
                result = amplify.solve(model=bqm, client=self.client, num_solves=num_solves)
                print_or_record(self.logger, 'Connect successfully to the solver')
                solve_flag = True
            except RuntimeError:
                print_or_record(self.logger, "Solve fail. Try to solve again.")
                solve_flag = False
        return result

    def solve(self, bqm, seed=None):
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
            print('Did not find the solution')
            q_values = None
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
        self._matrix_to_array()
        self.var = np.array(self.var).reshape(1, -1)
        return

    def _matrix_to_array(self):
        if isinstance(self.var, amplify.Matrix):
            self.var = self.var.variable_array
        return

    def _simple_constraints(self, f, constraints_mat):
        f.append(equal_to(constraints_mat, 0))
        return f

    def _bias_provided_constraints(self, f, constraints_mat, bias):
        f.append(equal_to(constraints_mat, bias))
        return f

    def _weight_provided_constraints(self, f, constraints_mat, bias, weight):
        f.append(equal_to(constraints_mat, bias) * weight)
        return f

    def build_equation(self, constraints_mat):
        eq = []
        return amplify.sum(self._build_equation(eq, constraints_mat))

    def build(self):
        def append_cons_eq(cons_obj, _eq, _num_eq):
            cons = cons_obj.build_constraints()
            _eq.append(self.build_equation(cons))
            _num_eq += cons_obj.num_constraints
            return _eq, _num_eq

        num_eq, eq = 0, []
        if hasattr(self, 'cons_l'):
            eq, num_eq = append_cons_eq(self.cons_l, eq, num_eq)
        if hasattr(self, 'cons_q'):
            eq, num_eq = append_cons_eq(self.cons_q, eq, num_eq)
        print(f'Dealing with {num_eq} constraints')
        return amplify.sum(eq)


class AmplifySolverProxy(SolverProxy):
    def __init__(self, num_var=None, use_constraints=False,
                 mat_l=None, mat_q=None):
        super().__init__(sol_name=SolverName.amplify()[0],
                         real_sol_name=SolverName.amplify()[1],
                         num_var=num_var, use_constraints=use_constraints,
                         mat_l=mat_l, mat_q=mat_q)
        return

    def unwrap(self):
        amp_sol = AmplifySolver(init_const_num_var=self.num_var,
                                use_constraints=self.use_constraints,
                                mat_l=self.coef_mat_l, mat_q=self.coef_mat_q)
        amp_sol = self._transfer_logger(amp_sol)
        amp_sol.load_timeout_or_read_limit(self.timeout)
        amp_sol.load_solver()
        if self.use_constraints:
            print('Finish building constraints')
        return amp_sol


class AmplifyForLeapSolverProxy(SolverProxy):
    def __init__(self, num_var=None, use_constraints=False,
                 mat_l=None, mat_q=None):
        super().__init__(sol_name=SolverName.amplify_leap_hybrid()[0],
                         real_sol_name=SolverName.amplify_leap_hybrid()[1],
                         num_var=num_var, use_constraints=use_constraints,
                         mat_l=mat_l, mat_q=mat_q)
        return

    def unwrap(self):
        amp_sol = AmplifyForLeapHybridSolver(init_const_num_var=self.num_var,
                                             use_constraints=self.use_constraints,
                                             mat_l=self.coef_mat_l, mat_q=self.coef_mat_q)
        amp_sol = self._transfer_logger(amp_sol)
        amp_sol.load_timeout_or_read_limit(self.timeout)
        amp_sol.load_solver()
        if self.use_constraints:
            print('Finish building constraints')
        return amp_sol


class AmplifyForDWaveSolverProxy(SolverProxy):
    def __init__(self, num_var=None, use_constraints=False,
                 mat_l=None, mat_q=None):
        super().__init__(sol_name=SolverName.amplify_dwave()[0],
                         real_sol_name=SolverName.amplify_dwave()[1],
                         num_var=num_var, use_constraints=use_constraints,
                         mat_l=mat_l, mat_q=mat_q)
        return

    def unwrap(self):
        amp_sol = AmplifyForDWaveSolver(init_const_num_var=self.num_var,
                                        use_constraints=self.use_constraints,
                                        mat_l=self.coef_mat_l, mat_q=self.coef_mat_q)
        amp_sol = self._transfer_logger(amp_sol)
        amp_sol.load_timeout_or_read_limit(self.timeout)
        amp_sol.load_solver()
        if self.use_constraints:
            print('Finish building constraints')
        return amp_sol
