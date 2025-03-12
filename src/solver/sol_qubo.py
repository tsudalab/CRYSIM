import numpy as np
import amplify
import yaml
from amplify import equal_to, sum_poly

from src.log_analysis import print_or_record
from src.registry import Component, SolverName, RegressionName
from src.regressor.poly_sim import collect_poly_terms, use_interaction_terms, build_fm_based_on_amplify

with open("token.yml", "r") as tk_dict:
    params = yaml.safe_load(tk_dict)
fae_tk = params["fae_token"]
dw_tk = params["dw_token"]


class SolverComponent(Component):
    def __init__(self, sol_name: str, real_sol_name: str, use_constraints: bool,
                 proxy: bool, qubo: bool):
        name = sol_name if not use_constraints else f'{sol_name}_c'
        real_name = real_sol_name if not use_constraints else f'{real_sol_name}_constrained'
        self.proxy = proxy
        self.qubo = qubo
        super().__init__(name, real_name)
        return


class CombinatorialSolver(SolverComponent):
    def __init__(self, sol_name, real_sol_name, use_constraints):
        super().__init__(sol_name, real_sol_name, use_constraints, proxy=False, qubo=True)
        self.logger = None
        return

    def get_logger(self, logger):
        self.logger = logger
        return

    def load_solver(self):
        raise NotImplementedError

    def init_solver(self, *args, **kwargs):
        return

    def solve(self, *args, **kwargs):
        raise NotImplementedError


class ComplexCombinatorialSolver(CombinatorialSolver):
    def __init__(self, sol_name, real_sol_name, init_const_num_var,
                 use_constraints, mat_l, mat_q):
        super().__init__(sol_name=sol_name, real_sol_name=real_sol_name,
                         use_constraints=use_constraints)
        self.init_const_num_var = init_const_num_var
        self.use_constraints = use_constraints

        if init_const_num_var is not None:
            print("Solver initialization starts based on the "
                  "passed initial constant number of variables. ")
            self._gen_var(init_const_num_var)
            if use_constraints:
                self.coef_mat_l, self.coef_mat_q = mat_l, mat_q
                # self.g = self._add_constraints(var_list=self.var)
                self.g = self._build_constraints(self.get_var())
        return

    def _gen_var(self, num_var):
        self.var = None
        return

    def get_var(self):
        return self.var

    def load_solver(self):
        raise NotImplementedError

    def _constraints_assert(self):
        assert ((self.coef_mat_l is not None) or (self.coef_mat_q is not None)), \
            'Coefficient matrices for linear and quadratic constraints are not provided. ' \
            'Call the get_constraints_coefficient_matrix method before adding constraints. '
        print('Optimizing with constraints')
        return

    def _build_constraints_obj(self, var_list):
        raise NotImplementedError

    def _build_constraints(self, var_list):
        self._constraints_assert()
        constraints_builder = self._build_constraints_obj(var_list)
        constraints = constraints_builder.build()
        return constraints

    def _add_constraints(self, objective, constraints):
        raise NotImplementedError

    def _build_q(self, q):
        raise NotImplementedError

    def _build_q_from_sklearn(self, w, num_var, learning_model, interaction_terms):
        raise NotImplementedError

    def _build_bqm(self, f):
        return None

    def init_solver(self, q, num_var=None, learning_model=None, from_sklearn=False,
                    interaction_terms=None):
        """ This method is used when the number of variables is not provided until now. """
        if from_sklearn:
            assert (num_var is not None) and (learning_model is not None)
            self._gen_var(num_var)
            f = self._build_q_from_sklearn(w=q, num_var=num_var, learning_model=learning_model,
                                           interaction_terms=interaction_terms)
        else:
            num_var = np.max([i[0] for i in list(q.keys())]) + 1
            self._gen_var(num_var)
            f = self._build_q(q=q)
        if self.use_constraints:
            # If the coefficient matrices are not derived,
            # use self.get_constraints_coefficient_matrix() before adding constraints.
            # If the number of variables is not decided (init_const_num_var is None),
            # usually the matrices are also unlikely to have been obtained.
            # Therefore, the constraints builder will not be initialized
            # when the Solver (the class) is initialized.
            # Therefore, it should be initialized here (in the next line).
            cons = self._build_constraints(var_list=self.var)
            f = self._add_constraints(objective=f, constraints=cons)
        else:
            f = self._build_bqm(f)
        return f

    def gen_solver(self, q, learning_model=None, from_sklearn=False, interaction_terms=None):
        """ This method is used only after variables have been generated. """
        assert self.init_const_num_var is not None
        if from_sklearn:
            assert learning_model is not None
            f = self._build_q_from_sklearn(w=q, num_var=self.init_const_num_var,
                                           learning_model=learning_model,
                                           interaction_terms=interaction_terms)
        else:
            f = self._build_q(q=q)
        print_or_record(self.logger, "Get bqm")
        if self.use_constraints:
            f = self._add_constraints(objective=f, constraints=self.g)
        else:
            f = self._build_bqm(f)
        print_or_record(self.logger, "Finish adding constraints")
        return f

    def solve(self, *args, **kwargs):
        raise NotImplementedError


class AmplifySolver(ComplexCombinatorialSolver):
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


def is_not_single_mat(coef_mat):
    return True if isinstance(coef_mat, tuple) or isinstance(coef_mat, list) else False


class ConstraintsEquationsBuilder:
    def __init__(self, var, constraints_coefficient_matrix):
        self.var = var
        self.coef_mat = constraints_coefficient_matrix
        self._generate_var_mat()
        return

    @property
    def i_am_not_single_mat(self):
        return is_not_single_mat(self.coef_mat)

    @property
    def num_constraints(self):
        return self.coef_mat[0].shape[0] if self.i_am_not_single_mat else self.coef_mat.shape[0]

    def _generate_var_mat(self):
        self.var_mat = np.array([0, 0])
        return

    def build_constraints(self):
        # import time
        # a = time.process_time()
        if not self.i_am_not_single_mat:
            cont_mat = np.sum(np.multiply(self.coef_mat, self.var_mat), axis=1)
        elif len(self.coef_mat) == 2:
            cont_mat = (np.sum(np.multiply(self.coef_mat[0], self.var_mat), axis=1),
                        self.coef_mat[1])
        elif len(self.coef_mat) == 3:
            cont_mat = (np.sum(np.multiply(self.coef_mat[0], self.var_mat), axis=1),
                        self.coef_mat[1], self.coef_mat[2])
        else:
            raise NotImplementedError
        # print(f'bc: {time.process_time() - a}')
        return cont_mat


class LinearConstraints(ConstraintsEquationsBuilder):
    def __init__(self, var, constraints_coefficient_matrix):
        super().__init__(var, constraints_coefficient_matrix)
        return

    def _generate_var_mat(self):
        self.var_mat = np.tile(self.var, (self.num_constraints, 1))  # (num_l, num_var)
        return


class QuadraticConstraints(ConstraintsEquationsBuilder):
    def __init__(self, var, constraints_coefficient_matrix):
        super().__init__(var, constraints_coefficient_matrix)
        return

    def _generate_var_mat(self):
        assert self.var.shape[1] == self.num_constraints
        self.var_mat = np.tile(self.var, (self.num_constraints, 1))  # (num_q, num_var)
        # var_q = self.var[var_q_ids] if var_q_ids is not None else self.var[:, -self.num_q:]
        var_q = self.var.reshape(-1, 1)  # (num_var, 1)
        self.var_mat = self.var_mat * var_q
        return

    def supplement(self, var_all_num):
        var_q_num = self.var_mat.shape[1]
        eq_q_num = self.var_mat.shape[0]
        self.var_mat = np.concatenate(
            [np.zeros(shape=(eq_q_num, (var_all_num - var_q_num))), self.var_mat], axis=1)
        return


class ConstraintsBuilder:
    def __init__(self, var_list, mat_l, mat_q):
        self._load_variables(var_list)
        self._load_cons_mat(mat_l, mat_q)
        return

    def _load_variables(self, var_list):
        self.var = None
        return

    def _load_cons_mat(self, mat_l, mat_q):
        if mat_l is not None:
            self.cons_l = LinearConstraints(self.var, mat_l)
        if mat_q is not None:
            num_q = mat_q[0].shape[0] if is_not_single_mat(mat_q) else mat_q.shape[0]
            var_q = self.var[:, -num_q:]
            self.cons_q = QuadraticConstraints(var_q, mat_q)
            self.cons_q.supplement(self.var.shape[1])
        return

    def _build_equation(self, eq, constraints_mat):
        if constraints_mat is None:
            return 0

        if isinstance(constraints_mat, tuple):
            if len(constraints_mat) == 2:
                for i in range(len(constraints_mat[0])):
                    eq = self._bias_provided_constraints(eq, constraints_mat[0][i],
                                                         constraints_mat[1][i])
            elif len(constraints_mat) == 3:
                for i in range(len(constraints_mat[0])):
                    eq = self._weight_provided_constraints(eq, constraints_mat[0][i],
                                                           constraints_mat[1][i],
                                                           constraints_mat[2][i])
            else:
                raise NotImplementedError
        else:
            for constraint in constraints_mat:
                eq = self._simple_constraints(eq, constraint)
        return eq

    def build_equation(self, constraints_mat):
        raise NotImplementedError

    def _simple_constraints(self, f, constraints_mat):
        return None

    def _bias_provided_constraints(self, f, constraints_mat, bias):
        return None

    def _weight_provided_constraints(self, f, constraints_mat, bias, weight):
        return None

    def build(self):
        raise NotImplementedError


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
        def append_cons_eq(cons_obj, _eq, num_eq):
            cons = cons_obj.build_constraints()
            _eq.append(self.build_equation(cons))
            num_eq += cons_obj.num_constraints
            return _eq, num_eq

        _num_eq, eq = 0, []
        if hasattr(self, 'cons_l'):
            eq, _num_eq = append_cons_eq(self.cons_l, eq, _num_eq)
        if hasattr(self, 'cons_q'):
            eq, _num_eq = append_cons_eq(self.cons_q, eq, _num_eq)
        print(f'Dealing with {_num_eq} constraints')
        return amplify.sum(eq)


class SolverProxy(SolverComponent):
    """ Obtaining some information that is necessary for Solver to be instantiated, but cannot be obtained
        as the strategy itself when composing the objects from different strategies. """

    def __init__(self, sol_name, real_sol_name,
                 num_var, use_constraints, mat_l, mat_q):
        super().__init__(sol_name, real_sol_name, use_constraints, proxy=True, qubo=True)
        self.num_var = num_var
        self.use_constraints = use_constraints
        self.coef_mat_l, self.coef_mat_q = mat_l, mat_q
        self.logger = None
        self.timeout = None
        return

    def get_num_var(self, num_var):
        self.num_var = num_var
        return

    def get_timeout_or_read_limit(self, timeout):
        self.timeout = timeout
        return

    def get_constraints_coefficient_matrix(self, coef_mat_l, coef_mat_q):
        self.use_constraints = True
        self.coef_mat_l, self.coef_mat_q = coef_mat_l, coef_mat_q
        return

    def get_logger(self, logger):
        self.logger = logger
        return

    def remove_constraints(self):
        self.use_constraints = False
        self.coef_mat_l, self.coef_mat_q = None, None
        return

    def _transfer_logger(self, sol_real):
        sol_real.get_logger(self.logger)
        return sol_real

    def unwrap(self):
        """ Initialize the Solver class and return the solver object. """
        raise NotImplementedError


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
