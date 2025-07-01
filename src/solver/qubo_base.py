import numpy as np

from src.registry import Component
from src.log_analysis import print_or_record


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
    def __init__(self, sol_name, real_sol_name, init_const_num_var, use_constraints):
        super().__init__(sol_name=sol_name, real_sol_name=real_sol_name,
                         use_constraints=use_constraints)
        self.init_const_num_var = init_const_num_var
        self.use_constraints = use_constraints

        if init_const_num_var is not None:
            print("Solver initialization starts based on the "
                  "passed initial constant number of variables. ")
            self._gen_var(init_const_num_var)
        return

    def _gen_var(self, num_var):
        self.var = None
        return

    def get_var(self):
        return self.var

    def load_solver(self):
        raise NotImplementedError

    def _build_constraints_obj(self, var_list):
        raise NotImplementedError

    def _build_constraints_base(self, var_list):
        constraints_builder = self._build_constraints_obj(var_list)
        constraints = constraints_builder.build()
        self.g = constraints
        return constraints

    def _build_constraints(self, var_list):
        return self._build_constraints_base(var_list)

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


class ComplexQuadCombinatorialSolver(ComplexCombinatorialSolver):
    def __init__(self, sol_name, real_sol_name, init_const_num_var,
                 use_constraints, mat_l, mat_q):
        super().__init__(sol_name=sol_name, real_sol_name=real_sol_name,
                         use_constraints=use_constraints, init_const_num_var=init_const_num_var)

        if init_const_num_var is not None:
            if use_constraints:
                self.coef_mat_l, self.coef_mat_q = mat_l, mat_q
                # self.g = self._add_constraints(var_list=self.var)
                self._build_constraints(self.get_var())
        return

    def _constraints_assert(self):
        assert ((self.coef_mat_l is not None) or (self.coef_mat_q is not None)), \
            'Coefficient matrices for linear and quadratic constraints are not provided. ' \
            'Call the get_constraints_coefficient_matrix method before adding constraints. '
        print('Optimizing with constraints')
        return

    def _build_constraints(self, var_list):
        self._constraints_assert()
        return self._build_constraints_base(var_list)


""" Constraints builders """


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


""" proxy builder """


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
