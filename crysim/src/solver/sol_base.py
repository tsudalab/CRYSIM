import numpy as np

from crysim.src.registry import Component, RegressionName
from logzero import logger as log_printer
from crysim.src.regressor.poly_reg_nn import build_fm, quad_list_to_q_mat, collect_poly_terms


class SolverComponent(Component):
    def __init__(self, sol_name: str, real_sol_name: str, use_constraints: bool,
                 proxy: bool, qubo: bool):
        name = sol_name if not use_constraints else f'{sol_name}_c'
        real_name = real_sol_name if not use_constraints else f'{real_sol_name}_constrained'
        self.use_constraints = use_constraints
        self.proxy = proxy
        self.qubo = qubo
        super().__init__(name, real_name)
        return

    """ basic """

    def _gen_var(self, *args, **kwargs):
        self.var = None
        return

    def get_var(self):
        return self.var

    def load_solver(self):
        raise NotImplementedError
    
    def gen_solver(self, *args, **kwargs):
        raise NotImplementedError
    
    def _build_model(self, objective):
        return objective

    def solve(self, *args, **kwargs):
        raise NotImplementedError

    """ time out related parameters """

    def _load_timeout_or_read_limit(self):
        return

    def load_timeout_or_read_limit(self, timeout):
        if timeout is not None:
            self.timeout = timeout
            log_printer.info(f"Receive timeout: {timeout}")
        else:
            self.timeout = self._load_timeout_or_read_limit()
        return

    """ constraints """

    def _build_constraints(self, var_list):
        self.g = None
        return

    def _build_constrained_model(self, objective, constraints):
        raise NotImplementedError


class QuadSolver(SolverComponent):
    def __init__(self, sol_name, real_sol_name, init_const_num_var, use_constraints, mat_l, mat_q, qubo, custom_sum=sum, **kwargs):
        super().__init__(sol_name=sol_name, real_sol_name=real_sol_name, use_constraints=use_constraints, proxy=False, qubo=qubo)
        self._custom_sum = custom_sum

        self.num_var = init_const_num_var
        self.logger = None
        
        self.from_sklearn = False
        self.learning_model = None
        self.interaction_terms = None

        if init_const_num_var is not None:
            log_printer.info("Solver initialization starts based on the "
                  "passed initial constant number of variables. ")
            self._gen_var(init_const_num_var)
            if use_constraints:
                self.coef_mat_l, self.coef_mat_q = mat_l, mat_q
                assert ((self.coef_mat_l is not None) or (self.coef_mat_q is not None)), \
                    'Coefficient matrices for linear and quadratic constraints are not provided. ' \
                    'Call the get_constraints_coefficient_matrix method before adding constraints. '
                log_printer.info('Optimizing with constraints')
                self._build_constraints(self.get_var())

        return

    def _build_q(self, q):
        """ General usage of q_dict, with items as (term_i1, term_i2): coef """
        q, b = q
        bqm = self._custom_sum(self.var[term[0]] * self.var[term[1]] * coef for term, coef in q.items()) + b
        return bqm

    def _build_q_from_sklearn(self, w, num_var, learning_model, interaction_terms):
        """ General usage of building obj from np.array """
        if isinstance(w, int) or isinstance(w, float):
            log_printer.info("Return void model")
            return 0
        w, b = w
        if (learning_model == "linear") or (learning_model == 1):
            obj = self.var @ w + b
            log_printer.info("Return linear model")
            return obj
        elif (learning_model == "quadratic") or (learning_model == 2):
            q_mat = quad_list_to_q_mat(w[num_var:], num_var=num_var, interaction_terms=interaction_terms)
            obj = self.var @ q_mat @ self.var + self.var @ w[:num_var] + b
            log_printer.info("Return simple quadratic model")
            return obj

        elif learning_model == RegressionName.fm()[0]:
            w_lin, w_inter = w
            lin_terms, inter_terms = build_fm(w_lin=w_lin, w_inter=w_inter, var=self.var, custom_sum=self._custom_sum)
            obj = lin_terms + inter_terms + b
            log_printer.info("Return factorization machine model")
            return obj

        elif isinstance(learning_model, int):
            poly_terms = collect_poly_terms(self.var, learning_model, interaction_terms)
            obj = self._custom_sum(np.prod(term) * w[i] for i, term in enumerate(poly_terms)) + b
            log_printer.info("Return high order model")
            return obj
        else:
            raise Exception("Unexpected learning_model")
        
    def get_model_params(self, from_sklearn, learning_model, interaction_terms):
        self.from_sklearn = from_sklearn
        self.learning_model, self.interaction_terms = learning_model, interaction_terms
        return
    
    def get_obj_params(self, w):
        self.w = w
        return
    
    def init_solver(self, num_var=None):
        """ This method is used when the number of variables is not provided until now. """
        if self.from_sklearn:
            assert (num_var is not None) and (self.learning_model is not None)
            self._gen_var(num_var)
            f = self._build_q_from_sklearn(w=self.w, num_var=num_var, learning_model=self.learning_model,
                                           interaction_terms=self.interaction_terms)
        else:
            num_var = np.max([i[0] for i in list(self.w.keys())]) + 1
            self._gen_var(num_var)
            f = self._build_q(q=self.w)
        if self.use_constraints:
            # If the coefficient matrices are not derived,
            # use self.get_constraints_coefficient_matrix() before adding constraints.
            # If the number of variables is not decided (init_const_num_var is None),
            # usually the matrices are also unlikely to have been obtained.
            # Therefore, the constraints builder will not be initialized
            # when the Solver (the class) is initialized.
            # Therefore, it should be initialized here (in the next line).
            cons = self._build_constraints(var_list=self.var)
            f = self._build_constrained_model(objective=f, constraints=cons)
        else:
            f = self._build_model(f)
        return f

    def gen_solver(self):
        """ This method is used only after variables have been generated. """
        assert self.num_var is not None
        if self.from_sklearn:
            assert self.learning_model is not None
            f = self._build_q_from_sklearn(w=self.w, num_var=self.num_var, learning_model=self.learning_model,
                                           interaction_terms=self.interaction_terms)
        else:
            f = self._build_q(q=self.w)
        log_printer.info("Get bqm")
        if self.use_constraints:
            f = self._build_constrained_model(objective=f, constraints=self.g)
        else:
            f = self._build_model(f)
        log_printer.info("Finish adding constraints")
        return f


""" Constraints builders """


class ConstraintsBuilder:
    def __init__(self, var_list, mat_l, mat_q):
        self.lmbda_l, self.lmbda_q = 1.0, 1.0
        self._load_variables(var_list)
        self._load_cons_mat(mat_l, mat_q)
        return

    def _load_variables(self, var_list):
        self.var = var_list
        return

    def _load_cons_mat(self, mat_l, mat_q):
        if mat_l is not None:
            mat_l, bias_l, weight_l = mat_l
            num_l = mat_l.shape[0]
            var_l = np.tile(self.var, (num_l, 1))  # (num_l, num_var)
            self.cons_l = np.sum(mat_l * var_l, axis=1), bias_l, weight_l
        if mat_q is not None:
            mat_q, bias_q, weight_q = mat_q
            num_q = mat_q.shape[0]
            cons_q_mat = np.sum([self.var @ quad_list_to_q_mat(quad_coef_list=mat_q[i],
                                                               num_var=self.var.shape[-1],
                                                               interaction_terms=None) @ self.var
                                 for i in range(num_q)], axis=1)
            self.cons_q = cons_q_mat, bias_q, weight_q
        return

    def build(self):
        constraints = []
        num_eq = 0
        if hasattr(self, 'cons_l'):
            for i in range(self.cons_l[0].shape[0]):
                if self.lmbda_l != 1:
                    constraints.append(self._build_equation(self.cons_l[0][i], self.cons_l[1][i], self.cons_l[2][i]) * self.lmbda_l)
                else:
                    constraints.append(self._build_equation(self.cons_l[0][i], self.cons_l[1][i], self.cons_l[2][i]))
            num_eq += self.cons_l[0].shape[0]
        if hasattr(self, 'cons_q'):
            for i in range(self.cons_q[0].shape[0]):
                if self.lmbda_q != 1:
                    constraints.append(self._build_equation(self.cons_q[0][i], self.cons_q[1][i], self.cons_q[2][i]) * self.lmbda_q)
                else:
                    constraints.append(self._build_equation(self.cons_q[0][i], self.cons_q[1][i], self.cons_q[2][i]))
            num_eq += self.cons_q[0].shape[0]
        log_printer.info(f'Dealing with {num_eq} constraints')
        return constraints

    def _build_equation(self, mat, bias, weight):
        return (mat - bias) * weight


class SoftConstraintsBuilder(ConstraintsBuilder):
    def __init__(self, var_list, mat_l, mat_q, lmbda_l, lmbda_q):
        super().__init__(var_list, mat_l, mat_q)
        self.lmbda_l = lmbda_l
        self.lmbda_q = lmbda_q
        return

    def _build_equation(self, mat, bias, weight):
        return (mat - bias) ** 2 * weight


""" proxy builder """


class SolverProxy(SolverComponent):
    """ Obtaining some information that is necessary for Solver to be instantiated, but cannot be obtained
        as the strategy itself when composing the objects from different strategies. """

    def __init__(self, sol_name, real_sol_name, num_var, use_constraints, mat_l, mat_q, qubo=True):
        super().__init__(sol_name, real_sol_name, use_constraints, proxy=True, qubo=qubo)
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

    def remove_constraints(self):
        self.use_constraints = False
        self.coef_mat_l, self.coef_mat_q = None, None
        return

    def _load(self):
        raise NotImplementedError

    def unwrap(self):
        """ Initialize the Solver class and return the solver object. """
        sol_real = self._load()
        sol_real.load_timeout_or_read_limit(self.timeout)
        sol_real.load_solver()
        if self.use_constraints:
            log_printer.info('Finish building constraints')
        return sol_real
