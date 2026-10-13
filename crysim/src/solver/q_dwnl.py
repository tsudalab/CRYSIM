try:
    import neal as nl
    import dimod
except ModuleNotFoundError:
    pass

from crysim.src.registry import SolverName, RegressionName
from logzero import logger as log_printer
from crysim.src.solver.sol_base import QuadSolver, SoftConstraintsBuilder, SolverProxy
from crysim.src.regressor.poly_reg_nn import quad_list_to_q_mat, build_fm


class DWaveNealSolver(QuadSolver):
    def __init__(self, init_const_num_var=None, use_constraints=False, mat_l=None, mat_q=None, lmbda_l=1.0, lmbda_q=1.0):
        self.lmbda_l = lmbda_l
        self.lmbda_q = lmbda_q
        super().__init__(sol_name=SolverName.simulated_annealing()[0],
                         real_sol_name=SolverName.simulated_annealing()[1],
                         init_const_num_var=init_const_num_var,
                         use_constraints=use_constraints,
                         mat_l=mat_l, mat_q=mat_q, qubo=True)
        return

    def _gen_var(self, num_var):
        self.var = dimod.BinaryArray([f'x{i}' for i in range(num_var)])
        return

    def _load_timeout_or_read_limit(self):
        num_var = self.get_var().size
        if num_var <= 5000:
            timeout = 15000
        elif 5000 < num_var <= 8000:
            timeout = 25000
        else:
            timeout = 50000
        log_printer.info("Since # of steps is not given, decide it based on # of bits")
        log_printer.info(f"Based on current # of bits: {num_var}, # of steps is {timeout}")
        return timeout

    def load_solver(self):
        self.sampler = nl.SimulatedAnnealingSampler()
        self.sampler.parameters['num_sweeps'] = self.timeout
        return

    def _build_q_from_sklearn(self, w, num_var, learning_model, interaction_terms):
        if isinstance(w, int) or isinstance(w, float):
            return 0
        w, b = w
        if (learning_model == "linear") or (learning_model == 1):
            bqm = dimod.BinaryQuadraticModel("BINARY")
            bqm.add_linear_from_array(w)
            return bqm + b
        elif (learning_model == "quadratic") or (learning_model == 2):
            bqm = dimod.BinaryQuadraticModel("BINARY")
            q_mat = quad_list_to_q_mat(w[num_var:], num_var=num_var, interaction_terms=interaction_terms)
            bqm.add_linear_from_array(w[:num_var])
            bqm.add_quadratic_from_dense(q_mat)
            return bqm + b

        elif learning_model == RegressionName.fm()[0]:
            w_lin, w_inter = w
            lin_terms, inter_terms = build_fm(w_lin=w_lin, w_inter=w_inter, var=self.var, custom_sum=sum)
            return lin_terms + inter_terms + b

        else:
            raise Exception("Unexpected learning_model")

    def _build_constraints(self, var_list):
        constraints_builder = SoftConstraintsBuilder(
            var_list=var_list, mat_l=self.coef_mat_l, mat_q=self.coef_mat_q, lmbda_l=self.lmbda_l, lmbda_q=self.lmbda_q)
        self.g = constraints_builder.build()
        return self.g
    
    def _build_constrained_model(self, objective, constraints):
        for c in constraints:
            objective += c
        return objective

    def solve(self, bqm):
        sampleset = self.sampler.sample(bqm)
        return sampleset.record['sample'][0], sampleset.record['energy'][0]


class DWaveNealSolverProxy(SolverProxy):
    def __init__(self, num_var=None, use_constraints=False,
                 mat_l=None, mat_q=None, lmbda_l=100, lmbda_q=100):
        super().__init__(sol_name=SolverName.simulated_annealing()[0],
                         real_sol_name=SolverName.simulated_annealing()[1],
                         num_var=num_var, use_constraints=use_constraints,
                         mat_l=mat_l, mat_q=mat_q)
        self.lmbda_l, self.lmbda_q = lmbda_l, lmbda_q
        return

    def _load(self):
        dw_sol = DWaveNealSolver(init_const_num_var=self.num_var,
                                 use_constraints=self.use_constraints,
                                 mat_l=self.coef_mat_l, mat_q=self.coef_mat_q,
                                 lmbda_l=self.lmbda_l, lmbda_q=self.lmbda_q)
        return dw_sol
