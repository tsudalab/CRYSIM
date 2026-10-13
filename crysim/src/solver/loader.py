import time
from crysim.src.registry import SolverName
from crysim.src.solver.cl_hy import HyperOptProxy
from crysim.src.solver.q_dwnl import DWaveNealSolverProxy
from crysim.src.solver.q_amplify import AmplifySolverProxy, AmplifyForLeapSolverProxy, AmplifyForDWaveSolverProxy
from crysim.src.solver.q_cpsat import CPSATProxy
from crysim.src.solver.q_gurobi import GurobiSolverProxy


def load_solver_or_proxy(sol_name, timeout=None, mt_lambda=None, use_constraints=None,
                         grb_lic_dir=None, constraints_lmbda_l=None, constraints_lmbda_q=None):

    start_time = time.process_time()
    if sol_name == SolverName.amplify()[0]:
        sol = AmplifySolverProxy(use_constraints=use_constraints)
    elif sol_name == SolverName.amplify_leap_hybrid()[0]:
        sol = AmplifyForLeapSolverProxy(use_constraints=use_constraints)
    elif sol_name == SolverName.amplify_dwave()[0]:
        sol = AmplifyForDWaveSolverProxy(use_constraints=use_constraints)

    elif sol_name == SolverName.gurobi()[0]:
        sol = GurobiSolverProxy(use_constraints=use_constraints, grb_lic_dir=grb_lic_dir)

    elif sol_name == SolverName.simulated_annealing()[0]:
        sol = DWaveNealSolverProxy(use_constraints=use_constraints, lmbda_l=constraints_lmbda_l, lmbda_q=constraints_lmbda_q)

    elif SolverName.hyperopt()[0] in sol_name:
        if '4oh' in sol_name:
            if SolverName.simulated_annealing()[0] in sol_name:
                sol = HyperOptProxy(algo=SolverName.simulated_annealing()[0], is_var_bit=True, 
                                    use_constraints=use_constraints, lmbda_l=constraints_lmbda_l, lmbda_q=constraints_lmbda_q)
            elif SolverName.bayesian_optimization()[0] in sol_name:
                sol = HyperOptProxy(algo=SolverName.bayesian_optimization()[0], is_var_bit=True, 
                                    use_constraints=use_constraints, lmbda_l=constraints_lmbda_l, lmbda_q=constraints_lmbda_q)
        else:
            if SolverName.simulated_annealing()[0] in sol_name:
                sol = HyperOptProxy(algo=SolverName.simulated_annealing()[0], is_var_bit=False, 
                                    use_constraints=use_constraints, lmbda_l=constraints_lmbda_l, lmbda_q=constraints_lmbda_q)
            elif SolverName.bayesian_optimization()[0] in sol_name:
                sol = HyperOptProxy(algo=SolverName.bayesian_optimization()[0], is_var_bit=False, 
                                    use_constraints=use_constraints, lmbda_l=constraints_lmbda_l, lmbda_q=constraints_lmbda_q)

    elif sol_name == SolverName.cpsat_ortools()[0]:
        sol = CPSATProxy(use_constraints=use_constraints)

    else:
        raise NotImplementedError

    sol = load_timeout(sol, timeout)
    print(f"load solver time: {time.process_time() - start_time}")

    return sol


def load_timeout(sol, timeout):
    sol.get_timeout_or_read_limit(timeout)
    return sol
