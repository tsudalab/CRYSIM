import time
from src.registry import SolverName
from src.solver.sol_cl import HyperOneHotSolver, HyperContinuousSolver, HyperSetSolver
from src.solver.sol_qubo import AmplifySolverProxy, AmplifyForLeapSolverProxy, AmplifyForDWaveSolverProxy


def load_solver_or_proxy(sol_name, opt_name, timeout=None, use_constraints=None):
    start_time = time.process_time()
    if sol_name == SolverName.amplify()[0]:
        if ('oh' in opt_name) and use_constraints:
            sol = AmplifySolverProxy(use_constraints=True)
            sol = load_timeout(sol, timeout)
        else:
            sol = AmplifySolverProxy(use_constraints=False)
            sol = load_timeout(sol, timeout)
    elif sol_name == SolverName.amplify_leap_hybrid()[0]:
        if ('oh' in opt_name) and use_constraints:
            sol = AmplifyForLeapSolverProxy(use_constraints=True)
            sol = load_timeout(sol, timeout)
        else:
            sol = AmplifyForLeapSolverProxy(use_constraints=False)
            sol = load_timeout(sol, timeout)
    elif sol_name == SolverName.amplify_dwave()[0]:
        if ('oh' in opt_name) and use_constraints:
            sol = AmplifyForDWaveSolverProxy(use_constraints=True)
            sol = load_timeout(sol, timeout)
        else:
            sol = AmplifyForDWaveSolverProxy(use_constraints=False)
            sol = load_timeout(sol, timeout)

    elif f'{SolverName.hyperopt()[0]}4oh' in sol_name:
        if SolverName.simulated_annealing()[0] in sol_name:
            sol = HyperOneHotSolver(SolverName.simulated_annealing()[0])
        elif SolverName.bayesian_optimization()[0] in sol_name:
            sol = HyperOneHotSolver(SolverName.bayesian_optimization()[0])
        else:
            raise NotImplementedError

    elif (SolverName.hyperopt()[0] in sol_name) and ('oh' not in sol_name):
        if SolverName.simulated_annealing()[0] in sol_name:
            sol = HyperContinuousSolver(SolverName.simulated_annealing()[0])
        elif SolverName.bayesian_optimization()[0] in sol_name:
            sol = HyperContinuousSolver(SolverName.bayesian_optimization()[0])
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
    print(f"load solver time: {time.process_time() - start_time}")

    return sol


def load_set_solver(set_sol_name):
    if SolverName.simulated_annealing()[0] in set_sol_name:
        set_sol = HyperSetSolver(SolverName.simulated_annealing()[0])
    elif SolverName.bayesian_optimization()[0] in set_sol_name:
        set_sol = HyperSetSolver(SolverName.bayesian_optimization()[0])
    else:
        raise NotImplementedError
    return set_sol


def load_timeout(sol, timeout):
    sol.name = f'{sol.name}-t{timeout}'
    sol.real_name += f'{sol.real_name}-t{timeout}'
    sol.get_timeout_or_read_limit(timeout)
    return sol
