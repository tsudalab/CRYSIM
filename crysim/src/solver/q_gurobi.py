import numpy as np
import yaml

try:
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError:
    pass

from crysim.src.registry import SolverName
from logzero import logger as log_printer
from crysim.src.solver.sol_base import QuadSolver, ConstraintsBuilder, SolverProxy


grb_tk = {}
with open("token.yml", "r") as tk_dict:
    params = yaml.safe_load(tk_dict)
grb_tk_name = params["gurobi_token_file"]
try:
    with open(f"{grb_tk_name}", "r") as grb_lic:
        lic = grb_lic.readlines()
    grb_tk = {"WLSACCESSID": lic[3][12:-1],
              "WLSSECRET": lic[4][10:-1],
              "LICENSEID": lic[5][10:-1]}
except FileNotFoundError:
    pass


class GurobiSolver(QuadSolver):
    def __init__(self, init_const_num_var=None, use_constraints=False,
                 mat_l=None, mat_q=None, grb_lic_dir=None):
        self.grb_tk = grb_tk
        self.grb_lic_dir = grb_lic_dir
        self.load_solver()
        super().__init__(sol_name=SolverName.gurobi()[0], real_sol_name=SolverName.gurobi()[1],
                         init_const_num_var=init_const_num_var, use_constraints=use_constraints,
                         mat_l=mat_l, mat_q=mat_q, qubo=True, custom_sum=gp.quicksum)
        return

    def _gen_var(self, num_var):
        self.var = np.array([self.grb_model.addVar(vtype=GRB.BINARY, name=f"x{i}") for i in range(num_var)])
        return

    def _load_timeout_or_read_limit(self):
        num_var = self.get_var().size
        if num_var <= 5000:
            timeout = 1500.0
        elif 5000 < num_var <= 8000:
            timeout = 2500.0
        else:
            timeout = 5000.0
        log_printer.info("Since time limit is not given, decide it based on # of bits")
        log_printer.info(f"Based on current # of bits: {num_var}, time limit is {timeout}")
        return timeout

    def load_timeout_or_read_limit(self, timeout):
        if timeout is not None:
            self.timeout = timeout
            log_printer.info(f"Receive timeout: {timeout}")
        else:
            self.timeout = self._load_timeout_or_read_limit()
        self.grb_model.Params.TimeLimit = self.timeout
        return

    def load_solver(self):
        if self.grb_lic_dir is None:
            if len(self.grb_tk) == 0:
                with open("token.yml", "r") as tk_dict:
                    params = yaml.safe_load(tk_dict)
                self.grb_tk = params["gurobi_token"]
                assert (self.grb_tk["WLSACCESSID"] != "-") and (self.grb_tk["WLSSECRET"] != "-") \
                       and (self.grb_tk["LICENSEID"] != "-"), "Please provide license for Gurobi"
        else:
            log_printer.info(f"load license from {self.grb_lic_dir}")
            with open(self.grb_lic_dir, "r") as grb_lic:
                lic = grb_lic.readlines()
            self.grb_tk = {"WLSACCESSID": lic[3][12:-1],
                           "WLSSECRET": lic[4][10:-1],
                           "LICENSEID": lic[5][10:-1]}

        options = {
            "WLSACCESSID": self.grb_tk["WLSACCESSID"],
            "WLSSECRET": self.grb_tk["WLSSECRET"],
            "LICENSEID": int(self.grb_tk["LICENSEID"]),
        }
        env = gp.Env(params=options)
        self.grb_model = gp.Model("GRB-CSP", env=env)
        return

    def _build_q(self, q):
        q, b = q
        obj = gp.QuadExpr() + b
        for term, coef in q.items():
            obj.add(expr=self.var[term[0]] * self.var[term[1]], mult=coef)
        return obj

    def _build_constraints(self, var_list):
        constraints_builder = GurobiConstraintsBuilder(
            var_list=var_list, mat_l=self.coef_mat_l, mat_q=self.coef_mat_q)
        self.g = constraints_builder.build()
        return self.g

    def _build_constrained_model(self, objective, constraints):
        for i, c in enumerate(constraints):
            self.grb_model.addConstr(c, name=f"c{i}")
        return objective

    def _solve(self, bqm, num_solves):
        solve_flag = False
        while not solve_flag:
            try:
                self.grb_model.setObjective(bqm, GRB.MINIMIZE)
                self.grb_model.optimize()
                solve_flag = True
            except gp.GurobiError:
                log_printer.info("Solve fail. Try to solve again.")
                solve_flag = False
        return self.grb_model

    def solve(self, bqm):
        result = self._solve(bqm, num_solves=1)
        try:
            q_values = np.array([v.X for v in result.getVars()])

            # # TEST RESULTS: correctly built the BQM
            # energy_values = result.ObjVal
            # w_lin = result.getAttr("Obj", result.getVars())
            # q_name = [v.VarName for v in result.getVars()]
            # e = 0
            # for i in range(len(w_lin)):
            #     e += q_values[i] * w_lin[i]
            # obj = result.getObjective()
            # for i in range(obj.size()):
            #     var1 = obj.getVar1(i)
            #     var2 = obj.getVar2(i)
            #     coe = obj.getCoeff(i)
            #     e += var1.X * var2.X * coe
            #     i1, i2 = var1.index, var2.index
            #     n1, n2 = var1.VarName, var2.VarName
            #     assert (n1 == q_name[i1]) and (n2 == q_name[i2])
            # assert e == energy_values, f"manual: {e}; calculated: {energy_values}"
            # log_printer.info(
            #                 f"pass the test with manual: {e}; calculated: {energy_values}")  # pass

        except gp.GurobiError:
            print('Did not find the solution')
            q_values = None
        return q_values, result.ObjVal


class GurobiConstraintsBuilder(ConstraintsBuilder):
    def __init__(self, var_list, mat_l, mat_q):
        super().__init__(var_list, mat_l, mat_q)
        return

    def _build_equation(self, mat, bias, weight):
        return mat == bias


class GurobiSolverProxy(SolverProxy):
    def __init__(self, num_var=None, use_constraints=False,
                 mat_l=None, mat_q=None, grb_lic_dir=None):
        self.grb_lic_dir = grb_lic_dir
        super().__init__(sol_name=SolverName.gurobi()[0],
                         real_sol_name=SolverName.gurobi()[1],
                         num_var=num_var, use_constraints=use_constraints,
                         mat_l=mat_l, mat_q=mat_q)
        return

    def unwrap(self):
        grb_sol = GurobiSolver(init_const_num_var=self.num_var,
                               use_constraints=self.use_constraints,
                               mat_l=self.coef_mat_l, mat_q=self.coef_mat_q,
                               grb_lic_dir=self.grb_lic_dir)
        grb_sol.load_timeout_or_read_limit(self.timeout)
        if self.use_constraints:
            print('Finish building constraints')
        return grb_sol
