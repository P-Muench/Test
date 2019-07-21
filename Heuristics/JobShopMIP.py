import sys
from gurobipy import *
import random as r
import math
from Datastructures.Job import *
import _heapq
from Heuristics.Heuristic import Heuristic
import _multiprocessing
from joblib import Parallel, delayed


class JobShopMIP(Heuristic):

    def __init__(self, jobs, abilities, cmax, timeout, machines, upper_bound=sys.maxsize):
        super().__init__(jobs, abilities, cmax, timeout, upper_bound)
        self.machines = machines

        self.time_indexed = True

    def solve(self):
        if self.time_indexed:
            return self._solve_time_indexed()
        else:
            return self._solve_non_time_indexed()

    def _solve_time_indexed(self):
        # For testing, a model that solves the FJSSP has been developed. Not finalized.
        operations = self.operations
        jobs = self.jobs
        machines = self.machines

        psum_t = {j: j.get_p_sum() for j in jobs}

        for o in operations:
            o.start_at(0)
        ES = {j: j.get_earliest_start(set_initial_path=True, extend_precedences=True) for j in jobs}

        T = list(range(int(self.cmax + max(o.get_proc_time() for o in operations))))
        max_T = T[-1]
        big_M = max_T*2

        possible_machines = {o: [m for m in machines if o.get_abilities().issubset(m.get_assigned_abilities())] for o in operations}
        possible_operations = {m: [o for o in operations if o.get_abilities().issubset(m.get_assigned_abilities())] for m in machines}

        possible_combinations = [(o, m) for o in operations for m in possible_machines[o]]
        mod = Model("choose paths")
        # StartingTime
        S = mod.addVars(possible_combinations, T, vtype=GRB.BINARY, name="S")
        # order of operations
        y = mod.addVars(((o1, o2) for j in jobs for o1 in j.iter_operations() for o2 in j.iter_operations() if not o1 == o2),
                        vtype=GRB.BINARY, name="y")

        # assign every job once
        mod.addConstrs(quicksum(S[o, m, t] for t in T for m in possible_machines[o]) == 1 for o in operations)

        # makespan
        mod.addConstrs(quicksum(S[o, m, t]*(t + o.get_proc_time()) for t in T) <= self.cmax for (o, m) in possible_combinations)

        # Max parallel tasks on machine
        def get_lhs(m_var, t_var):
            return quicksum(
                S[o_var, m_var, t2] for o_var in possible_operations[m_var] for t2 in range(max(t_var - o_var.get_proc_time() + 1, 0), t_var + 1))

        # Create Constraints in parallel
        # Not more than one job per machine at one timelsot
        lh_side = Parallel()(delayed(get_lhs)(m, t) for m in machines for t in range(max_T))
        mod.addConstrs(lh_side[i] <= 1 for i in range(len(lh_side)))

        # Fix given order through precedence relations
        mod.addConstrs(y[o1, o2] == 1 for j in jobs for o1 in j.iter_operations() for o2 in o1.iter_succ())

        # Pre-Process
        mod.addConstrs(quicksum(S[o, m, t] for m in possible_machines[o] for t in range(ES[j][o])) == 0 for j in jobs for o in j.iter_operations())

        # Preparation: save linear terms:
        lin_terms_lhs = {}
        lin_terms_rhs = {}
        for j in jobs:
            for o in j.iter_operations():
                lin_terms_lhs[j, o] = quicksum((t + o.get_proc_time()) * S[o, m, t] for t in T for m in possible_machines[o])
                lin_terms_rhs[j, o] = quicksum(t * S[o, m, t] for t in T for m in possible_machines[o])

        for j in jobs:
            for o1 in j.iter_operations():
                ter_lhs = lin_terms_lhs[j, o1]
                for o2 in j.iter_operations():
                    if not o1 == o2:
                        # Stick to precedence relation while scheduling
                        mod.addConstr(
                            ter_lhs
                            <=
                            lin_terms_rhs[j, o2] + big_M * y[o2, o1])

                        # Order operations belonging to 1 job
                        mod.addConstr(y[o1, o2] + y[o2, o1] == 1)

        mod.setParam("LogToConsole", 1 if self.verbose else 0)
        mod.setParam("MIPFocus", 1)
        mod.setParam("SolutionLimit", 1)
        mod.setParam("TimeLimit", self.timeout)

        mod.optimize()
        if mod.getAttr("Status") == GRB.INFEASIBLE or mod.getAttr("SolCount") == 0:
            print("Scheduling Infeasible")
            return False
        else:
            mod.write("schedule.sol")
            for o in operations:
                for m in possible_machines[o]:
                    for t in T:
                        if S[o, m, t].x > 0.5:
                            o.start_at(t)
                            o.assign_machine(m)
                            m.assign_operation(o)
            for j in jobs:
                unscheduled = sorted(j.graph.nodes.copy(), key=lambda o: (o.get_start(), o.get_start() + o.get_proc_time()))
                sorted_operations = []
                while len(unscheduled) != 0:
                    for v in unscheduled.copy():
                        v_schedulable = True
                        for v_pred in v.iter_pred():
                            v_schedulable &= (v_pred in sorted_operations)
                        if v_schedulable:
                            sorted_operations.append(v)
                            unscheduled.remove(v)
                            break
                for i in range(len(sorted_operations) - 1):
                    sorted_operations[i].choose_succ(sorted_operations[i + 1])
            return True

    def _solve_non_time_indexed(self):
        raise NotImplementedError()
