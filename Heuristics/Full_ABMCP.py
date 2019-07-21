from Datastructures.Operation import *
from Datastructures.RandomGraph import *
import Datastructures.Ability
from Datastructures.Job import *
from Datastructures.Machine import Machine
from Datastructures.AbilityConfigurations import AbilityConfigurations
from gurobipy import *
from Heuristics.Heuristic import Heuristic
from Heuristics.RABMCP_MIP import RABMCP_MIP
from joblib import Parallel, delayed


class Full_ABMCP(Heuristic):

    fixed_num_machines = None

    def __init__(self, jobs, abilities, cmax, timeout):
        super().__init__(jobs, abilities, cmax, timeout)

    def _solve_low_multiplicity(self):
        # Implementation of IP (4.1)

        operations = self.operations
        jobs = self.jobs
        abilities = self.abilities

        counter = 0
        ac = AbilityConfigurations(abilities)
        for o in operations:
            ac.add_subset(o.get_abilities())
        print(len(ac.iter_configs()))
        mod = Model("Full_MIP")

        psum_t = {j: j.get_p_sum() for j in jobs}

        # Give possibility to fix the number of machines beforehand
        if Full_ABMCP.fixed_num_machines is None:
            machines_temp = list(range(len(self.operations)))
        else:
            machines_temp = list(range(Full_ABMCP.fixed_num_machines))

        print("No. of possible Machines: " + str(len(machines_temp)))

        bigM = self.cmax
        # # Assignment Jobs to Machine
        x = mod.addVars(operations, machines_temp, vtype=GRB.BINARY, name="x")
        # Assignment of abilities to machines
        z = mod.addVars(abilities, machines_temp, vtype=GRB.BINARY, name="z")
        # StartingTime
        S = mod.addVars(operations, name="S")
        # order of operations among one job
        w = mod.addVars(
            ((o1, o2) for j in jobs for o1 in j.iter_operations() for o2 in j.iter_operations() if not o1 == o2),
            vtype=GRB.BINARY, name="w")
        # order of operations on a machine m
        v = mod.addVars(operations, operations, machines_temp,
            vtype=GRB.BINARY, name="w")

        # Only equip as many abilites to a machine as possible due to resource constraints
        mod.addConstrs(quicksum(a.get_resource() * z[a, m] for a in abilities) <= 1 for m in machines_temp)

        # Only allow assignments job to machine if machine k has all the required abilities
        mod.addConstrs(
            x[o, m] <= quicksum(z[a, m] for a in o.get_abilities()) / len(o.get_abilities()) for o in operations for m in
            machines_temp
            if
            len(o.get_abilities()) > 0)

        # Determine max finish time of all jobs
        mod.addConstrs(S[o] <= self.cmax for o in operations)

        for j in jobs:
            for o1 in j.iter_operations():
                lhs_temp = S[o1] + o1.get_proc_time() - self.cmax
                for o2 in j.iter_operations():
                    if o1 != o2:
                        # Decide order of operations among one job
                        mod.addConstr(lhs_temp <= -self.cmax*w[o1, o2] + S[o2])
                        # Order operations belonging to 1 job
                        mod.addConstr(w[o1, o2] + w[o2, o1] == 1)

        # Fix given order through precedence relations
        mod.addConstrs(w[o1, o2] == 1 for j in jobs for o1 in j.iter_operations() for o2 in o1.iter_succ())

        mod.addConstrs(x[o1, m] + x[o2, m] - v[o1,o2,m] - v[o2,o1,m] <= 1 for o1 in operations for o2 in operations for m in machines_temp)

        for o1 in operations:
            lhs_temp = S[o1] + o1.get_proc_time() - self.cmax
            for o2 in operations:
                if o1 != o2:
                    for m in machines_temp:
                        # Decide order of operations among one job
                        mod.addConstr(lhs_temp <= -self.cmax*v[o1, o2, m] + S[o2])

        # A jobs must be assigned to a machine
        mod.addConstrs(quicksum(x[o, m] for m in machines_temp) == 1 for o in operations)

        mod.setParam("LogToConsole", 1 if self.verbose else 0)
        mod.setParam("TimeLimit", self.timeout)

        mod.setObjective(quicksum(a.get_cost() * z[a, m] for a in abilities for m in machines_temp))
        mod.optimize()

        if mod.getAttr("SolCount") > 0:
            machines_used = self._get_solution(x, z, S, machines_temp)
        else:
            mod.computeIIS()
            mod.write("test.ilp")
            return []
        return machines_used

    def _solve_low_multiplicity_time_indexed(self):
        # implemented for internal testing. Not finalized

        operations = self.operations
        jobs = self.jobs
        abilities = self.abilities

        ac = AbilityConfigurations(abilities)
        for o in operations:
            ac.add_subset(o.get_abilities())

        counter = 0
        machines = []
        for a in ac.iter_configs():
            for i in range(1):
                m = Machine(str(counter))
                m.set_assigned_abilities(a)
                machines.append(m)
                counter += 1
        print("No. of possible Machines: " + str(len(machines)))

        mod = Model("Full_MIP")

        use_machine = mod.addVars(machines, vtype=GRB.BINARY)
        psum_t = {j: j.get_p_sum() for j in jobs}

        for o in operations:
            o.start_at(0)
        ES = {j: j.get_earliest_start(set_initial_path=True, extend_precedences=True) for j in jobs}

        T = list(range(int(self.cmax + max(o.get_proc_time() for o in operations))))
        max_T = T[-1]
        big_M = max_T * 2

        possible_machines = {o: [m for m in machines if o.get_abilities().issubset(m.get_assigned_abilities())] for o in
                             operations}
        possible_operations = {m: [o for o in operations if o.get_abilities().issubset(m.get_assigned_abilities())] for
                               m in machines}

        possible_combinations = [(o, m) for o in operations for m in possible_machines[o]]
        print("Possible assignments: " + str(len(possible_combinations)))
        print("Possible assignments (time-indexed): " + str(len(possible_combinations)*len(T)) + "\nBuilding Model")
        # StartingTime
        S = mod.addVars(possible_combinations, T, vtype=GRB.BINARY, name="S")
        # order of operations
        y = mod.addVars(
            ((o1, o2) for j in jobs for o1 in j.iter_operations() for o2 in j.iter_operations() if not o1 == o2),
            vtype=GRB.BINARY, name="y")

        # assign every job once
        mod.addConstrs(quicksum(S[o, m, t] for t in T for m in possible_machines[o]) == 1 for o in operations)
        mod.addConstrs(quicksum(S[o, m, t] for t in T for o in operations for m in possible_machines[o]) <= use_machine[m] for m in machines)

        # makespan
        mod.addConstrs(
            quicksum(S[o, m, t] * (t + o.get_proc_time()) for t in T) <= self.cmax for (o, m) in possible_combinations)

        # Max parallel tasks on machine
        def get_lhs(m_var, t_var):
            return quicksum(
                S[o_var, m_var, t2] for o_var in possible_operations[m_var] for t2 in
                range(max(t_var - o_var.get_proc_time() + 1, 0), t_var + 1))

        # Create Constraints in parallel
        # Not more than one job per machine at one timelsot
        lh_side = Parallel()(delayed(get_lhs)(m, t) for m in machines for t in range(max_T))
        mod.addConstrs(lh_side[i] <= 1 for i in range(len(lh_side)))

        # Fix given order through precedence relations
        mod.addConstrs(y[o1, o2] == 1 for j in jobs for o1 in j.iter_operations() for o2 in o1.iter_succ())

        # Pre-Process
        mod.addConstrs(
            quicksum(S[o, m, t] for m in possible_machines[o] for t in range(ES[j][o])) == 0 for j in jobs for o in
            j.iter_operations())

        # Preparation: save linear terms:
        lin_terms_lhs = {}
        lin_terms_rhs = {}
        for j in jobs:
            for o in j.iter_operations():
                lin_terms_lhs[j, o] = quicksum(
                    (t + o.get_proc_time()) * S[o, m, t] for t in T for m in possible_machines[o])
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
        mod.setParam("TimeLimit", self.timeout)

        mod.setObjective(quicksum(use_machine[m]*(sum(a.get_cost() for a in m.get_assigned_abilities())) for m in machines))

        mod.optimize()

        machines_used = []
        if mod.getAttr("SolCount") > 0:
            machines_used = [m for m in machines if use_machine[m].x > 0.5]
            for m in machines_used:
                for o in possible_operations[m]:
                    if any(S[o, m, t] > 0.5 for t in T):
                        m.assign_operation(o)
                        o.assign_machine(m)
                        for t in T:
                            if S[o, m, t] > 0.5:
                                o.start_at(t)
                                break
        return machines_used

    def _get_solution(self, x, z, S, machines_temp):
        machines_used = dict()
        for m in machines_temp:
            assigned_abilities = []
            for a in self.abilities:
                if z[a, m].x > 0.5:
                    assigned_abilities.append(a)
            if len(assigned_abilities) > 0:
                mac = Machine(str(m))
                mac.set_assigned_abilities(assigned_abilities)
                machines_used[m] = mac

        for o in self.operations:
            for m in machines_used:
                if x[o, m].x > 0.5:
                    o.assign_machine(machines_used[m])
                    machines_used[m].assign_operation(o)
            o.start_at(S[o].x)

        return list(m for m in machines_used.values() if len(m.get_assigned_operations()) > 0)

    def _solve_high_multiplicity(self):
        # there is no high multiplicity version of this, so use low multiplicty instead
        return self._solve_low_multiplicity()
