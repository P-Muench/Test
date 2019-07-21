from Datastructures.Operation import *
from Datastructures.RandomGraph import *
import Datastructures.Ability
from Datastructures.Job import *
from Datastructures.Machine import Machine
from Datastructures.AbilityConfigurations import AbilityConfigurations
from gurobipy import *
from Heuristics.Heuristic import Heuristic
import time


# IP (5.2) for LM case, extension for HM
class RABMCP_MIP(Heuristic):

    def __init__(self, jobs, abilities, cmax, timeout):
        super().__init__(jobs, abilities, cmax, timeout)
        self._approx_num_of_machines = -1
        self._lower_bound = 0

    def _guess_num_of_machines(self):
        """
        Guess to number of machines that are to be used.
        """
        reqs = {}
        for o in self.operations:
            t = tuple(sorted(o.get_abilities(), key=lambda a: self.abilities.index(a)))
            if t not in reqs:
                reqs[t] = [o]
            else:
                reqs[t].append(o)

        self._approx_num_of_machines = sum(RABMCP_MIP.bin_packing_approx(reqs[ac], self.cmax) for ac in reqs)

    @staticmethod
    def bin_packing_approx(operations: List[Operation], size: float) -> int:
        """
        First fit decreasing algorithm

        :param operations:
        :param size:
        :return:
        """
        sorted_operations = sorted(operations, key=lambda o: -o.get_proc_time())
        bins: Dict[int, List[Operation]] = {b: list() for b in range(len(operations))}
        for o in sorted_operations:
            for b in bins:
                if sum(o2.get_proc_time() for o2 in bins[b]) + o.get_proc_time() <= size:
                    bins[b].append(o)
                    break
        return sum(1 for b in bins if len(bins[b]) > 0)

    def _solve_low_multiplicity(self):
        operations = self.operations
        start_time = time.clock()
        abilities = self.abilities

        if self._approx_num_of_machines != -1:
            self._approx_num_of_machines = self.n
        else:
            self._guess_num_of_machines()
            print("Manual guess for num of machines not provided. Guessed the number of machines: " + str(self._approx_num_of_machines))
        machines_temp = list(range(self._approx_num_of_machines))

        bigM = self.cmax

        mod = Model("FJSS")
        # Assignment Jobs to Machine
        x = mod.addVars(operations, machines_temp, vtype=GRB.BINARY, name="x")
        # Assignment of abilities to machines
        z = mod.addVars(abilities, machines_temp, vtype=GRB.BINARY, name="z")

        mod.setObjective(
            quicksum(a.get_cost() * z[a, m] for a in abilities for m in machines_temp))

        # Determine max finish time of all jobs
        mod.addConstrs(quicksum(x[o, m] * o.get_proc_time() for o in operations) <= self.cmax for m in machines_temp)

        # A jobs must be assigned to a machine
        mod.addConstrs(quicksum(x[o, m] for m in machines_temp) == 1 for o in operations)

        # Machine Configuration constraints:
        # Only allow assignments job to machine if machine k has all the required abilities
        mod.addConstrs(
            x[o, m] <= quicksum(z[a, m] for a in o.get_abilities()) / len(o.get_abilities()) for o in operations for m in
            machines_temp
            if
            len(o.get_abilities()) > 0)

        # Only equip as many abilites to a machine as possible due to resource constraints
        mod.addConstrs(quicksum(a.get_resource() * z[a, m] for a in abilities) <= 1 for m in machines_temp)

        # Symmetry breaker
        mod.addConstrs(
            quicksum(z[a2, m] for a2 in abilities) >= z[a, m + 1] for a in abilities for m in machines_temp[:-1])

        mod.setParam("LogToConsole", 1 if self.verbose else 0)
        mod.setParam("LogFile", self.output_filename)

        mod.setParam("TimeLimit", self.timeout - (time.clock() - start_time))

        mod.optimize()
        self._lower_bound = mod.getAttr("ObjBound")
        return RABMCP_MIP.get_machines(x, z, machines_temp, abilities, operations)

    @staticmethod
    def get_machines(x, z, machines_temp, abilities, operations):
        machines_used = dict()
        for m in machines_temp:
            assigned_abilities = []
            for a in abilities:
                if z[a, m].x > 0.5:
                    assigned_abilities.append(a)
            if len(assigned_abilities) > 0:
                mac = Machine(str(m))
                mac.set_assigned_abilities(assigned_abilities)
                machines_used[m] = mac

        for o in operations:
            for m in machines_used:
                if x[o, m].x > 0.5:
                    o.assign_machine(machines_used[m])
                    machines_used[m].assign_operation(o)

        # Sanity Check:
        for o in operations:
            assert o.get_assigned_machine() is not None, "Operation not assigned"
        for m in machines_used.values():
            for o in m.get_assigned_operations():
                assert o.get_assigned_machine() == m

        return list(m for m in machines_used.values() if len(m.get_assigned_operations()) > 0)

    def _solve_high_multiplicity(self):
        start_time = time.clock()
        operations = self.operations
        abilities = self.abilities

        if self._approx_num_of_machines != -1:
            self._approx_num_of_machines = self.n
        else:
            self._guess_num_of_machines()
            print("Manual guess for num of machines not provided. Guessed the number of machines: " + str(
                self._approx_num_of_machines * self.multiplier))
        machines_temp = list(range(int(self._approx_num_of_machines * self.multiplier)))

        bigM = self.cmax

        mod = Model("FJSS")
        # Assignment Jobs to Machine
        x = mod.addVars(operations, machines_temp, vtype=GRB.INTEGER, name="x")

        y = mod.addVars(operations, machines_temp, vtype=GRB.BINARY, name="y")
        # Assignment of abilities to machines
        z = mod.addVars(abilities, machines_temp, vtype=GRB.BINARY, name="z")

        mod.setObjective(
            quicksum(a.get_cost() * z[a, m] for a in abilities for m in machines_temp))

        # Determine max finish time of all jobs
        mod.addConstrs(quicksum(x[o, m] * o.get_proc_time() for o in operations) <= self.cmax for m in machines_temp)

        # A jobs must be assigned to a machine
        mod.addConstrs(quicksum(x[o, m] for m in machines_temp) == o._multiplier for o in operations)

        mod.addConstrs(x[o, m] <= o._multiplier*y[o, m] for o in operations for m in machines_temp)

        # Machine Configuration constraints:
        # Only allow assignments job to machine if machine k has all the required abilities
        mod.addConstrs(
            y[o, m] <= quicksum(z[a, m] for a in o.get_abilities()) / len(o.get_abilities()) for o in operations for m
            in
            machines_temp if len(o.get_abilities()) > 0)

        # Only equip as many abilites to a machine as possible due to resource constraints
        mod.addConstrs(quicksum(a.get_resource() * z[a, m] for a in abilities) <= 1 for m in machines_temp)

        # Symmetry breaker
        mod.addConstrs(
            quicksum(z[a2, m] for a2 in abilities) >= z[a, m + 1] for a in abilities for m in machines_temp[:-1])

        # mod.read("relaxed.sol")
        # mod.update()

        mod.setParam("LogToConsole", 1 if self.verbose else 0)
        mod.setParam("TimeLimit", self.timeout - (time.clock() - start_time))
        mod.optimize()

        self._lower_bound = mod.getAttr("ObjBound")
        return RABMCP_MIP.get_machines_high_multiplicity(x, z, machines_temp, abilities, self.jobs)

    @staticmethod
    def get_machines_high_multiplicity(x, z, machines_temp, abilities, jobs: List[Job]):
        j_copies = {j: j.to_low_multiplicity() for j in jobs}
        machines_used = dict()

        for m in machines_temp:
            assigned_abilities = []
            for a in abilities:
                if z[a, m].x > 0.5:
                    assigned_abilities.append(a)
            if len(assigned_abilities) > 0:
                mac = Machine(str(m))
                mac.set_assigned_abilities(assigned_abilities)
                machines_used[m] = mac

        for j in jobs:
            for pos, o in enumerate(j.graph.nodes):
                assigned_copies = 0
                for m in machines_used:
                    if x[o, m].x > 0.5:
                        for i in range(int(x[o, m].x + .4)):
                            j_copy = j_copies[j][i + assigned_copies]
                            o_copy = j_copy.graph.nodes[pos]
                            o_copy.assign_machine(m)
                            machines_used[m].assign_operation(o_copy)
                        assigned_copies += int(x[o, m].x + .4)

        new_job_list: List[Job] = [j_new for j in jobs for j_new in j_copies[j]]
        new_machines: List[Machine] = list(m for m in machines_used.values() if len(m.get_assigned_operations()) > 0)

        # Sanity Check:
        counter = 0
        for j in new_job_list:
            for o in j.iter_operations():
                counter += 1
                if o.get_assigned_machine() is None:
                    print(str(o) + " not assigned")
                    try:
                        min_m = min([m for m in new_machines if o.get_abilities().issubset(m.get_assigned_abilities())], key=lambda m: m.get_load() + o.get_proc_time())
                        min_m.assign_operation(o)
                        o.assign_machine(min_m)
                    except Exception as e:
                        print(e)
                        print(str(o) + " could not be assigned")
                assert o.get_assigned_machine() is not None, "Operation " + str(o) + " not assigned"
        assigned_operations = []
        for m in new_machines:
            assigned_operations += m.get_assigned_operations()
        assert len(assigned_operations) == len(set(assigned_operations)), "Operations added to multiple machines"
        return new_machines, new_job_list

    @staticmethod
    def post_process_HM(jobs, operations, abilities, machines: List[Machine], cmax, lb=0, timelimit=60):
        machines_temp = list(range(len(machines) + 1))

        bigM = cmax

        mod = Model("FJSS")
        # Assignment Jobs to Machine
        x = mod.addVars(operations, machines_temp, vtype=GRB.INTEGER, name="x")

        y = mod.addVars(operations, machines_temp, vtype=GRB.BINARY, name="x")
        # Assignment of abilities to machines
        z = mod.addVars(abilities, machines_temp, vtype=GRB.BINARY, name="z")

        mod.setObjective(
            quicksum(a.get_cost() * z[a, m] for a in abilities for m in machines_temp))

        # Determine max finish time of all jobs
        mod.addConstrs(quicksum(x[o, m] * o.get_proc_time() for o in operations) <= cmax for m in machines_temp)

        # A jobs must be assigned to a machine
        mod.addConstrs(quicksum(x[o, m] for m in machines_temp) == o._multiplier for o in operations)

        mod.addConstrs(x[o, m] <= o._multiplier * y[o, m] for o in operations for m in machines_temp)

        # Machine Configuration constraints:
        # Only allow assignments job to machine if machine k has all the required abilities
        mod.addConstrs(
            y[o, m] <= quicksum(z[a, m] for a in o.get_abilities()) / len(o.get_abilities()) for o in operations for m
            in
            machines_temp
            if
            len(o.get_abilities()) > 0)

        # Only equip as many abilites to a machine as possible due to resource constraints
        mod.addConstrs(quicksum(a.get_resource() * z[a, m] for a in abilities) <= 1 for m in machines_temp)

        # Symmetry breaker
        mod.addConstrs(
            quicksum(z[a2, m] for a2 in abilities) >= z[a, m + 1] for a in abilities for m in machines_temp[:-1])

        for m_num, m in enumerate(machines):
            for a in m.get_assigned_abilities():
                z[a, m_num].start = 1
            for o in m.get_assigned_operations():
                if o in operations:
                    x[o, m_num].start = 1

        mod.addConstr(quicksum(a.get_cost() * z[a, m] for a in abilities for m in machines_temp) >= lb)
        # mod.addConstr(quicksum(z[Ability._dummy, m] for m in machines_temp) == 1)

        mod.setParam("LogToConsole", 1)
        mod.setParam("MIPFocus", 1)
        mod.setParam("TimeLimit", timelimit)
        mod.optimize()

        print("Postprocessing improved solution to value " + str(mod.ObjVal))
        return RABMCP_MIP.get_machines_high_multiplicity(x, z, machines_temp, abilities, jobs)

    @classmethod
    def post_post_process_HM(cls, jobs, operations, abilities, machines, cmax, _lower_bound, time_spent_postprocessing):
        bigM = cmax

        mod = Model("FJSS")
        # Assignment Jobs to Machine
        x = mod.addVars(operations, machines, vtype=GRB.INTEGER, name="x")

        y = mod.addVars(operations, machines, vtype=GRB.BINARY, name="x")
        # Assignment of abilities to machines
        z = mod.addVars(abilities, machines, vtype=GRB.BINARY, name="z")
        makespan = mod.addVars(machines, name="makespan")

        mod.setObjective(quicksum(makespan[m]*makespan[m]/(cmax*cmax) for m in machines))

        # Determine max finish time of all jobs
        mod.addConstrs(quicksum(x[o, m] * o.get_proc_time() for o in operations) <= makespan[m] for m in machines)
        mod.addConstrs(makespan[m] <= cmax for m in machines)

        # A jobs must be assigned to a machine
        mod.addConstrs(quicksum(x[o, m] for m in machines) == o._multiplier for o in operations)

        mod.addConstrs(x[o, m] <= o._multiplier * y[o, m] for o in operations for m in machines)

        # Machine Configuration constraints:
        # Only allow assignments job to machine if machine k has all the required abilities
        mod.addConstrs(
            y[o, m] <= quicksum(z[a, m] for a in o.get_abilities()) / len(o.get_abilities()) for o in operations for m
            in
            machines
            if
            len(o.get_abilities()) > 0)

        # Only equip as many abilites to a machine as possible due to resource constraints
        mod.addConstrs(quicksum(a.get_resource() * z[a, m] for a in abilities) <= 1 for m in machines)

        for m in machines:
            for a in abilities:
                if a in m.get_assigned_abilities():
                    mod.addConstr(z[a, m] == 1)
                else:
                    mod.addConstr(z[a, m] == 0)
            # for o in m.get_assigned_operations():
            #     if o in operations:
            #         x[o, m].start = 1

        mod.setParam("LogToConsole", 1)
        mod.setParam("MIPFocus", 1)
        mod.setParam("TimeLimit", time_spent_postprocessing)
        mod.optimize()

        print("Post-post-processing improved makespan to value " + str(max(makespan[m].x for m in machines)) + "\nObjective is %s"%(mod.ObjVal))
        return RABMCP_MIP.get_machines_high_multiplicity(x, z, machines, abilities, jobs)
