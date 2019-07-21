import math
import time
import sys
from Datastructures.Ability import Ability
from Heuristics.Heuristic import Heuristic
from Datastructures.AbilityConfigurations import AbilityConfigurations
from Datastructures.Machine import Machine
from Datastructures.Operation import Operation
import networkx as nx
from gurobipy import *
import scipy.misc
from typing import List, Set, Dict, Iterable, Generator, Tuple


# Algorithms as developed in chapter 5.4
class SmallKHeuristic(Heuristic):

    def __init__(self, jobs, abilities, cmax, timeout, upper_bound=sys.maxsize):
        super().__init__(jobs, abilities, cmax, timeout, upper_bound)

    def _solve_low_multiplicity(self):
        start_time = time.clock()

        operations = self.operations
        abilities = self.abilities
        timeout = self.timeout
        n = self.n
        c_max = self.cmax

        best_cost = self.upper_bound

        ac = AbilityConfigurations(abilities)
        for o in operations:
            ac.add_subset(o.get_abilities())
        all_configs = list(ac.iter_configs())

        num_iterations = int(scipy.misc.comb(len(all_configs) + len(operations), len(all_configs)))

        sol = []
        print("Trying all " + str(num_iterations) + " combinations.")
        for y in max_n_iterator(all_configs, n):
            if time.clock() - start_time > timeout:
                print("Time limit reached")
                break
            machines = []
            counter = 0
            for pos, num in enumerate(y):
                config = all_configs[pos]
                for i in range(num):
                    m = Machine(str(counter))
                    m.set_assigned_abilities(config)
                    machines.append(m)
                    counter += 1
            if sum(m.get_cost() for m in machines) >= best_cost:
                continue
            c_max_computed = LST_Rounding(operations, machines, c_max)
            if c_max_computed <= 2 * c_max:
                machines = duplicate_machines(operations, machines, c_max)
                machine_cost = sum(m.get_cost() for m in machines)
                if machine_cost < best_cost:
                    best_cost = machine_cost
                    sol = machines
        else:
            print("Extensive Search Terminated")

        print("Found " + str(len(sol)) + " machines")
        for m in sol:
            for o in m.get_assigned_operations():
                o.assign_machine(m)
        return sol

    def _solve_high_multiplicity(self):
        operations = self.operations
        abilities = self.abilities
        timeout = self.timeout

        best_cost = self.upper_bound

        ac = AbilityConfigurations(abilities)
        for o in operations:
            ac.add_subset(o.get_abilities())
        all_configs = list(ac.iter_configs())

        mod = Model("SmallKHeur_Prep")

        y = mod.addVars(all_configs, vtype=GRB.INTEGER, name="y")
        possible_assignments = {(o, config) for config in all_configs for o in operations if o.get_abilities().issubset(set(config)) and o.get_proc_time() > 0}
        x = mod.addVars(possible_assignments, vtype=GRB.CONTINUOUS, name="x")
        mod.addConstrs(quicksum(x[o, config]*(1.0/o.get_proc_time()) for config in all_configs if o.get_abilities().issubset(set(config))) == o._multiplier for o in operations if o.get_proc_time() > 0)

        mod.addConstrs(quicksum(x[o, config] for o in operations if (o, config) in possible_assignments) <= self.cmax*y[config] for config in all_configs)
        mod.setObjective(quicksum(y[config]*AbilityConfigurations.get_cost(config) for config in all_configs))

        mod.setParam("TimeLimit", timeout)
        mod.optimize()
        if mod.getAttr("Status") == GRB.INFEASIBLE:
            mod.computeIIS()
            mod.write("test.ilp")
            raise AssertionError("Model is infeasible")

        num_of_machines = {config: y[config].x for config in all_configs if y[config].x > .5}
        mod = Model("SmallKHeur")
        possible_assignments = {(o, config) for (o, config) in possible_assignments if config in num_of_machines}
        x = mod.addVars(possible_assignments, vtype=GRB.CONTINUOUS, name="x")
        c_max_var = mod.addVar(vtype=GRB.CONTINUOUS, name="c_max")
        mod.addConstrs(quicksum(x[o, config] * (1.0*num_of_machines[config] / (o.get_proc_time())) for config in num_of_machines if
                                (o, config) in possible_assignments) == o._multiplier for o in operations if
                       o.get_proc_time() > 0)

        mod.addConstrs(
            quicksum(x[o, config] for o in operations if (o, config) in possible_assignments) <= c_max_var for config in num_of_machines)

        mod.setObjective(c_max_var)
        mod.optimize()

        num_per_machine = {(o, m): x[o, m].x*num_of_machines[m]/(o.get_proc_time()) for (o, m) in possible_assignments}
        integral_assignments = {(o, m): math.floor(num_per_machine[o, m]) for (o, m) in possible_assignments}
        fractional_assignments = {(o, m) for (o, m) in possible_assignments if num_per_machine[o, m] - integral_assignments[o, m] > 1/(o.get_proc_time() + 1)}

        G = nx.Graph()

        G.add_edges_from(fractional_assignments)
        matching = nx.algorithms.matching.max_weight_matching(G)

        for t in matching:
            if isinstance(t[0], tuple):
                t = (t[1], t[0])
            if t in integral_assignments:
                integral_assignments[t] += 1
            else:
                integral_assignments[t] = 1

        return self._get_machines_HM(all_configs, integral_assignments)

    def _get_machines_HM(self, all_configs, integral_assignments):
        # Convert to low multiplicity
        operations_per_machine = {m: dict() for m in all_configs}
        for (o, m), num in integral_assignments.items():
            if num > 0:
                operations_per_machine[m][o] = num
        operations_per_machine = {m: d for m, d in operations_per_machine.items() if len(d) > 0}

        assigned_copies = {o: 0 for o in self.operations}
        j_copies = {j: j.to_low_multiplicity() for j in self.jobs}
        mapping_o = {o: [] for o in self.operations}
        for j in self.jobs:
            for pos, o in enumerate(j.graph.nodes):
                for j_copy in j_copies[j]:
                    mapping_o[o].append(j_copy.graph.nodes[pos])

        machines = []
        counter = 0
        for config in operations_per_machine:
            machines_to_add = [Machine(counter)]
            for (o, num) in operations_per_machine[config].items():
                for i in range(num):
                    o_copy = mapping_o[o][assigned_copies[o]]
                    for m in machines_to_add:
                        if m.get_load() + o.get_proc_time() <= self.cmax:
                            m.assign_operation(o_copy)
                            o_copy.assign_machine(m)
                            break
                    else:
                        counter += 1
                        m = Machine(counter)
                        machines_to_add.append(m)
                        m.assign_operation(o_copy)
                        o_copy.assign_machine(m)
                    assigned_copies[o] += 1
            counter += 1
            for m in machines_to_add:
                m.set_assigned_abilities(config)
            machines.extend(machines_to_add)

        new_job_list = [j_new for j in self.jobs for j_new in j_copies[j]]

        m = Machine("Dummy")
        m.assign_ability(Ability._dummy)
        machines.append(m)
        for j in new_job_list:
            for o in j.iter_operations():
                if o.is_dummy():
                    m.assign_operation(o)
                    o.assign_machine(m)
        # Sanity Check
        for m in machines:
            for o in m.get_assigned_operations():
                if o.get_assigned_machine() != m:
                    print("Failure during operation assignement. Error.")
                    raise AssertionError
            assert m.get_load() <= self.cmax, "Overloaded machine " + str(m)
        for j in new_job_list:
            for o in j.iter_operations():
                if o.get_assigned_machine() is None:
                    for m in machines:
                        if m.get_load() + o.get_proc_time() <= self.cmax and m.get_assigned_abilities().issuperset(o.get_abilities()):
                            m.assign_operation(o)
                            o.assign_machine(m)
                            break
                    else:
                        m = Machine(counter)
                        m.assign_operation(o)
                        o.assign_machine(m)
                        m.set_assigned_abilities(o.get_abilities())

        return machines, new_job_list


def max_n_iterator(all_configs, n):
    start = [0]*len(all_configs)
    for y in iterator_helper(start, n):
        yield y


def iterator_helper(configs_list, i):
    if i == 0:
        yield configs_list
    else:
        yield configs_list
        for j in range(len(configs_list)):
            y_inc = configs_list.copy()
            y_inc[j] += 1
            for y_new in iterator_helper(y_inc[j:], i - 1):
                yield configs_list[:j] + y_new


# Lenstra, Shmoys, Tardos Rounding
def LST_Rounding(operations, machines, cmax_in):
    c_max_out = 3*cmax_in

    mod = Model("LST")

    valid_combinations = [(o, m) for o in operations for m in machines if o.get_abilities().issubset(m.get_assigned_abilities())]
    x = mod.addVars(valid_combinations, vtype=GRB.CONTINUOUS, name="x")

    mod.addConstrs(quicksum(x[o, m] for m in machines if (o, m) in valid_combinations) == 1 for o in operations)

    mod.addConstrs(quicksum(x[o, m]*o.get_proc_time() for o in operations if (o, m) in valid_combinations) <= cmax_in for m in machines)
    mod.setParam("LogToConsole", 0)
    mod.optimize()

    if mod.getAttr("Status") == GRB.INFEASIBLE:
        return c_max_out
    else:
        integral_assignments = [(o, m) for (o, m) in valid_combinations if x[o, m].x >= .9999]
        fractional_assignments = [(o, m) for (o, m) in valid_combinations if .9999 > x[o, m].x > 0.0001]

        G = nx.Graph()

        G.add_edges_from(fractional_assignments)
        matching = nx.algorithms.matching.max_weight_matching(G)

        for t in matching:
            if type(t[0]) != Machine:
                integral_assignments.append(t)
            else:
                integral_assignments.append((t[1], t[0]))

        for (o, m) in integral_assignments:
            assert len(integral_assignments) == len(operations)
            o.assign_machine(m)
            m.assign_operation(o)
        c_max_out = max(m.get_load() for m in machines)
        return c_max_out


def duplicate_machines(operations: list, machines: list, c_max: int) -> list:
    for o in operations:
        o.assign_machine(None)

    former_len = len(machines)
    for i in range(former_len):
        m = machines[former_len - 1 - i]
        # Delete all machines without assigned operations...
        if len(m.get_assigned_operations()) == 0:
            del machines[former_len - 1 - i]
        else:
            # ... and duplicate those whose capacity is exceeded...
            if m.get_load() > c_max:
                m2 = Machine(str(m.name) + "_duplicate")
                m2.set_assigned_abilities(m.get_assigned_abilities())
                # ... by removing its longest job
                longest_operation = max(m.get_assigned_operations(), key=lambda o: o.get_proc_time())
                m2.assign_operation(longest_operation)
                m.remove_operation(longest_operation)
                machines.append(m2)
    for m in machines:
        for o in m.get_assigned_operations():
            o.assign_machine(m)

    # Sanity Check
    for m in machines:
        for o in m.get_assigned_operations():
            if o.get_assigned_machine() != m:
                print("Failure during operation assignement. Error.")
                raise AssertionError
    for o in operations:
        if o.get_assigned_machine() is None:
            print("Unassigned operation. Error.")
            raise AssertionError
    return machines

