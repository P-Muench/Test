import math
from gurobipy import *
from typing import List, Set, TYPE_CHECKING, Dict, Union
from Datastructures.Job import Job
from Datastructures.Ability import Ability
from Datastructures.OperationCollection import OperationCollection
from Datastructures.OperationCollection_HM import OperationCollection_HM
from Datastructures.AbilityConfigurations import AbilityConfigurations
from Datastructures.Operation import Operation
from Datastructures.Machine import Machine
from Heuristics.RABMCP_MIP import RABMCP_MIP
import random
import sys
from Heuristics.Heuristic import Heuristic
import time


class ColumnGeneration(Heuristic):
    # implements the column generation heuristc as in Chapter 5.5.3

    time_spent_postprocessing = 60
    post_post_processing = False

    def __init__(self, jobs: "List[Job]", abilities: "List[Ability]", cmax: float, timeout: float,
                 upper_bound: float = sys.maxsize) -> None:
        super(ColumnGeneration, self).__init__(jobs, abilities, cmax, timeout, upper_bound)
        self._all_configs = []
        self._sub_operations = {}
        self._lower_bound = 0
        self._only_relaxed = False

        self._config_counter = 0
        random.seed(1)

    def _solve_low_multiplicity(self):
        # For readability
        operations = self.operations

        start_time = time.clock()

        # Preparation: List all possible ability configurations and store for each one which operations can run on it
        ac = AbilityConfigurations(self.abilities)
        for o in self.operations:
            ac.add_subset(o.get_abilities())
        self._all_configs = list(ac.iter_configs())
        self._sub_operations = {tuple(config): [o for o in self.operations if o.get_abilities().issubset(config)] for config in self._all_configs}

        # Create an initial feasible solution for the master problem. Store operation_collections in powerset
        # operation_collection know their cost and resource requirements
        collections_containing_o = {}
        powerset: Set[OperationCollection] = set()
        for o in operations:
            oc: OperationCollection = OperationCollection()
            oc.add_operation(o)
            powerset.add(oc)
            collections_containing_o[o] = [oc]

        # Master problem
        mod = Model("ColumnGeneration")
        # Variables to choose operation_collections or not
        x = mod.addVars(powerset, vtype=GRB.CONTINUOUS, name="x")
        # each operations must appear at least once in a collection -> partition
        constrs = mod.addConstrs(quicksum(x[p] for p in collections_containing_o[o]) >= 1 for o in operations)
        # minimize t
        mod.setObjective(quicksum(x[p] * p.get_cost() for p in powerset))
        mod.setParam("LogToConsole", 1 if self.verbose else 0)
        mod.setParam("LogFile", "")
        mod.optimize()

        print("Initial solution found with value " + str(round(mod.ObjVal, 2)) + "\nStarting Column Generation")

        counter = 0
        termination_by_timeout = False
        while True:
            if time.clock() - start_time > self.timeout*3.0/4:
                termination_by_timeout = True
                break
            new_ocs = self.dual({o: constrs[o].Pi for o in operations})
            if len(new_ocs) == 0:
                break
            else:
                verbosity = self.verbose and counter % 100 == 0
                verbosity = 1 if verbosity else 0
                for new_oc in new_ocs:
                    counter += 1
                    powerset.add(new_oc)
                    x[new_oc] = mod.addVar(vtype=GRB.CONTINUOUS, name="x", obj=new_oc.get_cost())
                    for o in new_oc:
                        mod.chgCoeff(constrs[o], x[new_oc], 1)
                        collections_containing_o[o].append(new_oc)
                mod.setParam("LogToConsole", verbosity)
                mod.setParam("LogFile", "")
                mod.optimize()

        if termination_by_timeout:
            self._lower_bound = -1
        else:
            self._lower_bound = mod.ObjVal
        print("Added " + str(counter) + " variables through column generation.")
        mod = Model("ColumnGeneration_Full")

        x = {p: mod.addVar(vtype=GRB.BINARY, name="x" + str(p)) for p in powerset}

        mod.addConstrs(quicksum(x[p] for p in collections_containing_o[o]) >= 1 for o in operations)

        mod.setObjective(quicksum(x[p] * p.get_cost() for p in powerset))
        mod.setParam("LogToConsole", 1 if self.verbose else 0)
        mod.setParam("LogFile", self.output_filename)

        mod.setParam("TimeLimit", self.timeout - ColumnGeneration.time_spent_postprocessing - (time.clock() - start_time))

        mod.optimize()
        print("Found a solution with value " + str(round(mod.ObjVal, 2)))

        # Postprocessing
        machines = self._get_solution_low_multiplicity(x)
        machines, new__job_list = RABMCP_MIP.post_process_HM(self.jobs, self.operations, self.abilities, machines,
                                                            self.cmax, self._lower_bound, timelimit=ColumnGeneration.time_spent_postprocessing)
        return machines, new__job_list

    def dual(self, y: "Dict[Operation, float]"):
        new_ocs = []
        for i in range(len(self._all_configs)):
            config = self._all_configs[self._config_counter]

            sub_operations = self._sub_operations[tuple(config)]
            sub_y = {o: y[o] for o in sub_operations}
            ks_sol = self._solve_knapsack(sub_operations, sub_y, AbilityConfigurations.get_cost(config))
            if len(ks_sol) > 0:
                if self.high_multiplicty:
                    oc = OperationCollection_HM()
                    oc.add_operations(ks_sol)
                else:
                    oc = OperationCollection()
                    oc.add_operations(ks_sol)
                # return 1 violated row of dual. If commented out, every violated row will be added as variable
                return [oc]
                new_ocs.append(oc)
            else:
                self._config_counter = (self._config_counter + 1) % len(self._all_configs)
        return new_ocs

    def _solve_knapsack(self, operations: "List[Operation]", y, val) -> Union[List[Operation], Dict[Operation, int]]:
        ks = Model("Knapsack")

        if self.high_multiplicty:
            x = ks.addVars(operations, vtype=GRB.INTEGER, name="x")
            for o in operations:
                ks.addConstr(x[o] <= o._multiplier)
        else:
            x = ks.addVars(operations, vtype=GRB.BINARY, name="x")

        ks.addConstr(quicksum(x[o]*o.get_proc_time() for o in operations) <= self.cmax)

        ks.setObjective(quicksum(x[o]*y[o] for o in operations), GRB.MAXIMIZE)

        ks.setParam("OutputFlag", 0)
        ks.optimize()

        if ks.ObjVal - val > 1E-4:
            if self.high_multiplicty:
                return {o: int(x[o].x + 0.4) for o in operations if x[o].x > .5}
            else:
                return [o for o in operations if x[o].x > .5]
        else:
            return []

    def _solve_high_multiplicity(self):
        operations = self.operations

        start_time = time.clock()

        ac = AbilityConfigurations(self.abilities)
        for o in self.operations:
            ac.add_subset(o.get_abilities())
        self._all_configs = list(ac.iter_configs())
        self._sub_operations = {tuple(config): [o for o in self.operations if o.get_abilities().issubset(config)] for
                                config in self._all_configs}

        collections_containing_o = {}
        powerset: Set[OperationCollection] = set()
        for o in operations:
            max_on_machine = int(math.floor(self.cmax / float(o.get_proc_time()))) if o.get_proc_time() != 0 else o._multiplier
            oc: OperationCollection_HM = OperationCollection_HM()
            oc.add_operation(o, num=1)
            powerset.add(oc)
            collections_containing_o[o] = [oc]

        mod = Model("ColumnGeneration")

        x = {}
        for p in powerset:
            x[p] = mod.addVar(vtype=GRB.CONTINUOUS, name="x" + str(p))

        constrs = mod.addConstrs(quicksum(x[p]*p.get_num_of(o) for p in collections_containing_o[o]) == o._multiplier for o in operations)

        mod.setObjective(quicksum(x[p] * p.get_cost() for p in powerset))
        mod.setParam("LogToConsole", 1 if self.verbose else 0)
        mod.optimize()

        print("Initial solution found with value " + str(round(mod.ObjVal, 2)) + "\nStarting Column Generation")

        counter = 0
        termination_by_timeout = False
        while True:
            if time.clock() - start_time > self.timeout*3.0/4:
                termination_by_timeout = True
                break
            new_ocs = self.dual({o: constrs[o].Pi for o in operations})
            if len(new_ocs) == 0:
                break
            else:
                verbosity = self.verbose and counter % 100 == 0
                verbosity = 1 if verbosity else 0
                for new_oc in new_ocs:
                    counter += 1
                    powerset.add(new_oc)
                    x[new_oc] = mod.addVar(vtype=GRB.CONTINUOUS, name="x" + str(new_oc), obj=new_oc.get_cost())
                    for o in new_oc:
                        coeff = new_oc[o]
                        mod.chgCoeff(constrs[o], x[new_oc], coeff)
                        collections_containing_o[o].append(new_oc)
                mod.setParam("LogToConsole", verbosity)
                mod.setParam("LogFile", "")
                mod.optimize()

        if termination_by_timeout:
            self._lower_bound = -1
        else:
            self._lower_bound = mod.ObjVal
        print("Added " + str(counter) + " variables through column generation.")
        if self._only_relaxed:
            return
        for o in operations:
            max_on_machine = int(math.floor(self.cmax / float(o.get_proc_time()))) if o.get_proc_time() != 0 else o._multiplier
            for i in range(1, min(max_on_machine, o._multiplier)):
                oc: OperationCollection_HM = OperationCollection_HM()
                oc.add_operation(o, num=i + 1)
                if oc.get_resource() > 1:
                    break
                else:
                    powerset.add(oc)
                    collections_containing_o[o] = [oc]

        mod = Model("ColumnGeneration_Full")

        x = {p: mod.addVar(vtype=GRB.INTEGER, name="x" + str(p)) for p in powerset}

        mod.addConstrs(
            quicksum(x[p] * p.get_num_of(o) for p in collections_containing_o[o]) >= o._multiplier for o in operations)

        mod.setObjective(quicksum(x[p] * p.get_cost() for p in powerset))
        mod.setParam("LogToConsole", 1 if self.verbose else 0)

        mod.setParam("TimeLimit", self.timeout - 180 - (time.clock() - start_time))
        mod.setParam("LogFile", self.output_filename)

        mod.optimize()

        print("Found a solution with value " + str(round(mod.ObjVal, 2)))

        x = {p: x[p] for p in powerset if x[p].x > .5}

        machines, new__job_list = self._get_solution_high_multiplicity(x)
        machines, new__job_list = RABMCP_MIP.post_process_HM(self.jobs, self.operations, self.abilities, machines, self.cmax, self._lower_bound, ColumnGeneration.time_spent_postprocessing)
        if ColumnGeneration.post_post_processing:
            machines, new__job_list = RABMCP_MIP.post_post_process_HM(self.jobs, self.operations, self.abilities, machines, self.cmax, self._lower_bound, ColumnGeneration.time_spent_postprocessing)
        return machines, new__job_list

    def _get_solution_low_multiplicity(self, x):
        machines = []
        counter = 0
        for p in x:
            if x[p].x > .5:
                m = Machine(str(counter))
                m.set_assigned_abilities(p.get_abilities())
                for o in p:
                    if o.get_assigned_machine() is None:
                        m.assign_operation(o)
                        o.assign_machine(m)
                if len(m.get_assigned_operations()) > 0:
                    machines.append(m)
                    counter += 1

        ColumnGeneration._sanity_check(machines, self.jobs)
        return machines

    def _get_solution_high_multiplicity(self, x):
        assigned_copies = {o: 0 for o in self.operations}
        j_copies = {j: j.to_low_multiplicity() for j in self.jobs}
        mapping_o = {o: [] for o in self.operations}
        for j in self.jobs:
            for pos, o in enumerate(j.graph.nodes):
                for j_copy in j_copies[j]:
                    mapping_o[o].append(j_copy.graph.nodes[pos])

        machines = []
        counter = 0
        for operations_collection in x:
            if x[operations_collection].x > .5:
                for i in range(int(x[operations_collection].x + .4)):
                    m = Machine(str(counter))
                    m.set_assigned_abilities(operations_collection.get_abilities())
                    for o in operations_collection:
                        for num_copy in range(operations_collection[o]):
                            if assigned_copies[o] < len(mapping_o[o]):
                                o_copy = mapping_o[o][assigned_copies[o]]
                                m.assign_operation(o_copy)
                                o_copy.assign_machine(m)
                                assigned_copies[o] += 1
                    machines.append(m)
                    counter += 1

        new_job_list: List[Job] = [j_new for j in self.jobs for j_new in j_copies[j]]

        ColumnGeneration._sanity_check(machines, new_job_list)

        return machines, new_job_list

    def _compute_lb(self):
        # For readability
        operations = self.operations

        start_time = time.clock()

        # Preparation: List all possible ability configurations and store for each one which operations can run on it
        ac = AbilityConfigurations(self.abilities)
        for o in self.operations:
            ac.add_subset(o.get_abilities())
        self._all_configs = list(ac.iter_configs())
        self._sub_operations = {tuple(config): [o for o in self.operations if o.get_abilities().issubset(config)] for config in self._all_configs}

        # Create an initial feasible solution for the master problem. Store operation_collections in powerset
        # operation_collection know their cost and resource requirements
        collections_containing_o = {}
        powerset: Set[OperationCollection] = set()
        for o in operations:
            oc: OperationCollection = OperationCollection()
            oc.add_operation(o)
            powerset.add(oc)
            collections_containing_o[o] = [oc]

        # Master problem
        mod = Model("ColumnGeneration")
        # Variables to choose operation_collections or not
        x = mod.addVars(powerset, vtype=GRB.CONTINUOUS, name="x")
        # each operations must appear at least once in a collection -> partition
        constrs = mod.addConstrs(quicksum(x[p] for p in collections_containing_o[o]) >= 1 for o in operations)
        # minimize t
        mod.setObjective(quicksum(x[p] * p.get_cost() for p in powerset))
        mod.setParam("LogToConsole", 1 if self.verbose else 0)
        mod.setParam("LogFile", "")
        mod.optimize()

        print("Initial solution found with value " + str(round(mod.ObjVal, 2)) + "\nStarting Column Generation")

        counter = 0
        termination_by_timeout = False
        while True:
            if time.clock() - start_time > self.timeout*3.0/4:
                termination_by_timeout = True
                break
            new_ocs = self.dual({o: constrs[o].Pi for o in operations})
            if len(new_ocs) == 0:
                break
            else:
                verbosity = self.verbose and counter % 100 == 0
                verbosity = 1 if verbosity else 0
                for new_oc in new_ocs:
                    counter += 1
                    powerset.add(new_oc)
                    x[new_oc] = mod.addVar(vtype=GRB.CONTINUOUS, name="x", obj=new_oc.get_cost())
                    for o in new_oc:
                        mod.chgCoeff(constrs[o], x[new_oc], 1)
                        collections_containing_o[o].append(new_oc)
                mod.setParam("LogToConsole", verbosity)
                mod.setParam("LogFile", "")
                mod.optimize()

        if termination_by_timeout:
            self._lower_bound = -1
        else:
            self._lower_bound = mod.ObjVal
        print("Added " + str(counter) + " variables through column generation.")

    @staticmethod
    def _sanity_check(machines, jobs):
        # Sanity Check:
        counter = 0
        for j in jobs:
            for o in j.iter_operations():
                counter += 1
                if o.get_assigned_machine() is None:
                    raise AssertionError("Operation " + str(o) + " not assigned")
        assigned_operations = []
        for m in machines:
            assigned_operations += m.get_assigned_operations()
        assert len(assigned_operations) == counter, "Operations added to multiple machines"


def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)
