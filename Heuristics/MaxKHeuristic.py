import math
import time
from Datastructures.AbilityConfigurations import AbilityConfigurations
from Datastructures.Machine import Machine
from Datastructures.Job import Job
from Heuristics.Heuristic import Heuristic
import sys
from typing import List, Tuple


# Algorithms as developed in chapter 5.5
class MaxKHeuristic(Heuristic):

    def __init__(self, jobs, abilities, cmax, timeout, upper_bound=sys.maxsize):
        super().__init__(jobs, abilities, cmax, timeout, upper_bound)

    def _solve_low_multiplicity(self):
        start_time = time.clock()

        abilities = self.abilities
        cmax = self.cmax
        timeout = self.timeout
        operations = self.operations

        ac = AbilityConfigurations(abilities)
        for o in operations:
            ac.add_subset(o.get_abilities())
        all_configs = ac.iter_configs()
        max_cost = self.upper_bound

        machines = []
        unassigned = sorted(operations, key=lambda o: o.get_proc_time())
        machine_counter = 0

        while len(unassigned) > 0:
            if time.clock() - start_time > timeout:
                print("Timelimit reached.")
                break
            m = Machine(str(machine_counter))
            minimum = max_cost
            B = set()
            I_B = list()
            for a in all_configs:
                cost = AbilityConfigurations.get_cost(a)
                p_sum = 0
                I_temp = []
                for o in unassigned:
                    if o.get_abilities().issubset(a):
                        if p_sum + o.get_proc_time() <= cmax:
                            p_sum += o.get_proc_time()
                            I_temp.append(o)
                        else:
                            break
                if float(cost) < minimum * len(I_temp):
                    minimum = float(cost) / len(I_temp)
                    B = a
                    I_B = I_temp

            print("Found machine config: " + str(B))
            print("Assigned no. of jobs: " + str(len(I_B)))
            machines.append(m)
            m.set_assigned_abilities(B)
            for o in I_B:
                m.assign_operation(o)
                unassigned.remove(o)
            machine_counter += 1

        # If Timelimit reached, create rest of machines stupidly
        for o in unassigned:
            m = Machine(str(machine_counter))
            m.set_assigned_abilities(o.get_abilities())

            m.assign_operation(o)
            machines.append(m)

            machine_counter += 1

        for m in machines:
            for o in m.get_assigned_operations():
                o.assign_machine(m)

        print("Found " + str(machine_counter - 1) + " machines")
        return machines

    def _solve_high_multiplicity(self) -> Tuple[List[Machine], List[Job]]:
        """

        :return:
        """
        start_time = time.clock()
        abilities = self.abilities
        cmax = self.cmax
        timeout = self.timeout
        operations = self.operations

        ac = AbilityConfigurations(abilities)
        for o in operations:
            ac.add_subset(o.get_abilities())
        all_configs = ac.iter_configs()
        max_cost = self.upper_bound

        machines_temp = []
        unassigned = sorted(operations, key=lambda o: o.get_proc_time())
        left_to_assign = {o: o._multiplier for o in self.operations}

        while len(unassigned) > 0:
            if time.clock() - start_time > timeout:
                print("Timelimit reached.")
                break
            m = {"operations": {}, "abilities": set()}
            minimum = max_cost
            for a in all_configs:
                cost = AbilityConfigurations.get_cost(a)
                p_sum = 0
                I_temp = {}
                num_total_assigned_operations = 0
                for o in unassigned:
                    if o.get_abilities().issubset(a):
                        if o.get_proc_time() == 0:
                            num_assignable_operations = left_to_assign[o]
                        else:
                            num_assignable_operations = min(math.floor((cmax - p_sum) / (o.get_proc_time())),
                                                            left_to_assign[o])
                        if num_assignable_operations > 0:
                            I_temp[o] = num_assignable_operations
                            num_total_assigned_operations += num_assignable_operations
                            p_sum += num_assignable_operations * o.get_proc_time()
                        else:
                            break
                if float(cost) < minimum * num_total_assigned_operations:
                    minimum = float(cost) / num_total_assigned_operations
                    m = {"operations": I_temp, "abilities": a}
            machines_temp.append(m)
            for o, assigned_o in m["operations"].items():
                left_to_assign[o] -= assigned_o
                if left_to_assign[o] == 0:
                    unassigned.remove(o)

        # If Timelimit reached, create rest of machines stupidly
        for o in unassigned:
            while left_to_assign[o] > 0:
                if o.get_proc_time() == 0:
                    num_assignable_operations = left_to_assign[o]
                else:
                    num_assignable_operations = min(math.floor(cmax / (o.get_proc_time())), left_to_assign[o])
                m = {"operations": {o: num_assignable_operations}, "abilities": o.get_abilities()}
                machines_temp.append(m)
                left_to_assign[o] -= num_assignable_operations

        return self._get_machines_HM(machines_temp)

    def _get_machines_HM(self, machines_temp):
        # Convert to low multiplicity
        assigned_copies = {o: 0 for o in self.operations}
        j_copies = {j: j.to_low_multiplicity() for j in self.jobs}
        mapping_o = {o: [] for o in self.operations}
        for j in self.jobs:
            for pos, o in enumerate(j.graph.nodes):
                for j_copy in j_copies[j]:
                    mapping_o[o].append(j_copy.graph.nodes[pos])

        machines = []
        counter = 0
        for m_temp in machines_temp:
            m = Machine(counter)
            machines.append(m)
            m.set_assigned_abilities(m_temp["abilities"])
            for (o, num) in m_temp["operations"].items():
                for i in range(num):
                    o_copy = mapping_o[o][assigned_copies[o]]
                    m.assign_operation(o_copy)
                    o_copy.assign_machine(m)
                    assigned_copies[o] += 1
            counter += 1

        new_job_list = [j_new for j in self.jobs for j_new in j_copies[j]]

        # Sanity Check
        for m in machines:
            for o in m.get_assigned_operations():
                if o.get_assigned_machine() != m:
                    print("Failure during operation assignement. Error.")
                    raise AssertionError
        for j in new_job_list:
            for o in j.iter_operations():
                if o.get_assigned_machine() is None:
                    print("Unassigned operation. Error.")
                    raise AssertionError

        return machines, new_job_list
