import sys
import time

from Datastructures.Machine import Machine
from Heuristics.JobShopMIP import JobShopMIP
from Heuristics.MaxKHeuristic import MaxKHeuristic
from Heuristics.RABMCP_MIP import RABMCP_MIP
from Heuristics.SmallKHeuristic import SmallKHeuristic
from Heuristics.JobShopApproximation import JobShopApproximation
from Heuristics.ColumnGeneration import ColumnGeneration
from Heuristics.Full_ABMCP import Full_ABMCP
import random as r
import plotly as py
import plotly.figure_factory as ff
from typing import List, Set, Dict


# Implements the Fix-and-Spread Algorithm
class Optimizer:
    MIP_MODEL = 1
    SMALL_K_HEURISTIC = 2
    MAX_K_HEURISTIC = 3
    FULL_MIP_MODEL = 4
    COLUMN_GENERATION = 5

    JSA_MIP = 6
    JSA_APPROX = 7

    # Initial factor for search can be determined here. T-Bar = (p_min + T) / factor
    initial_binary_factor = 2

    def __init__(self, jobs: list, abilities: list, c_max: int, method=MIP_MODEL, timelimit_heuristic=sys.maxsize, timelimit=sys.maxsize):
        self.machines: List[Machine] = []
        self.jobs = jobs
        self._seed = 0
        self._method = method
        self.c_max = c_max
        self._timelimit_heuristic = timelimit_heuristic
        self._timelimit_scheduling = timelimit_heuristic
        self._timelimit = timelimit
        self._operations = [o for j in self.jobs for o in j.iter_operations()]
        self.abilities = abilities
        self.verbose = False
        self.choose_paths_randomly = False
        self.opt_choose_paths = False
        self.JSA_repetitions = 20
        self.JSA_method = Optimizer.JSA_APPROX
        self.outfile_name = "Gantt-Sol"
        self.max_binary = 1000

        self._low_multiplicity_jobs = []
        self._low_multiplicity_operations = []

        self.high_multiplicty = False
        assert c_max >= max(j.get_p_sum() for j in self.jobs)

    def set_seed(self, seed):
        self._seed = seed

    def set_timelimit_subproblem(self, timelimit):
        self._timelimit_heuristic = timelimit

    def set_timelimit(self, timelimit):
        self._timelimit = timelimit

    def set_solution_method(self, method):
        self._method = method

    def _clear(self):
        for o in self._operations:
            o.clear_solution()
        for m in self.machines:
            m.clear_solution()
        self._low_multiplicity_jobs = []

    def optimize(self):
        self._clear()
        self._low_multiplicity_jobs = [j_new for j in self.jobs for j_new in j.to_low_multiplicity()]
        self._low_multiplicity_operations = [o for j in self._low_multiplicity_jobs for o in j.iter_operations()]
        self._obtain_initial_solution()
        if not self._method == Optimizer.FULL_MIP_MODEL:
            self._binary_search()
        else:
            heuristic = Full_ABMCP(self._low_multiplicity_jobs, self.abilities, self.c_max, self._timelimit)
            heuristic.verbose = self.verbose
            self.machines = heuristic.solve()

    def _binary_search(self):
        start_time = time.clock()

        longest_proc_time = max(o.get_proc_time() for j in self.jobs for o in j.iter_operations())

        upper_bound = self.c_max
        lower_bound = longest_proc_time

        best_schedule = Optimizer.get_schedule_from_operations(self._low_multiplicity_operations)
        best_sol_value = self.get_objective_value()
        best_machines = self.machines
        job_temp = self._low_multiplicity_jobs
        operations_temp = self._low_multiplicity_operations

        counter = 0
        c_max_relax = int((upper_bound + lower_bound) / Optimizer.initial_binary_factor)
        while upper_bound - lower_bound > 1:
            print("Time: %s seconds"%(time.clock() - start_time))
            if time.clock() - start_time > self._timelimit:
                print("Terminated Binary Search through Time Limit")
                break
            if counter >= self.max_binary:
                break

            print("T-Bar: " + str(round(c_max_relax, 2)))

            heuristic = None
            if self._method == Optimizer.MIP_MODEL:
                heuristic = RABMCP_MIP(self.jobs, self.abilities, c_max_relax, self._timelimit_heuristic)
            if self._method == Optimizer.SMALL_K_HEURISTIC:
                heuristic = SmallKHeuristic(self.jobs, self.abilities, c_max_relax, self._timelimit_heuristic, best_sol_value)
            if self._method == Optimizer.MAX_K_HEURISTIC:
                heuristic = MaxKHeuristic(self.jobs, self.abilities, c_max_relax, self._timelimit_heuristic, best_sol_value)
            if self._method == Optimizer.COLUMN_GENERATION:
                heuristic = ColumnGeneration(self.jobs, self.abilities, c_max_relax, self._timelimit_heuristic, best_sol_value)

            heuristic.verbose = self.verbose
            if self.high_multiplicty:
                heuristic.high_multiplicty = self.high_multiplicty
                self.machines, self._low_multiplicity_jobs = heuristic.solve()

                self._low_multiplicity_operations = [o for j in self._low_multiplicity_jobs for o in
                                                     j.iter_operations()]
            else:
                self.machines = heuristic.solve()
            if self._schedule(counter):
                if self.get_schedule_length() <= self.c_max:
                    lower_bound = c_max_relax
                    if self.get_objective_value() < best_sol_value:
                        best_schedule = Optimizer.get_schedule_from_operations(self._low_multiplicity_operations)
                        best_machines = self.machines
                        best_sol_value = self.get_objective_value()
                        job_temp = self._low_multiplicity_jobs
                        operations_temp = self._low_multiplicity_operations
            else:
                upper_bound = c_max_relax
            c_max_relax = (upper_bound + lower_bound) / 2.0
            counter += 1

        # Restore best found solution
        self._low_multiplicity_jobs = job_temp
        self._low_multiplicity_operations = operations_temp
        self.machines = best_machines
        for m in self.machines:
            for o in m.get_assigned_operations():
                o.assign_machine(m)
                o.start_at(best_schedule[o])

    @staticmethod
    def get_schedule_from_operations(operations):
        return {o: o.get_start() for o in operations}

    def _schedule(self, i: int):
        if self.JSA_method == Optimizer.JSA_APPROX:
            if self.high_multiplicty:
                JSA = JobShopApproximation(self._low_multiplicity_jobs, self.abilities, self.c_max, self._timelimit_scheduling, self.machines)
            else:
                JSA = JobShopApproximation(self.jobs, self.abilities, self.c_max, self._timelimit_scheduling, self.machines)
            JSA.choose_paths_randomly = self.choose_paths_randomly
            JSA.opt_choose_paths = self.opt_choose_paths
            JSA.set_max_num_iterations(self.JSA_repetitions)
            JSA.set_seed(self._seed + i)
            JSA.verbose = self.verbose
            return JSA.solve()
        else:
            if self.high_multiplicty:
                JSA = JobShopMIP(self._low_multiplicity_jobs, self.abilities, self.c_max, self._timelimit_scheduling, self.machines)
            else:
                JSA = JobShopMIP(self.jobs, self.abilities, self.c_max, self._timelimit_scheduling, self.machines)
            JSA.verbose = self.verbose
            return JSA.solve()

    def get_schedule_length(self) -> float:
        if self.high_multiplicty:
            return max(o.get_start() + o.get_proc_time() for o in self._low_multiplicity_operations)
        else:
            return max(o.get_start() + o.get_proc_time() for o in self._operations)

    def get_objective_value(self) -> float:
        return sum(m.get_cost() for m in self.machines)

    def _obtain_initial_solution(self):
        counter = 0
        for j in self._low_multiplicity_jobs:
            for o in j.iter_operations():
                m = Machine(str(counter))
                m.set_assigned_abilities(o.get_abilities())

                m.assign_operation(o)
                o.assign_machine(m)
                self.machines.append(m)

                counter += 1

        for j in self._low_multiplicity_jobs:
            j.get_earliest_start(set_initial_path=True)

    def print_solution(self, discretization: int):
        try:
            print("Output_v1")
            for m in self.machines:
                print(m)

            print("\nOutput_v2")
            for m in self.machines:
                print(str(m.name))
                print("Abilities: " + " ".join(a.name for a in m.get_assigned_abilities()))
                print("Operations: " + str({o.name: o.get_start() for o in m.get_assigned_operations()}))
        except Exception as e:
            print(e)

        df = []
        for m in self.machines:
            for j in self._low_multiplicity_jobs:
                for o in j.iter_operations():
                    if o in m.get_assigned_operations():
                        df += [dict(Task="M" + str(m.name), Start=o.get_start()*discretization, Finish=(o.get_start() + o.get_proc_time())*discretization, Resource=str(j), Name=o.id)]

        r.seed(1)
        colors = {str(j): (r.random(), r.random(), r.random()) for j in self._low_multiplicity_jobs}
        # Print Gantt Chart with plotly
        fig = ff.create_gantt(df, index_col='Resource', colors=colors, show_colorbar=True, group_tasks=True, showgrid_x=True,
                              showgrid_y=True, task_names='Name')
        fig["layout"]["xaxis"]["type"] = "linear"

        x = []
        y = []
        text = []
        for i in range(len(df)):
            # set text that appears while hovering over a job
            fig['data'][i].update(text=df[i]["Name"], hoverinfo="text")
            x.append(fig["layout"]["shapes"][i]['x0'] + (
                        - fig["layout"]["shapes"][i]['x0'] + fig["layout"]["shapes"][i]['x1']) / 2.0)
            y.append(fig["layout"]["shapes"][i]['y1'] + 0.1)
            text.append(df[i]["Name"])
        trace0 = py.graph_objs.Scatter(
            x=x,
            y=y,
            text=text,
            mode='text',
            hoverinfo='none',
            showlegend=False,
        )
        py.offline.plot(fig, filename='Output/' + str(self.outfile_name) + '_no_text.html')
        fig["data"].append(trace0)
        py.offline.plot(fig, filename='Output/' + str(self.outfile_name) + '_text.html')


