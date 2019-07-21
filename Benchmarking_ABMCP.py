from RandomInputGeneration.DataGenerator import DataGenerator
from Optimizer import Optimizer
import random as r
from Heuristics.MaxKHeuristic import MaxKHeuristic
from Heuristics.RABMCP_MIP import RABMCP_MIP
from Heuristics.SmallKHeuristic import SmallKHeuristic
from Heuristics.ColumnGeneration import ColumnGeneration
from typing import List
import math


if __name__ == '__main__':
    num_difficulties = 4
    num_instances = 3
    num_runs = 3
    timelimit = [300, 600, 3600, 4 * 3600]

    output_log = ""

    # Parameters
    num_abilities = [3, 5, 15, 25]
    num_jobs = [2, 5, 10, 20]
    max_num_operations = [5, 5, 25, 50]
    cost_min = [1, 1, 1, 1]
    cost_max = [10, 10, 10, 25]
    JSA_reps = [1000, 500, 2000, 750]

    high_multiplicity = [3, 10, 20, 50]

    for i in range(num_difficulties):
        for seed in range(num_instances):
            num_ab = num_abilities[i]
            num_j = num_jobs[i]
            max_num_o = max_num_operations[i]
            c_min = cost_min[i]
            c_max = cost_max[i]
            reps = JSA_reps[i]

            t_limit = timelimit[i]

            hm = high_multiplicity[i]

            dg = DataGenerator(seed, num_ab, num_j, max_num_o, c_min, c_max)
            dg.high_multiplicity = hm
            jobs, abilities = dg.generate()

            operations_hm = [o for j in jobs for o in j.iter_operations()]

            total_proc_time = sum(o.get_proc_time() * hm for o in operations_hm)
            longest_proc_time = max(o.get_proc_time() for o in operations_hm)

            makespan = longest_proc_time*(hm)*1.5

            for run in range(num_runs):
                # Fix and Spread
                opt = Optimizer(jobs, abilities, makespan)

                opt.set_seed(seed)
                opt.set_solution_method(Optimizer.COLUMN_GENERATION)
                opt.high_multiplicty = True

                opt.set_timelimit_subproblem(t_limit)
                opt._timelimit_scheduling = t_limit
                opt.set_timelimit(t_limit * 6)
                opt.choose_paths_randomly = True
                opt.opt_choose_paths = i <= 1
                opt.JSA_repetitions = reps
                opt.JSA_method = Optimizer.JSA_APPROX
                opt.verbose = True

                output_log += "F-and-S: Lvl_%s_Seed_%s_Run_%s\r\n" % (i, seed, run)
                try:
                    opt.optimize()
                    ub = sum(m.get_cost() for m in opt.machines)
                    output_log += "\tUB: " + str(ub) + "\r\n"
                except Exception as e:
                    output_log += "\tERROR: " + str(e) + "\r\n"
                print(output_log)

                for o in operations_hm:
                    o.clear_solution()

                opt.set_solution_method(Optimizer.FULL_MIP_MODEL)
                try:
                    opt.optimize()
                    output_log += "Model:\r\n"
                    ub = sum(m.get_cost() for m in opt.machines)
                    output_log += "\tUB: " + str(ub) + "\r\n"
                except Exception as e:
                    output_log += "\tERROR: " + str(e) + "\r\n"

                h = ColumnGeneration(jobs, abilities, makespan, t_limit * 5)
                h.verbose = True
                h.high_multiplicty = True
                h.multiplier = hm
                try:
                    h._only_relaxed = True
                    h.solve()
                    lb = h._lower_bound
                    output_log += "\tLB: " + str(lb) + "\r\n"
                except AssertionError as e:
                    output_log += "ASSERTION ERROR: " + str(e) + "\r\n"
                except Exception as e:
                    output_log += "ERROR: " + str(e) + "\r\n"

                output_log += "---------------" + "\r\n"
                print(output_log)
