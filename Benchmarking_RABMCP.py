from RandomInputGeneration.DataGenerator import DataGenerator
from Optimizer import Optimizer
import random as r
from Heuristics.MaxKHeuristic import MaxKHeuristic
from Heuristics.RABMCP_MIP import RABMCP_MIP
from Heuristics.SmallKHeuristic import SmallKHeuristic
from Heuristics.ColumnGeneration import ColumnGeneration
from typing import List


def heur(heur_method, heur_method_str, output_log):
    output_log += heur_method_str + "_lvl" + str(i) + "_seed" + str(seed) + "_run" + str(run) + "\r\n"

    output_log += "LM\r\n"
    j_copies = [j_new for j in jobs for j_new in j.to_low_multiplicity()]
    h = heur_method(j_copies, abilities, makespan, t_limit)
    h.verbose = True
    h.set_output_file(heur_method_str + "LM_lvl" + str(i) + "_seed" + str(seed) + "_run" + str(run) + ".txt")
    try:
        machines = h.solve()
    except AssertionError as e:
        output_log += "ASSERTION ERROR: " + str(e) + "\r\n"
    except Exception as e:
        output_log += "ERROR: " + str(e) + "\r\n"
    else:
        if heur_method == RABMCP_MIP or heur_method == ColumnGeneration:
            lb = h._lower_bound
            output_log += "\tLB: " + str(lb) + "\r\n"
        ub = sum(m.get_cost() for m in machines)
        output_log += "\tUB: " + str(ub) + "\r\n"

    output_log += "HM\r\n"
    h = heur_method(jobs, abilities, makespan, t_limit)
    h.verbose = True
    h.set_output_file(heur_method_str + "HM_lvl" + str(i) + "_seed" + str(seed) + "_run" + str(run) + ".txt")
    h.high_multiplicty = True
    h.multiplier = hm
    try:
        machines, jobs_new = h.solve()
    except AssertionError as e:
        output_log += "ASSERTION ERROR: " + str(e) + "\r\n"
    except Exception as e:
        output_log += "ERROR: " + str(e) + "\r\n"
    else:
        if heur_method == RABMCP_MIP or heur_method == ColumnGeneration:
            lb = h._lower_bound
            output_log += "\tLB: " + str(lb) + "\r\n"
        ub = sum(m.get_cost() for m in machines)
        output_log += "\tUB: " + str(ub) + "\r\n"

        output_log += "-------------------------------\r\n"
    return output_log


if __name__ == '__main__':
    num_difficulties = 4
    num_instances = 3
    num_runs = 3
    timelimit = [300, 3600, 3*3600, 8 * 3600]

    output_log = ""

    # Parameters
    num_abilities = [3, 5, 15, 25]
    num_jobs = [2, 5, 10, 20]
    max_num_operations = [5, 5, 25, 50]
    cost_min = [1, 1, 1, 1]
    cost_max = [10, 10, 10, 25]

    high_multiplicity = [3, 10, 20, 50]

    for i in range(num_difficulties):
        for seed in range(num_instances):
            num_ab = num_abilities[i]
            num_j = num_jobs[i]
            max_num_o = max_num_operations[i]
            c_min = cost_min[i]
            c_max = cost_max[i]

            t_limit = timelimit[i]

            hm = high_multiplicity[i]

            dg = DataGenerator(seed, num_ab, num_j, max_num_o, c_min, c_max)
            dg.high_multiplicity = hm
            jobs, abilities = dg.generate()
            DataGenerator.print_jobs(jobs, "Output/test_" + str(i) + "_" + str(seed) + ".tex")
            dg.print_instance("Output/test_" + str(i) + "_" + str(seed) + ".txt")

            operations_hm = [o for j in jobs for o in j.iter_operations()]

            total_proc_time = sum(o.get_proc_time() * hm for o in operations_hm)
            longest_proc_time = max(o.get_proc_time() for o in operations_hm)

            makespan = longest_proc_time * (hm/3)

            for run in range(num_runs):
                # Column Generation+
                # Column Generation without post-processing can be read from gurobi logs
                output_log = heur(ColumnGeneration, "ColumnGeneration", output_log)
                print(output_log)

            for run in range(num_runs):
                # SmallKHeuristic
                output_log = heur(SmallKHeuristic, "SmallKHeuristic", output_log)
                print(output_log)

            for run in range(num_runs):
                # MIP
                output_log = heur(RABMCP_MIP, "MIP", output_log)
                print(output_log)

            for run in range(num_runs):
                # MaxKHeuristic
                output_log = heur(MaxKHeuristic, "MaxKHeuristic", output_log)
                print(output_log)
