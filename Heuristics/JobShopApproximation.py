import sys
from gurobipy import *
import random as r
import math
from Datastructures.Job import *
import _heapq
from Heuristics.Heuristic import Heuristic
import _multiprocessing
from joblib import Parallel, delayed
from time import clock


class JobShopApproximation(Heuristic):

    def __init__(self, jobs, abilities, cmax, timeout, machines, upper_bound=sys.maxsize):
        super().__init__(jobs, abilities, cmax, timeout, upper_bound)
        self.max_it = 1000
        self.machines = machines

        self.choose_paths_randomly = False
        self.opt_choose_paths = False

    def set_max_num_iterations(self, num):
        self.max_it = num

    def set_seed(self, seed):
        r.seed(seed)

    def _choose_path(self):
        operations = self.operations
        jobs = self.jobs
        machines = self.machines

        psum_t = {j: j.get_p_sum() for j in jobs}

        if self.choose_paths_randomly and not self.opt_choose_paths:
            # compute in parallel
            lbd = lambda x: x.get_earliest_start(set_initial_path=True, random_path=self.choose_paths_randomly)
            foo = Parallel()(delayed(lbd)(j) for j in jobs)
        if self.opt_choose_paths:
            print("Building path model")
            start_time = clock()

            ES = {j: j.get_earliest_start(set_initial_path=True, random_path=self.choose_paths_randomly) for j in jobs}
            LS = {j: j.get_latest_start() for j in jobs}

            T = {o: list(range(ES[j][o], LS[j][o] + 1)) for j in jobs for o in j.iter_operations()}
            max_T = max(T[o][-1] for o in operations)
            big_M = max_T + max(o.get_proc_time() for o in operations)

            mod = Model("choose paths")
            # StartingTime
            S = mod.addVars(((o, t) for o in operations for t in T[o]), vtype=GRB.BINARY, name="S")
            # number of parallel tasks on machines
            z = mod.addVar(vtype=GRB.INTEGER, name="z")
            # order of operations
            y = mod.addVars(((o1, o2) for j in jobs for o1 in j.iter_operations() for o2 in j.iter_operations() if not o1 == o2),
                            vtype=GRB.BINARY, name="y")

            # Fix start and end times of jobs
            for j in jobs:
                if abs(j.graph.in_node.get_start() - int(j.graph.in_node.get_start())) > .01:
                    print("Error with starting times for job " + str(j))
                mod.addConstr(S[j.graph.in_node, int(j.graph.in_node.get_start())] == 1)
                mod.addConstr(S[j.graph.out_node, int(j.graph.in_node.get_start() + psum_t[j] - j.graph.out_node.get_proc_time())] == 1)

            # assign every job once
            mod.addConstrs(quicksum(S[o, t] for t in T[o]) == 1 for o in operations)

            # Max parallel tasks on machine
            def get_lhs(m_var, t_var):
                return quicksum(
                    S[o_var, t2] for o_var in m_var.get_assigned_operations() for t2 in range(max(t_var - o_var.get_proc_time() + 1, 0), t_var + 1)
                    if t2 in T[o_var]) - z

            # Create Constraints in parallel
            lh_side = Parallel()(delayed(get_lhs)(m, t) for m in machines for t in range(max_T))
            mod.addConstrs(lh_side[i] <= 0 for i in range(len(lh_side)))

            if clock() - start_time > 60:
                return
            # Replaces:
            # mod.addConstrs(quicksum(
            #     S[o, t2] for o in m.get_assigned_operations() for t2 in range(max(t - o.get_proc_time() + 1, 0), t + 1) if
            #     t2 in T[o]) <= z
            #                for m in machines for t in range(max_T))

            # Fix given order through precedence relations
            mod.addConstrs(y[o1, o2] == 1 for j in jobs for o1 in j.iter_operations() for o2 in o1.iter_succ())

            # Preparation: save linear terms:
            lin_terms_lhs = {}
            lin_terms_rhs = {}
            for j in jobs:
                for o in j.iter_operations():
                    lin_terms_lhs[j, o] = quicksum((t + o.get_proc_time()) * S[o, t] for t in T[o])
                    lin_terms_rhs[j, o] = quicksum(t * S[o, t] for t in T[o])

            if clock() - start_time > 60:
                return
            # Replaces:
            # mod.addConstrs(
            #     quicksum((t + o1.get_proc_time()) * S[o1, t] for t in T[o1])
            #     <=
            #     quicksum(t * S[o2, t] for t in T[o2]) + big_M * y[o2, o1])

            for j in jobs:
                if clock() - start_time > 60:
                    return
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

            mod.setParam("LogToConsole", 0)
            mod.setParam("MIPFocus", 1)
            mod.setParam("TimeLimit", 60)

            for j in jobs:
                for o in j.iter_operations():
                    if not o == j.graph.out_node:
                        S[o, o.get_start()].start = 1

            # Minimize the number of parallel tasks on one machine
            mod.setObjective(z)
            mod.optimize()
            if mod.getAttr("Status") == GRB.INFEASIBLE:
                mod.computeIIS()
                mod.write("test.ilp")
                print("Postprocessing Infeasible")
                raise AssertionError
            else:
                for o in operations:
                    for t in T[o]:
                        if S[o, t].x > 0.5:
                            o.start_at(t)
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

    def solve(self):
        # Ceil every processing time to its nearest power of 2 (or 0)
        for o in self.operations:
            o.apply_transformation(lambda x: int(math.pow(2, math.ceil(math.log2(x))) if x > 0 else 0))

        pi_max = max(m.get_load() for m in self.machines)

        start_time = clock()
        for i in range(self.max_it):
            if clock() - start_time > self.timeout:
                print("Scheduling halted due to timeout")
                return False
            print("\nIteration " + str(i) + "/" + str(self.max_it))

            s, cmax_real = self._approx(pi_max)
            if len(s) > 0:
                for o in self.operations:
                    o.start_at(s[o])

            if cmax_real <= self.cmax:
                print("Given CMax:\t" + str(round(self.cmax, 2)) + "\nComputed Cmax:\t" + str(
                    round(cmax_real, 2)) + "\nSuccess!")
                return True
            else:
                print("Given CMax:\t" + str(round(self.cmax, 2)) + "\nComputed Cmax:\t" + str(
                    round(cmax_real, 2)) + "\nNeed to repeat.")
        return False

    def _approx(self, pi_max):
        operations = self.operations
        jobs = self.jobs
        for o in self.operations:
            o.apply_transformation(lambda x: int(math.pow(2, math.ceil(math.log2(x))) if x > 0 else 0))

        operations_start = {}
        for j in jobs:
            operations_start[j] = int(round(pi_max * r.random()))
        operations_start_min = min(operations_start.values())
        for j in jobs:
            o = j.graph.in_node
            o.start_at(operations_start[j] - operations_start_min)

        print("\tPhase 1: Choosing Paths")
        self._choose_path()

        print("\tPhase 2: Recursive Spread")
        p_trans_max = max(o.get_proc_time() for o in operations)
        L = max(j.graph.out_node.get_start() + j.graph.out_node.get_proc_time() for j in jobs) + 1

        operations = [o for j in jobs for o in j.get_path()]
        blocks = create_blocks(operations, L, p_trans_max, L)
        for b in blocks:
            blocks[b] = recursive_spread(blocks[b], p_trans_max)
        order = [e for b in blocks for e in blocks[b]]

        print("\tPhase 3: Postprocessing")
        # redo transformation of processing times
        for o in operations:
            o.apply_transformation(lambda x: x)

        return self._postprocess_alt(order)

    def _postprocess_alt(self, blocks: "List[Operation]"):
        operations = self.operations
        machines = self.machines
        jobs = self.jobs
        bigM = sum(o.get_proc_time() for o in operations)

        mod = Model("postprocess")
        start_time_model = clock()
        y = {}
        index_o_in_blocks = {}
        for pos, o in enumerate(blocks):
            index_o_in_blocks[o] = pos

        o_on_m = {}
        for m in machines:
            o_on_m[m] = list(sorted(m.get_assigned_operations(), key=lambda o: index_o_in_blocks[o]))

        S = mod.addVars(operations, vtype=GRB.CONTINUOUS, name="S")
        c_max = mod.addVar(vtype=GRB.CONTINUOUS, name="CMax")

        mod.setObjective(c_max)

        # Determine max finish time of all jobs
        mod.addConstrs(c_max >= S[o] + o.get_proc_time() for o in operations)

        # Precedence constraints
        mod.addConstrs((S[o1] + o1.get_proc_time() <= S[o1.get_chosen_succ()] for j in jobs for o1 in j.iter_operations() if not o1 == j.graph.out_node), name="Chose_path")

        # No two jobs can run on a machine at the same time
        for m in machines:
            for i in range(len(m.get_assigned_operations()) - 1):
                o1 = o_on_m[m][i]
                o2 = o_on_m[m][i + 1]
                mod.addConstr(S[o1] + o1.get_proc_time() <= S[o2], name="From_blocks[" + str(o1) + ", " + str(o2) + "]")

        mod.addConstr(c_max <= self.cmax)

        print("Model creation took %s"%(round(clock() - start_time_model, 1)))
        mod.setParam("LogFile", "")
        mod.setParam("LogToConsole", 0)
        # One solution is enough. Must not be optimal.
        mod.setParam("SolutionLimit", 1)
        mod.setParam("TimeLimit", 3600)
        mod.optimize()

        if mod.getAttr("Status") == GRB.INFEASIBLE:
            return {}, 2*self.cmax
        else:
            S_new = {o: S[o].x for o in operations}
            return S_new, c_max.x


def create_blocks(operations, L, p_max, modulo):
    blocks = {b: [] for b in range(math.ceil(L/float(p_max)))}
    for o in operations:
        index = math.floor(math.fmod(o.get_start(), modulo)/float(p_max))
        blocks[index].append(o)
    return blocks


# Recursive spread of jobs as described in Shmoys et al
def recursive_spread(block, p_max):
    if len(block) == 0 or p_max == 0:
        return block
    final_block = []
    next_block = []
    for o in block:
        if o.get_proc_time() == p_max:
            final_block.append(o)
        else:
            next_block.append(o)

    next_block = create_blocks(next_block, p_max, p_max/2, p_max)

    final_block = recursive_spread(next_block[0], int(p_max/2)) + recursive_spread(next_block[1], int(p_max/2)) + final_block
    # Sanity check
    for i in range(len(final_block)):
        if final_block[len(final_block) - 1 - i].get_chosen_succ() in final_block[:len(final_block) - 1 - i]:
            print("Error")

    return final_block


def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return iter(shuffled)


# Test instance from original paper [Shmoys et. al.]
if __name__ == "__main__":
    op1 = Operation("1", 2, [])
    op1.start_at(0)
    op2 = Operation("2", 4, [])
    op2.start_at(0)
    op3 = Operation("3", 1, [])
    op3.start_at(0)
    op4 = Operation("4", 2, [])
    op4.start_at(1)
    op5 = Operation("5", 1, [])
    op5.start_at(3)
    op6 = Operation("6", 4, [])
    op6.start_at(2)

    block = [op1, op2, op3, op4, op5, op6]

    sol1 = recursive_spread(block, 4)
    print(sol1)
    assert sol1 == [op3, op1, op4, op5, op2, op6]

    op7 = Operation("7", 2, [])
    op7.start_at(4)
    op8 = Operation("8", 4, [])
    op8.start_at(4)
    op9 = Operation("9", 1, [])
    op9.start_at(6)
    op10 = Operation("10", 2, [])
    op10.start_at(6)
    op11 = Operation("11", 1, [])
    op11.start_at(7)

    block = [op7, op8, op9, op10, op11]

    sol2 = recursive_spread(block, 4)
    print(sol2)
    assert sol2 == [op7, op9, op11, op10, op8]

