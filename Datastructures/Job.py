import _heapq
from Datastructures.RandomGraph import Graph
from Datastructures.Operation import Operation
from typing import List, Iterator, Generator, Dict, Set
import networkx as nx
import random


class Job:

    def __init__(self, name: str, graph: Graph) -> None:
        """

        :type graph: Graph
        :type name: str
        :param name: Name of a Job
        :param graph: Job's precedence graph
        """
        self.name = name
        self.graph = graph

        self._high_multiplicity = 1

    def get_p_sum(self) -> float:
        """
        Returns total length (in processing time) of a job. Incorporates possible transofmrations of job processing times.

        :return: Job length.
        """
        return sum(o.get_proc_time() for o in self.iter_operations())

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @staticmethod
    def _merge_paths(l: List[List[Operation]]) -> List[Operation]:
        """
        Receives a list of paths and merges them which results in a feasible order of operations.

        :param l: A list of paths.
        :return: Feasible order of operations.
        """
        assert type(l) is list
        while len(l) > 2:
            l = [Job._merge_paths(l[:2])] + l[2:]
        if len(l) == 1:
            return l[0]
        l_result = []
        m = l[0]
        n = l[1]
        m_counter, n_counter = 0, 0
        while len(l_result) < len(set(m + n)):
            if m_counter == len(m):
                l_result.extend(n[n_counter:])
                continue
            if n_counter == len(n):
                l_result.extend(m[m_counter:])
                continue
            if m[m_counter] == n[n_counter]:
                l_result.append(m[m_counter])
                m_counter += 1
                n_counter += 1
                continue
            if m[m_counter] not in n[n_counter:]:
                l_result.append(m[m_counter])
                m_counter += 1
            else:
                l_result.append(n[n_counter])
                n_counter += 1
        return l_result

    def iter_operations(self) -> Iterator[Operation]:
        """
        Let's one iteratre over all the contained operations.
        """
        for o in self.graph.nodes:
            yield o

    def print(self):
        """
        Print tikz code that canbe embedded into a LaTeX-document for visualization. Only works if rows and columns are set appropriately (e.g. while random creation)

        :return:
        """
        start_str = "\\begin{tikzpicture}[shorten >=1pt,->]\n  \\tikzstyle{vertex}=[circle,fill=black!25,minimum size=17pt,inner sep=0pt]\n\\tikzstyle{dummyvertex}=[circle,fill=black!50,minimum size=17pt,inner sep=0pt]\n  \\foreach \\name/\\x/\\y in "
        dummy_str = "\n\\node[vertex] (G-\\name) at (\\x,\\y) {$\\name$};\n\\foreach \\name/\\x/\\y in "
        mid_str = "\n\\node[dummyvertex] (G-\\name) at (\\x,\\y) {$\\name$};\n  \\foreach \\from/\\to in "
        end_str = " \\draw (G-\\from) -- (G-\\to);\n\n\\end{tikzpicture}\n\\vspace{1cm}\n"

        "test".startswith("te")
        s = "{" + ", ".join(str(n.id) + "/" + str(n.col) + "/" + str(n.row) for n in self.graph.nodes if
                            not n.is_dummy()) + "}"
        s_dummy = "{" + ", ".join(str(n.id) + "/" + str(n.col) + "/" + str(n.row) for n in self.graph.nodes if
                                  n.is_dummy()) + "}"
        s2 = "{" + ", ".join(str(e[0].id) + "/" + str(e[1].id) for e in self.graph.edges) + "}"

        return start_str + s + dummy_str + s_dummy + mid_str + s2 + end_str

    def get_path(self) -> Iterator[Operation]:
        """
        If a graph has been chosen, then this let's one iterate over the path.
        """
        n = self.graph.in_node
        assert n.get_chosen_succ() is not None, "No path has been chosen"
        while n.get_chosen_succ() is not None:
            yield n
            n = n.get_chosen_succ()
        yield n

    def get_earliest_start(self, set_initial_path: bool = False, random_path: bool = False, extend_precedences=False) -> Dict[Operation, float]:
        """
        Computes earliest start times of each operation and returns them as a Dictionary.
        See Lemma 4.11

        :param random_path: If set to True, then the resulting paths will be set randomly
        :param set_initial_path: If this is true, then a feasible order of operation is stored and can be accessed via
        .get_path()
        :return: Dictionary of starting times.
        """
        S = {self.graph.in_node}
        Q = {n: set() for n in self.graph.nodes}
        path = []
        predecessors: Dict[Operation: Set[Operation]] = {n: set(n.iter_pred()) for n in self.graph.nodes}
        successors: Dict[Operation: Set[Operation]] = {n: set(n.iter_succ()) for n in self.graph.nodes}

        while len(S) > 0:
            if random_path:
                v = random.sample(S, 1)[0]
                S.remove(v)
            else:
                v = S.pop()
            path.append(v)

            for w in successors[v].copy():
                # delete edges
                successors[v].remove(w)
                predecessors[w].remove(v)
                Q[w] = Q[w].union(Q[v].union([v]))
                if len(predecessors[w]) == 0:
                    S.add(w)

        ES = {n: int(sum(o.get_proc_time() for o in Q[n])) for n in self.graph.nodes}

        if set_initial_path:
            s = self.graph.in_node.get_start()
            for i in range(len(path) - 1):
                path[i].choose_succ(path[i + 1])
                path[i].start_at(s)
                s += path[i].get_proc_time()
            path[-1].start_at(s)

        if extend_precedences:
            for o in self.graph.nodes:
                for o_pred in Q[o]:
                    if o_pred not in o.iter_pred() and o_pred != o:
                        o.add_pred(o_pred)
                    if o not in o_pred.iter_succ() and o_pred != o:
                        o_pred.add_succ(o)
        return ES

    def get_latest_start(self) -> Dict[Operation, float]:
        """
        Computes latest start times of each operation and returns them as a Dictionary.
        Old Version. Works similar to Lemma 4.11 but slower.

        :return: Dictionary of starting times.
        """
        LS = {n: -1 for n in self.graph.nodes}
        paths = {n: [] for n in self.graph.nodes}
        H = []

        LS[self.graph.out_node] = 0
        paths[self.graph.out_node].append(self.graph.out_node)

        for v2 in self.graph.out_node.iter_pred():
            LS[v2] = self.graph.out_node.get_proc_time()
            paths[v2] = paths[self.graph.out_node] + [v2]
            _heapq.heappush(H, (LS[v2] + v2.get_proc_time(), v2))

        while len(H) > 0:
            v = _heapq.heappop(H)[1]
            for v2 in v.iter_pred():
                for v3 in v2.iter_succ():
                    if LS[v3] < 0:
                        break
                else:
                    paths[v2] = Job._merge_paths([paths[v3].copy() for v3 in v2.iter_succ()])
                    path_len = sum(o.get_proc_time() for o in paths[v2])
                    LS[v2] = path_len
                    paths[v2].append(v2)
                    _heapq.heappush(H, (path_len + v2.get_proc_time(), v2))

        p_sum_t = sum(o.get_proc_time() for o in self.graph.nodes)
        for l in LS:
            LS[l] = int(self.graph.in_node.get_start() + p_sum_t - LS[l] - l.get_proc_time())
        return LS

    def set_high_multiplicity(self, high_multiplicity: int) -> None:
        """
        Sets multiplier value for the high multiplicity case. Stored in the operations.

        :rtype: None
        """
        if high_multiplicity >= 1:
            self._high_multiplicity = high_multiplicity
            for o in self.iter_operations():
                o.set_high_multiplicity(high_multiplicity)

    def to_low_multiplicity(self):
        if self._high_multiplicity > 1:
            new_jobs = []
            for i in range(self._high_multiplicity):
                j = Job(self.name + "_" + str(i), Graph.copy(self.graph, i))
                new_jobs.append(j)
            return new_jobs
        else:
            return [self]

    def draw(self):
        # rudimentary drawing option

        latex_start = "\\documentclass[landscape]{article}\n\n\\usepackage{tikz}\n\\begin{document}\n\\pagestyle{empty}\n"
        latex_end = "\\end{document}"

        G = nx.from_edgelist(self.graph.edges)
        pos = nx.kamada_kawai_layout(G)
        for o in self.iter_operations():
            o.row = round(5*pos[o][0], 2)
            o.col = round(5*pos[o][1], 2)
        inner = self.print()

        filename = "Output/new.tex"

        with open(filename, 'w+') as file:
            file.write(latex_start + inner + latex_end)


def randomly(seq):
    shuffled = list(seq)
    random.shuffle(shuffled)
    return shuffled
