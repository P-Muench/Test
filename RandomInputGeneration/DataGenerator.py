import random as r
import numpy as np
import math
from Datastructures.RandomGraph import RandomGraph
from Datastructures.Operation import Operation
from Datastructures.Ability import Ability
from Datastructures.Job import Job
from typing import List


class DataGenerator:

    def __init__(self, seed, num_abilities, num_jobs, max_num_operations, cost_min, cost_max):
        self.cost_max = cost_max
        self.cost_min = cost_min
        self.max_num_operations = max_num_operations
        self.num_jobs = num_jobs
        self.num_abilities = num_abilities
        self.seed = seed
        self.high_multiplicity = 1

        self._abilities: List[Ability] = []
        self._jobs: List[Job] = []
        r.seed(seed)
        np.random.seed(seed)

        self.serial_prob = .8

    def set_prob_serialization(self, prob):
        self.serial_prob = prob

    def generate(self, seed: int = None):
        """
        Generates random instance data and returns them

        :rtype: (List[Job], List[Ability])
        """
        self._abilities: List[Ability] = []
        self._jobs: List[Job] = []

        if seed is not None:
            r.seed(seed)
            np.random.seed(seed)

        dummy_ability = Ability("Dummy", self.cost_min/100.0, 1)
        Ability._dummy = dummy_ability

        self.create_abilities()
        self.create_jobs()
        self._abilities.append(dummy_ability)
        for j in self._jobs:
            j.set_high_multiplicity(self.high_multiplicity)

        return self._jobs, self._abilities

    def create_graph(self, mu: int, seed: float) -> RandomGraph:
        """
        Creates a randomly connected precedence graph with mu operations. Random connections determined by seed value.

        :return: Connected precedence graph
        :rtype: Graph
        :param mu: Number of operations
        :param seed: Random seed
        """
        # r.seed(seed)
        k = len(self._abilities)

        graph_collection = []

        for i in range(mu):
            while True:
                if k > 1:
                    p = min(3.0/(k-1), .4)
                else:
                    p = 1
                num_abilities = np.random.binomial(k-1, p) + 1
                ab = choose_abilities(self._abilities, num_abilities)
                if sum(a.get_resource() for a in ab) <= 1:
                    break
            processing_time = round(r.expovariate(0.1)) + 1
            graph_collection.append(RandomGraph(Operation(str(i), processing_time, ab)))

        while len(graph_collection) > 1:
            i = r.randint(0, len(graph_collection) - 1)
            while True:
                j = r.randint(0, len(graph_collection) - 1)
                if j != i:
                    break

            if r.random() <= self.serial_prob:
                graph_collection[i].serialize(graph_collection[j])
            else:
                graph_collection[i].parallelize(graph_collection[j])
            del graph_collection[j]

        return graph_collection[0]

    def create_abilities(self):
        for i in range(self.num_abilities):
            res = np.random.beta(a=2, b=5)
            assert res <= 1
            cost = r.randint(self.cost_min, self.cost_max)
            self._abilities.append(Ability(str(i), cost, res))

    def create_jobs(self):
        for i in range(self.num_jobs):
            self._jobs.append(Job(str(i), self.create_graph(r.randint(2, self.max_num_operations), self.seed + 1)))

    @staticmethod
    def print_jobs(jobs, filename):
        latex_start = "\\documentclass[landscape]{article}\n\n\\usepackage{tikz}\n\\begin{document}\n\\pagestyle{empty}\n"
        latex_end = "\\end{document}"

        inner = "\n".join(j.print() for j in jobs)

        if not filename.endswith(".tex"):
            filename += ".tex"
        with open(filename, 'w+') as file:
            file.write(latex_start + inner + latex_end)

    def print_instance(self, filename):
        with open(filename, "w+") as f:
            f.write(str(len(self._jobs)) + "\r\n")
            f.write(str(len(self._abilities)) + "\r\n")
            for j in self._jobs:
                f.write("\t".join([j.name, str(len(j.graph.nodes)), str(j._high_multiplicity)]) + "\r\n")
            for a in self._abilities:
                f.write("\t".join([a.name, str(a.get_cost()), str(a.get_resource())]) + "\r\n")
            for j in self._jobs:
                for o in j.iter_operations():
                    f.write("\t".join([j.name, o.name, str(o.get_proc_time()), "\t".join(a.name for a in o.get_abilities())]) + "\r\n")
            for j in self._jobs:
                for (o1, o2) in j.graph.edges:
                    f.write(j.name + "\t" + o1.name + "\t" + o2.name + "\r\n")

            from Datastructures.AbilityConfigurations import AbilityConfigurations
            ac = AbilityConfigurations(self._abilities)
            for j in self._jobs:
                for o in j.iter_operations():
                    ac.add_subset(o.get_abilities())
            num_all_configs = len(list(ac.iter_configs()))
            f.write("infos:\t" + "\t".join([str(len(self._jobs)), str(sum(len(j.graph.nodes) for j in self._jobs)), str(len(self._abilities)), str(num_all_configs), str(j._high_multiplicity)]) + "\r\n")


def choose_abilities(iterable: list, k: int) -> list:
    """

    :param iterable: iterable
    :param k:
    :return:
    :rtype: list
    """
    ab = np.random.choice(iterable, size=k, replace=False)
    return ab.tolist()
