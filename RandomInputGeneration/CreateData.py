import random as r
import numpy as np
import math
from Datastructures.Operation import *
from Datastructures.RandomGraph import *
from Datastructures.Ability import *
from Datastructures.Job import *


def choose_abilities(iterable, k):
    l = np.random.choice(iterable, size=k, replace=False)
    return l.tolist()


def create_graph(abilities, n, seed):
    r.seed(seed)
    np.random.seed(seed)
    k = len(abilities)
    prob_ser = 0.8

    graph_collection = []

    for i in range(n):
        while True:
            num_abilities = min(k, round(r.expovariate(3.0/k)))
            num_abilities = max(1, num_abilities)
            ab = choose_abilities(abilities, num_abilities)
            if sum(a.get_resource() for a in ab) <= 1:
                break
        graph_collection.append(RandomGraph(Operation(str(i), round(r.expovariate(0.3)), ab)))

    while len(graph_collection) > 1:
        i = r.randint(0, len(graph_collection) - 1)
        while True:
            j = r.randint(0, len(graph_collection) - 1)
            if j != i:
                break
        if j < i:
            temp = i
            i = j
            j = temp

        if r.random() <= prob_ser:
            graph_collection[i].serialize(graph_collection[j])
        else:
            graph_collection[i].parallelize(graph_collection[j])
        del graph_collection[j]

    return graph_collection[0]


def create_abilities(k, cmin, cmax, seed):
    abilities = []

    r.seed(seed)

    for i in range(k):
        abilities.append(Ability(str(i), r.randint(cmin, cmax), math.pow(r.random(), 2)))

    return abilities


def create_jobs(n, o, abilities, seed):
    jobs = []

    r.seed(seed)

    for i in range(n):
        jobs.append(Job(str(i), create_graph(abilities, r.randint(2, o), seed + i)))

    return jobs
