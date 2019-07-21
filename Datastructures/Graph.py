import Datastructures
from Datastructures.Operation import Operation
from Datastructures.Ability import Ability
from typing import List, Tuple, Iterable


class Graph:

    def __init__(self) -> None:
        self.nodes: List[Operation] = []
        self.edges: List[Tuple[Operation, Operation]] = []
        self.in_node: Operation = None
        self.out_node: Operation = None

    def add_node(self, operation: Operation):
        self.nodes.append(operation)

    def add_edge(self, e: Tuple[Operation, Operation]):
        self.edges.append(e)
        o1 = e[0]
        o2 = e[1]
        if o2 not in o1.iter_succ():
            o1.add_succ(o2)
        if o1 not in o2.iter_pred():
            o2.add_pred(o1)

    def copy(self, copy_num):
        g_new = Graph()
        mapping = {}
        for o in self.nodes:
            o_copy = o.copy(copy_num)
            mapping[o] = o_copy
            g_new.add_node(o_copy)
        for e in self.edges:
            o1 = e[0]
            o2 = e[1]
            new_edge = (mapping[o1], mapping[o2])
            g_new.add_edge(new_edge)
        g_new.in_node = mapping[self.in_node]
        g_new.out_node = mapping[self.out_node]
        return g_new
