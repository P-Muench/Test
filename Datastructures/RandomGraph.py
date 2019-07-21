import Datastructures
from Datastructures.Operation import Operation
from Datastructures.Ability import Ability
from typing import List, Tuple
from Datastructures.Graph import Graph


class RandomGraph(Graph):
    # extension of Graph class that allows for serialization and parallelization as described in Def 6.2 + 6.3
    # keeps track of rows and columns of jobs such that they can be printed in latex code

    def __init__(self, node: Operation) -> None:
        super().__init__()
        self.nodes: List[Operation] = [node]
        self.edges: List[Tuple[Operation, Operation]] = []
        self.in_node: Operation = node
        self.out_node = node
        self.max_row = 1

    def serialize(self, other: "Datastructures.Graph.Graph") -> None:
        """
        Serializes self with other graph. Puts graph "other" at the end of self.

        :param other: Graph to be added at the end.
        """
        row_dif = other.out_node.row - self.out_node.row
        for n in other.nodes:
            n.col += self.out_node.col
            n.row -= row_dif
        self.nodes.extend(other.nodes)
        self.edges.extend(other.edges)
        self.edges.append((self.out_node, other.in_node))
        self.out_node.add_succ(other.in_node)
        other.in_node.add_pred(self.out_node)
        self.out_node = other.out_node

    def parallelize(self, other: "Datastructures.Graph.Graph") -> None:
        """
        Parallelizes self with other graph. Puts graph "other" in parallel to self. In order to do that, dummy nodes are
        added at start and end of graph. Dummy nodes require dummy ability and have processing time 0.

        :param other: Graph to be added at the end.
        """
        self.nodes.extend(other.nodes)
        self.edges.extend(other.edges)

        dummy_in = Operation("Dummy", 0, [Ability._dummy])
        dummy_out = Operation("Dummy", 0, [Ability._dummy])

        # adjust row and col
        for n in self.nodes:
            n.col += 1

        dummy_out.col = max(self.out_node.col, other.out_node.col) + 1
        dummy_in.row = self.max_row + 1
        dummy_out.row = self.max_row + 1
        for n in other.nodes:
            n.row += self.max_row + 1
        self.max_row += other.max_row + 1

        dummy_in.add_succ(self.in_node)
        self.in_node.add_pred(dummy_in)
        dummy_in.add_succ(other.in_node)
        other.in_node.add_pred(dummy_in)

        dummy_out.add_pred(self.out_node)
        self.out_node.add_succ(dummy_out)
        dummy_out.add_pred(other.out_node)
        other.out_node.add_succ(dummy_out)

        self.nodes.extend([dummy_in, dummy_out])

        self.edges.append((dummy_in, self.in_node))
        self.edges.append((dummy_in, other.in_node))
        self.edges.append((self.out_node, dummy_out))
        self.edges.append((other.out_node, dummy_out))

        self.in_node = dummy_in
        self.out_node = dummy_out

