import Datastructures
from typing import Callable, List, Set, Iterator, Union, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from Datastructures.Machine import Machine
    from Datastructures.Ability import Ability


class Operation:
    _counter: int = 0

    def __init__(self, name: str, processing_time: float, abilities: Iterable["Ability"]):
        """
        Initializes an Operation by a name. Its processing time and required abilities. Precedences must be assigned separately.

        :param name:
        :param processing_time:
        :param abilities:
        """
        self._multiplier = 1
        self._p = processing_time
        assert 0 <= sum(a.get_resource() for a in abilities) <= 1, "Ability requirement for operation " + str(name) + " invalid: " + ", ".join(str(a) for a in abilities)
        self._abilities: Set[Ability] = set(abilities)

        self._pred: List[Datastructures.Operation.Operation] = []
        self._succ: List[Datastructures.Operation.Operation] = []

        self.col = 1
        self.row = 1

        self._transformation: Callable[[float], float] = lambda x: x

        self.id = Operation._counter
        if name != "Dummy":
            self.name = str(name)
            self._is_dummy = False
        else:
            self.name = name + str(Operation._counter)
            self._is_dummy = True
        Operation._counter += 1

        self._machine: Datastructures.Machine.Machine = None
        self._chosen_succ: Datastructures.Operation.Operation = None
        self._start = 0

    def add_pred(self, node: "Datastructures.Operation.Operation") -> None:
        """
        Add a predecessor operation. Has to be completed before self can start.

        :param node: Predecessor
        """
        self._pred.append(node)

    def set_dummy(self, bool: bool):
        self._is_dummy = bool

    def add_succ(self, node: "Datastructures.Operation.Operation") -> None:
        """
        Add a successor operation. self has to be completed before successor can start

        :param node: Successor
        """
        self._succ.append(node)

    def iter_pred(self) -> Iterator["Datastructures.Operation.Operation"]:
        """
        Iterates over all predecessors in the precedence graph
        """
        for n in self._pred:
            yield n

    def iter_succ(self) -> Iterator["Datastructures.Operation.Operation"]:
        """
        Iterates over all successors in the precedence graph

        """
        for n in self._succ:
            yield n

    def assign_machine(self, m: "Datastructures.Machine.Machine"):
        """
        Saves the assigned machine to operation. The operation must be assigned to the machine separately.

        :param m: Machine to which operation is assigned.
        """
        self._machine = m

    def get_assigned_machine(self) -> "Datastructures.Machine.Machine":
        """
        Get the machine the operation has been assigned to

        :return: Assigned Machine
        """
        return self._machine

    def start_at(self, time: float) -> None:
        """
        Assign a start time to an operation.

        :param time: starting time.
        """
        self._start = time

    def get_start(self) -> float:
        """
        Get the assigned starting time

        :return: Starting time.
        """
        return self._start

    def choose_succ(self, node: "Datastructures.Operation.Operation"):
        """
        Assign an operation to be executed next.

        :param node: Next operation
        """
        self._chosen_succ = node

    def get_chosen_succ(self) -> "Datastructures.Operation.Operation":
        """
        Get the next node if a path has been chosen.

        :return: Next node to be executed.
        """
        return self._chosen_succ

    def clear_solution(self):
        """
        Removes solution values from Operation
        """
        self._chosen_succ = None
        self._start = 0
        self._machine = None

    def get_abilities(self) -> Set["Ability"]:
        """
        Get required abilities.

        :return: Set of abilities.
        """
        return self._abilities

    def get_proc_time(self) -> int:
        """
        Get processing time. Adjusted for transformations. See Operation._transformation.

        :rtype: int
        """
        return int(self._transformation(self._p))

    def is_dummy(self) -> bool:
        """
        Dummy nodes are inserted during random graph generation and might me ignored later on.

        :return: bool
        """
        return self._is_dummy

    def apply_transformation(self, transform_lambda: Callable[[float], float]):
        """
        Applies specified transformation function to operation's processing time.

        :type transform_lambda: Callable[[float], float]
        :param transform_lambda: Transformation function
        """
        self._transformation = transform_lambda

    def __str__(self) -> str:
        """
        Returns the Operations ID if it's not a Dummy-Operations, else "Dummy"+ID
        :rtype: str
        :return: Operation name
        """
        if self.is_dummy():
            return self.name
        else:
            return str(self.name)

    def __repr__(self) -> str:
        return str(self)

    def __lt__(self, other: "Datastructures.Operation.Operation"):
        """

        :rtype: bool
        """
        return self.id < other.id

    def set_high_multiplicity(self, high_multiplicity: int) -> None:
        """
        Sets a multiplier factor for the high multiplicity case.

        :param high_multiplicity:
        """
        self._multiplier = high_multiplicity

    def copy(self, copy_num):
        new_op = Operation(self.name + "_" + str(copy_num), self._p, self._abilities)
        new_op._is_dummy = self.is_dummy()
        new_op._transformation = self._transformation
        new_op._machine = self._machine
        new_op._start = self._start
        return new_op
