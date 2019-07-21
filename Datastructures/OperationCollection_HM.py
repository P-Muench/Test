from Datastructures.Job import Job
from Datastructures.Ability import Ability
from typing import List, Set, TYPE_CHECKING, Iterable, Dict
from Datastructures.OperationCollection import OperationCollection

if TYPE_CHECKING:
    from Datastructures.Operation import Operation
    import Datastructures


class OperationCollection_HM(OperationCollection):
    # Collection of operations as in IP (5.4), extended for high multiplicity as in column generation chapter
    _counter = 0

    def __init__(self):
        super().__init__()
        self._operations: Dict[Operation, int] = {}

    def add_operation(self, operation: "Operation", num=1):
        self._operations[operation] = num
        self._update()

    def add_operations(self, operations):
        self._operations = operations
        self._update()

    def __contains__(self, operation: "Operation"):
        return operation in self._operations.keys()

    def __iter__(self):
        for o in self._operations:
            yield o

    def get_num_of(self, operation):
        return self[operation]

    def __getitem__(self, operation: "Operation"):
        if operation not in self._operations:
            raise KeyError("Operation " + str(operation) + " not in operation collection " + str(self))
        else:
            return self._operations[operation]
