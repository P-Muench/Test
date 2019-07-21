from Datastructures.Job import Job
from Datastructures.Ability import Ability
from typing import List, Set, TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from Datastructures.Operation import Operation
    import Datastructures


class OperationCollection:
    # Collection of operations as in IP (5.4)
    _counter = 0

    def __init__(self):
        self._operations: "Set[Operation]" = set()
        self._resource = 0
        self._cost = 0
        self._abilities: "Set[Ability]" = set()
        self._id = OperationCollection._counter
        OperationCollection._counter += 1

    def add_operation(self, operation: "Operation"):
        self._operations.add(operation)
        self._update()

    def add_operations(self, operations: "Iterable[Operation]"):
        for o in operations:
            self._operations.add(o)
        self._update()

    def _update(self):
        self._abilities = set(a for o in self._operations for a in o.get_abilities())
        self._resource = sum(a.get_resource() for a in self._abilities)
        self._cost = sum(a.get_cost() for a in self._abilities)

    def get_resource(self):
        return self._resource

    def get_cost(self):
        return self._cost

    def get_abilities(self):
        return self._abilities

    def __contains__(self, operation: "Operation"):
        return operation in self._operations

    def __repr__(self):
        return str(self._id)

    def __iter__(self):
        for o in self._operations:
            yield o
