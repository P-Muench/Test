from typing import List, Set, Union, Iterable
from Datastructures.Ability import Ability
from Datastructures.Operation import Operation


class Machine:
    def __init__(self, name: str) -> None:
        self.name = str(name)
        self._assigned_operations: List[Operation] = []
        self._assigned_abilities: Set[Ability] = set()

    def __str__(self) -> str:
        """
        Returns machine's name and its assigned abilities.
        :return:e
        """
        return str("Machine " + self.name + " | Abilities " + " ".join(str(a.name) for a in self._assigned_abilities))

    def __repr__(self):
        return str(self)

    def assign_ability(self, ab: Ability) -> None:
        """
        Assign an ability to the machine.

        :param ab:
        """
        self._assigned_abilities.add(ab)

    def set_assigned_abilities(self, iterable: Iterable[Ability]) -> None:
        """
        Assigns multiple abilities to the machine.

        :param iterable:
        """
        for i in iterable:
            self.assign_ability(i)

    def get_assigned_abilities(self) -> Set[Ability]:
        return self._assigned_abilities

    def get_assigned_operations(self) -> List[Operation]:
        return self._assigned_operations

    def assign_operation(self, o: Operation) -> None:
        self._assigned_operations.append(o)

    def remove_operation(self, o: Operation) -> None:
        self._assigned_operations.remove(o)

    def set_assigned_operations(self, iterable: Iterable[Operation]) -> None:
        for i in iterable:
            self.assign_operation(i)

    def get_load(self) -> int:
        """
        Get the machine load of the machine.

        :return:
        """
        return sum(o.get_proc_time() for o in self._assigned_operations)

    def clear_solution(self) -> None:
        """
        Removes solution values from the machine object -> No assigned machines or abilities.
        """
        self._assigned_operations = []
        self._assigned_abilities = set()

    def get_cost(self) -> float:
        """
        Get cost of a machine, induced by its installed abilities.

        :return:
        """
        return sum(a.get_cost() for a in self._assigned_abilities)
