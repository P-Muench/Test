from Datastructures.Ability import *
from typing import List, Set, Union, Tuple, Iterable


class AbilityConfigurations:

    def __init__(self, abilities: List[Ability]):
        """
        Initializes ground set of abilities. To reduce looping complexity, also assign subsets that are required by operations.
        Example: If there are 2 ability requirements (a, b) and (a, c), abilityconfiguration (a), (b) and (c) are useless.

        :param abilities:
        """
        self._groundsets = set()
        self._abdict = {ab: i for i, ab in enumerate(abilities)}

    def add_subset(self, s: Iterable[Ability]):
        """
        To reduce looping complexity, also assign subsets that are required by operations.
        Example: If there are 2 ability requirements (a, b) and (a, c), abilityconfiguration (a), (b) and (c) are useless.

        :param s: Subset of abilities.
        """
        sorted_s = sorted(s, key=self._abdict.__getitem__)
        index_tuple = tuple(ab for ab in sorted_s)
        self._groundsets.add(index_tuple)

    def __repr__(self):
        s = "Contained set:\n"
        for g in self._groundsets:
            s += ("[" + ",".join(str(i) for i in g) + "]\n")
        return s

    def iter_configs(self) -> Set[Tuple[Ability]]:
        """
        Returns all possible and realistic ability configurations.

        :return:
        """
        allconfigs = [set(g) for g in self._groundsets]
        masterset = set()
        self._all_combs(set(), allconfigs, masterset)
        return masterset

    def _all_combs(self, conf: Set[Ability], others: List[Ability], masterset: Set[Tuple[Ability]]) -> None:
        """
        Helper function to recursively enumerate all configurations. New configurations are added to parameter masterset.

        :rtype: None
        :param conf: Current conf, all combinations of conf with elements of others are examined.
        :param others: List of abilites that have not been considered for conf before
        :param masterset: Here the configurations are added to
        """
        if len(others) > 0:
            for i, o in enumerate(others):
                if len(conf) == 0:
                    self._all_combs(o, others[i + 1:], masterset)
                else:
                    if AbilityConfigurations.get_resources(conf.union(o)) <= 1 and not (conf.issuperset(o) or conf.issubset(o)):
                        if not tuple(sorted(conf.union(o), key=self._abdict.__getitem__)) in masterset:
                            self._all_combs(conf.union(o), others[i+1:], masterset)
        if len(conf) > 0:
            masterset.add(tuple(sorted(conf, key=self._abdict.__getitem__)))

    @staticmethod
    def get_resources(abilities: Union[Set[Ability], List[Ability]]) -> float:
        """
        Get resources of a collection of abilities.

        :param abilities:
        :return: Resource cost
        """
        return sum(a.get_resource() for a in abilities)

    @staticmethod
    def get_cost(abilities: Iterable[Ability]) -> float:
        """
        Get cost of a collection of abilities.

        :param abilities:
        :return:
        """
        return sum(a.get_cost() for a in abilities)

    def get_groundsets(self) -> Set[Tuple[Ability]]:
        return self._groundsets.copy()


# Testing
if __name__ == '__main__':
    a = Ability("a", 1, 0.5)
    b = Ability("b", 2, 0.2)
    c = Ability("c", 1, 0.4)
    d = Ability("d", 1, 0.1)

    abilities_list = [a, b, c, d]
    ac = AbilityConfigurations(abilities_list)

    ac.add_subset([a, b])
    ac.add_subset([a, c])
    ac.add_subset([a, c])
    ac.add_subset([c, a])
    ac.add_subset([a])
    ac.add_subset([d])

    print(str(ac))

    configs = ac.iter_configs()
    print(configs)
