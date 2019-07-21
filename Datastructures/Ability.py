class Ability:
    _dummy = None

    def __init__(self, name: str, cost: float, resource: float):
        """
        Initializes an ability by a name, its cost and a resource requirement. Resource req. must be positive and <= 1

        :param name:
        :param cost:
        :param resource:
        """
        assert 0 <= resource <= 1, ""
        self.name = name
        self._c = cost
        self._r = resource

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_cost(self) -> float:
        """
        Returns cost of ability
        :return:
        """
        return self._c

    def get_resource(self) -> float:
        """
        Returns resource requirement. Is >= 0 and <= 1.
        :return:
        """
        return self._r
