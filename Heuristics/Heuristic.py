import sys
from typing import List, Set
from Datastructures.Operation import Operation
from Datastructures.Job import Job
from Datastructures.Ability import Ability


# Abstract class 'Heuristic'. Subclasses have to implement a solve method for LM and HM.
class Heuristic:

    def __init__(self, jobs: List[Job], abilities: List[Ability], cmax: float, timeout: float,
                 upper_bound: float = sys.maxsize) -> None:
        self.jobs = jobs
        self.abilities = abilities
        self.cmax = cmax
        self.timeout = timeout
        self.upper_bound = upper_bound
        self.verbose = False
        self.high_multiplicty = False
        self.multiplier = 1
        self.output_filename = ""

        self.operations: List[Operation] = [o for j in jobs for o in j.iter_operations()]
        self.n = sum(1 for o in self.operations if not o.is_dummy())

        assert cmax >= max(o.get_proc_time() for o in self.operations)

    def solve(self):
        if self.high_multiplicty:
            return self._solve_high_multiplicity()
        else:
            return self._solve_low_multiplicity()

    def _solve_high_multiplicity(self):
        """
        Abstract function that solves the RABMCP (High Multiplicity) via a heuristic to be implemented in subclasses.
        """
        raise NotImplementedError

    def _solve_low_multiplicity(self):
        """
        Abstract function that solves the RABMCP via a heuristic to be implemented in subclasses.
        """
        raise NotImplementedError

    def set_output_file(self, filename):
        self.output_filename = filename
