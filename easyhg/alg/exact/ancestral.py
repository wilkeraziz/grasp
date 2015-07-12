"""
:Authors: - Wilker Aziz
"""

from .inference import robust_inside, normalised_edge_inside, sample
from easyhg.grammar.semiring import SumTimes, Count


class AncestralSampler(object):

    def __init__(self, forest, tsort, omega=lambda e: e.weight, generations=10, semiring=SumTimes):
        self._forest = forest
        self._tsort = tsort
        self._omega = omega
        self._generations = generations
        self._semiring = semiring

        self._inside_nodes = robust_inside(self._forest,
                                           self._tsort,
                                           semiring,
                                           omega=omega,
                                           infinity=generations)
        self._inside_edges = normalised_edge_inside(self._forest,
                                                    self._inside_nodes,
                                                    semiring,
                                                    omega=omega)
        self._root = self._tsort.root()
        self._counts = None

    @property
    def forest(self):
        return self._forest

    @property
    def tsort(self):
        return self._tsort

    @property
    def inside(self):
        return self._inside_nodes

    @property
    def Z(self):
        """Return the partition function."""
        return self._inside_nodes[self._root]

    def n_derivations(self):
        if self._counts is None:
            self._counts = robust_inside(self._forest,
                                         self._tsort,
                                         semiring=Count,
                                         infinity=self._generations,
                                         omega=lambda e: Count.zero if e.weight == self._semiring.zero else Count.one)
        return self._counts[self._root]

    def sample(self, n):
        """Draw samples from the inverted CDF."""
        return sample(self._forest, self._root, self._semiring,
                           Iv=self._inside_nodes,
                           Ie=self._inside_edges,
                           omega=self._omega,
                           N=n)