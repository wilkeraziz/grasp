"""
The ancestral sampler of Chappelier and Rajman (200).


    @incollection{Chappelier+2000:MCParsing,
        Author = {Chappelier, Jean-C{\'e}dric and Rajman, Martin},
        Booktitle = {Natural Language Processing --- NLP 2000},
        Editor = {Christodoulakis, DimitrisN.},
        Pages = {106-117},
        Publisher = {Springer Berlin Heidelberg},
        Series = {Lecture Notes in Computer Science},
        Title = {{Monte-Carlo} Sampling for {NP}-Hard Maximization Problems in the Framework of Weighted Parsing},
        Volume = {1835},
        Year = {2000}}


It samples from the inverted CDF by efficiently computing node and edge values in the sum-times semiring.

:Authors: - Wilker Aziz
"""

from grasp.semiring import SumTimes, Counting
from .value import derivation_value, compute_edge_values
from .value import robust_value_recursion as compute_values
from .sample import sample_k


class AncestralSampler(object):

    def __init__(self, forest, tsort, omega=lambda e: e.weight, generations=10, semiring=SumTimes):
        self._forest = forest
        self._tsort = tsort
        self._omega = omega
        self._generations = generations
        self._semiring = semiring

        self._node_values = compute_values(self._forest,
                                           self._tsort,
                                           semiring,
                                           omega=omega,
                                           infinity=generations)

        self._edge_values = compute_edge_values(self._forest,
                                                semiring,
                                                self._node_values,
                                                omega=omega,
                                                normalise=not semiring.IDEMPOTENT)

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
        return self._node_values

    @property
    def Z(self):
        """Return the partition function."""
        return self._node_values[self._root]

    def n_derivations(self):
        if self._counts is None:
            self._counts = compute_values(self._forest,
                                          self._tsort,
                                          semiring=Counting,
                                          omega=lambda e: Counting.convert(self._omega(e), self._semiring),
                                          infinity=self._generations)
        return self._counts[self._root]

    def sample(self, n):
        """Draw samples from the inverted CDF."""
        return sample_k(self._forest, self._root, self._semiring,
                        node_values=self._node_values,
                        edge_values=self._edge_values,
                        omega=self._omega,
                        n_samples=n)

    def prob(self, d):
        return self._semiring.as_real(derivation_value(d, self._semiring, self.Z, self._omega))