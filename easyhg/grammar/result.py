"""
:Authors: - Wilker Aziz
"""

from functools import reduce


class Result(object):

    def __init__(self, triplets=[], Z=None):
        self._Z = Z
        self._triplets = list(triplets)

    def __len__(self):
        return len(self._triplets)

    def append(self, d, n, score):
        self._triplets.append((d, n, score))

    @property
    def Z(self):
        return self._Z

    def estimate(self, op):
        return reduce(op, (s for d, n, s in self._triplets))

    def count(self):
        return sum(n for d, n, s in self._triplets)

    def __iter__(self):
        return iter(self._triplets)

    def __getitem__(self, item):
        return self._triplets[item]