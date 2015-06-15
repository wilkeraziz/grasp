"""
:Authors: - Wilker Aziz
"""

from collections import defaultdict


class LinearModel(object):

    def __init__(self, weights):
        self._w = defaultdict(None, weights)

    def dot(self, features):
        if len(self._w) > len(features):
            x = features
            y = self._w
        else:
            x = self._w
            y = features
        dot = 0.0
        for k, v in x.items():
            try:
                dot += v * y[k]
            except KeyError:
                pass
        return dot


def cdec_basic():
    return LinearModel(dict(EgivenFCoherent=1.0, SampleCountF=1.0, CountEF=1.0, MaxLexFgivenE=1.0, MaxLexEgivenF=1.0, IsSingletonF=1.0, IsSingletonFE=1.0, Glue=1.0))