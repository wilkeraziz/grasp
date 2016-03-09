"""
:Authors: - Wilker Aziz
"""

from collections import defaultdict
import re
from ast import literal_eval
from grasp.cfg.rule cimport Rule, NewCFGProduction
from grasp.ptypes cimport weight_t
import numpy as np


cdef class Model:

    pass


cdef class PCFG(Model):

    def __init__(self, str fname='LogProb', transform=None):
        self._fname = fname
        self._transform = transform

    def __call__(self, NewCFGProduction rule):
        return rule.fvalue(self._fname) if self._transform is None else self._transform(rule.fvalue(self._fname))


cdef class DummyConstant(Model):

    def __init__(self, weight_t value):
        self._value = value

    def __call__(self, Rule whatever):
        return self._value


class LinearModel(object):

    def __init__(self, weights):
        self._w = defaultdict(None, weights)

    def dot(self, fvpairs):
        dot = 0
        for f, v in fvpairs:
            try:
                dot += v * self._w[f]
            except KeyError:
                pass
        return dot

    def get(self, fname):
        return self._w.get(fname, 0.0)

    def __str__(self):
        return ' '.join('{0}={1}'.format(k, v) for k, v in sorted(self._w.items()))


def cdec_basic():
    return LinearModel(dict(EgivenFCoherent=1.0, SampleCountF=1.0, CountEF=1.0, MaxLexFgivenE=1.0, MaxLexEgivenF=1.0, IsSingletonF=1.0, IsSingletonFE=1.0, Glue=1.0))


def get_weights(wmap, prefix):
    """Return weights prefixed by a given string."""
    return defaultdict(None, filter(lambda pair: pair[0].startswith(prefix), wmap.items()))
