"""
:Authors: - Wilker Aziz
"""

from collections import defaultdict
from grasp.cfg.utils import smart_ropen
import re
from ast import literal_eval


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


def load_cdef_file(path):
    wmap = {}
    with smart_ropen(path) as fi:
        for line in fi.readlines():
            fields = line.split()
            if len(fields) != 2:
                continue
            wmap[fields[0]] = float(fields[1])
    return LinearModel(wmap)


def get_weights(wmap, prefix):
    """Return weights prefixed by a given string."""
    return defaultdict(None, filter(lambda pair: pair[0].startswith(prefix), wmap.items()))