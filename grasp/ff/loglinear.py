"""
:Authors: - Wilker Aziz
"""

import numpy as np
from collections import defaultdict
from grasp.recipes import smart_ropen
from .extractor import Stateful, Stateless, TableLookup


class LogLinearModel(object):

    def __init__(self, wmap, extractors):
        self._wmap = defaultdict(None, wmap)
        # all scorers sorted by id
        self._extractors = tuple(sorted(extractors, key=lambda scorer: scorer.id))
        # lookup ones
        self._lookup = tuple(filter(lambda s: isinstance(s, TableLookup), self._extractors))
        # stateless ones
        self._stateless = tuple(filter(lambda s: isinstance(s, Stateless), self._extractors))
        # stateful ones
        self._stateful = tuple(filter(lambda s: isinstance(s, Stateful), self._extractors))

        # memorise the a weight representation for each extractor
        self._lookup_weights = tuple(extractor.weights(self._wmap) for extractor in self._lookup)
        self._stateless_weights = tuple(extractor.weights(self._wmap) for extractor in self._stateless)
        self._stateful_weights = tuple(extractor.weights(self._wmap) for extractor in self._stateful)

    @property
    def lookup(self):
        return self._lookup

    @property
    def stateless(self):
        return self._stateless

    @property
    def stateful(self):
        return self._stateful

    def lookup_score(self, freprs):
        return np.sum(self._lookup[i].dot(frepr, self._lookup_weights[i]) for i, frepr in enumerate(freprs))

    def stateless_score(self, freprs):
        return np.sum(self._stateless[i].dot(frepr, self._stateless_weights[i]) for i, frepr in enumerate(freprs))

    def stateful_score(self, freprs):
        """
        Return the score (a linear combination) associated with a certain vector representation.
        :param fvecs: the features of each scorer
        :param scorers: scorers which produced the features
        :return: dot product
        """
        return np.sum(self._stateful[i].dot(frepr, self._stateful_weights[i]) for i, frepr in enumerate(freprs))


