"""
:Authors: - Wilker Aziz
"""

import numpy as np
from .scorer import Stateless
from easyhg.grammar.symbol import Terminal


class RuleTable(object):

    CDEC_DEFAULT = 'EgivenFCoherent SampleCountF CountEF MaxLexFgivenE MaxLexEgivenF IsSingletonF IsSingletonFE'.split()

    def __init__(self, uid, name, weights, fnames):
        super(RuleTable, uid, name, weights)
        self._fnames = fnames

    def score(self, rule, fmap):
        fvalues = np.array([fmap.get(fname, 0.0) for fname in self._fnames])
        return fvalues.dot(self.weights)

class WordPenalty(Stateless):

    def __init__(self, uid, name, weights):
        super(WordPenalty, self).__init__(uid, name, weights)

    def score(self, rule):
        """
        :param word: a Terminal
        :param context: a state
        :returns: weight
        """
        return sum(1 for sym in filter(lambda s: isinstance(s, Terminal), rule.rhs)) * self.weights[0]