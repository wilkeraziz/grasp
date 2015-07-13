"""
:Authors: - Wilker Aziz
"""

from collections import defaultdict, Counter

from .extractor import Stateless
from easyhg.grammar.symbol import Terminal, Nonterminal


class WordPenalty(Stateless):
    def __init__(self, uid, name, penalty=1.0):
        super(WordPenalty, self).__init__(uid, name)
        self._penalty = penalty

    def __repr__(self):
        return '{0}(uid={1}, name={2}, penalty={3})'.format(WordPenalty.__name__,
                                                            repr(self.id),
                                                            repr(self.name),
                                                            repr(self._penalty))

    def weights(self, wmap):
        try:
            return wmap.get(self.name)
        except KeyError:
            raise KeyError('Missing weight for WordPenalty')

    def featurize(self, edge):
        """
        :param word: a Terminal
        :param context: a state
        :returns: weight
        """
        return sum(self._penalty for sym in filter(lambda s: isinstance(s, Terminal), edge.rhs))

    def dot(self, frepr, wrepr):
        """
        :param frepr: word penalty (a float)
        :param wrepr: weight (a float)
        :return: product
        """
        return frepr * wrepr


class ArityPenalty(Stateless):
    def __init__(self, uid, name, penalty=1.0):
        super(ArityPenalty, self).__init__(uid, name)
        self._penalty = penalty

    def __repr__(self):
        return '{0}(uid={1}, name={2}, penalty={3})'.format(ArityPenalty.__name__,
                                                            repr(self.id),
                                                            repr(self.name),
                                                            repr(self._penalty))

    def weights(self, wmap):  # using a sparse representation
        return defaultdict(None, ((k, v) for k, v in wmap.items() if k.startswith(self.name)))

    def featurize(self, edge):
        """
        :param rule:
        :returns: edge's arity
        """
        arity = sum(1 for _ in filter(lambda s: isinstance(s, Nonterminal), edge.rhs))
        return arity

    def dot(self, frepr, wrepr):
        """

        :param frepr: edge's arity (an integer)
        :param wrepr: weight map
        :return: penalty * wmap[arity]
        """
        arity = frepr
        return self._penalty * wrepr.get('{0}_{1}'.format(self.name, arity), 0)