"""
Stateless extractors.

:Authors: - Wilker Aziz
"""

from grasp.scoring.extractor cimport FValue, FMap
from grasp.ptypes cimport weight_t
from grasp.cfg.symbol cimport Symbol, Terminal, Nonterminal


cdef class Stateless(Extractor):

    def __init__(self, int uid, str name):
        super(Stateless, self).__init__(uid, name)

    cpdef FRepr featurize(self, edge): pass


cdef class WordPenalty(Stateless):

    def __init__(self, int uid, str name, weight_t penalty=1.0):
        super(WordPenalty, self).__init__(uid, name)
        self._penalty = penalty

    def __repr__(self):
        return '{0}(uid={1}, name={2}, penalty={3})'.format(WordPenalty.__name__,
                                                            repr(self.id),
                                                            repr(self.name),
                                                            repr(self._penalty))

    cpdef FRepr weights(self, dict wmap):
        try:
            return FValue(wmap[self.name])
        except KeyError:
            raise KeyError('Missing weight for WordPenalty')

    cpdef FRepr featurize(self, edge):
        """
        :param word: a Terminal
        :param context: a state
        :returns: weight
        """
        cdef int n = 0
        cdef Symbol sym
        for sym in edge.rhs:
            if isinstance(sym, Terminal):
                n += 1
        return FValue(self._penalty * n)


cdef class ArityPenalty(Stateless):

    def __init__(self, int uid, str name, weight_t penalty=1.0):
        super(ArityPenalty, self).__init__(uid, name)
        self._penalty = penalty

    def __repr__(self):
        return '{0}(uid={1}, name={2}, penalty={3})'.format(ArityPenalty.__name__,
                                                            repr(self.id),
                                                            repr(self.name),
                                                            repr(self._penalty))

    cpdef FRepr weights(self, dict wmap):  # using a sparse representation
        return FMap({k: v for k, v in wmap.items() if k.startswith(self.name)})

    cpdef FRepr featurize(self, edge):
        """
        :param rule:
        :returns: edge's arity
        """
        cdef int arity = 0
        cdef Symbol sym
        if edge.fvalue('GoalRule') != 0:
            # GoalRules are added after each scoring pass,
            # as the number of scoring passes may vary, we better not score such "dummy" rules
            return FMap({})
        for sym in edge.rhs:
            if isinstance(sym, Nonterminal):
                arity += 1
        return FMap({'{0}_{1}'.format(self.name, arity): self._penalty})
