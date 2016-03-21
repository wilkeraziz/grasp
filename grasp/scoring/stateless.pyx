"""
Stateless extractors.

:Authors: - Wilker Aziz
"""

from grasp.scoring.frepr cimport FRepr, FValue, FMap, FVec
from grasp.ptypes cimport weight_t
from grasp.cfg.symbol cimport Symbol, Terminal, Nonterminal
from grasp.recipes import re_key_value


cdef class WordPenalty(Stateless):
    """
    This counts the number of terminal symbols in a rule.
    It returns a simple FValue.
    """

    def __init__(self, int uid, str name, weight_t penalty=1.0):
        super(WordPenalty, self).__init__(uid, name)
        self._penalty = penalty

    def __getstate__(self):
        return super(WordPenalty,self).__getstate__(), {'penalty': self._penalty}

    def __setstate__(self, state):
        superstate, d = state
        self._penalty = d['penalty']
        super(WordPenalty,self).__setstate__(superstate)

    def __repr__(self):
        return '{0}(uid={1}, name={2}, penalty={3})'.format(WordPenalty.__name__,
                                                            repr(self.id),
                                                            repr(self.name),
                                                            repr(self._penalty))

    cpdef tuple fnames(self, wkeys):
        return tuple([self._name])

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

    cpdef FRepr constant(self, weight_t value):
        return FValue(value)

    @classmethod
    def construct(cls, int uid, str name, str cfgstr):
        cdef weight_t penalty = 1.0
        cfgstr, value = re_key_value('penalty', cfgstr, optional=True)
        if value:
            penalty = float(value)
        return WordPenalty(uid, name, penalty)

    @staticmethod
    def help():
        help_msg = ["# Counts the number of terminal symbols in the derivation.",
                    "# The contribution of each word is a constant (1.0 by default)."
                    "# This is an example of how to construct the feature:",
                    "WordPenalty penalty=1.0"]
        return '\n'.join(help_msg)

    @staticmethod
    def example():
        return 'WordPenalty penalty=1.0'


cdef class ArityPenalty(Stateless):
    """
    This counts the number of rules of each arity.
    It returns an FMap if configured to deal with unbounded arity, otherwise, FVec.
    """

    def __init__(self, int uid, str name, weight_t penalty=1.0, int max_arity=2):
        super(ArityPenalty, self).__init__(uid, name)
        self._penalty = penalty
        self._max_arity = max_arity

    def __getstate__(self):
        return super(ArityPenalty,self).__getstate__(), {'penalty': self._penalty, 'max_arity': self._max_arity}

    def __setstate__(self, state):
        superstate, d = state
        self._penalty = d['penalty']
        self._max_arity = d['max_arity']
        super(ArityPenalty,self).__setstate__(superstate)

    def __repr__(self):
        return '{0}(uid={1}, name={2}, penalty={3}, max_arity={4})'.format(ArityPenalty.__name__,
                                                                           repr(self.id),
                                                                           repr(self.name),
                                                                           repr(self._penalty),
                                                                           repr(self._max_arity))

    cpdef tuple fnames(self, wkeys):
        if self._max_arity < 0:
            return tuple(sorted([k for k in wkeys if k.startswith('{0}_'.format(self.name))]))
        else:
            return tuple(['{0}_{1}'.format(self.name, i) for i in range(self._max_arity + 1)])

    cpdef FRepr weights(self, dict wmap):
        if self._max_arity < 0:
            return FMap({k: v for k, v in wmap.items() if k.startswith('{0}_'.format(self.name))})
        else:
            return FVec([wmap['{0}_{1}'.format(self.name, i)] for i in range(self._max_arity + 1)])

    cpdef FRepr featurize(self, edge):
        """
        :param rule:
        :returns: edge's arity
        """
        cdef int arity = 0
        cdef Symbol sym
        cdef list penalties
        if edge.fvalue('GoalRule') != 0:
            # GoalRules are added after each scoring pass,
            # as the number of scoring passes may vary, we better not score such "dummy" rules
            return self.constant(0.0)
        for sym in edge.rhs:
            if isinstance(sym, Nonterminal):
                arity += 1
        if self._max_arity < 0:
            return FMap({'{0}_{1}'.format(self.name, arity): self._penalty})
        else:
            penalties = [0.0] * (self._max_arity + 1)
            penalties[arity] = self._penalty
            return FVec(penalties)

    cpdef FRepr constant(self, weight_t value):
        if self._max_arity < 0:
            return FMap([])
        else:
            return FVec([value for i in range(self._max_arity + 1)])

    @classmethod
    def construct(cls, int uid, str name, str cfgstr):
        cdef weight_t penalty = 1.0
        cdef int max_arity = 2
        cfgstr, value = re_key_value('penalty', cfgstr, optional=True)
        if value:
            penalty = float(value)
        cfgstr, value = re_key_value('max-arity', cfgstr, optional=True)
        if value:
            max_arity = int(value)
        return ArityPenalty(uid, name, penalty, max_arity)

    @staticmethod
    def help():
        help_msg = ["# Counts how many rules of a certain length the derivation contains.",
                    "# The length, or arity, is measure in number of nonterminals on the RHS.",
                    "# The contribution of each rule-type is a constant (1.0 by default).",
                    "# If a maximum arity is provided (2 by default), we produce a fixed number of dense scores.",
                    "# A negative arity indicates no limit, which turns this into a producer of sparse features."]
        return '\n'.join(help_msg)

    @staticmethod
    def example():
        return 'Arity penalty=1.0 max-arity=2'