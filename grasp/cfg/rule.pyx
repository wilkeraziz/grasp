"""
This module contains class definitions for rules, such as a context-free production.

:Authors: - Wilker Aziz
"""

import logging
from grasp.cfg.symbol cimport Terminal
from grasp.ptypes cimport weight_t
from cpython.object cimport Py_EQ, Py_NE


cdef class Rule:
    """
    A Rule is a container for a LHS symbol, a sequence of RHS symbols
    and a set of named features.
    """

    property lhs:
        def __get__(self):
            raise NotImplementedError()

    property rhs:
        def __get__(self):
            raise NotImplementedError()

    cpdef weight_t fvalue(self, fname, weight_t default=0.0):
        raise NotImplementedError()



cdef class NewCFGProduction(Rule):
    """
    Implements a context-free production.
    """

    def __init__(self, Nonterminal lhs, rhs, fmap):
        self._lhs = lhs
        self._rhs = tuple(rhs)
        self._fmap = dict(fmap)
        self._hash = hash((self._lhs, self._rhs, tuple(self._fmap.items())))

    property lhs:
        def __get__(self):
            """Return the LHS symbol (a Nonterminal) aka the head."""
            return self._lhs

    property rhs:
        def __get__(self):
            """A tuple of symbols (terminals and nonterminals) representing the RHS aka the tail."""
            return self._rhs

    property fpairs:
        def __get__(self):
            return self._fmap.items()

    cpdef weight_t fvalue(self, fname, weight_t default=0.0):
        return self._fmap.get(fname, default)

    def __hash__(self):
        return self._hash

    def __richcmp__(NewCFGProduction x, NewCFGProduction y, int opt):
        cdef bint eq = type(x) == type(y) and x.lhs == y.lhs  and x.rhs == y.rhs and x.fpairs == y.fpairs
        if opt == Py_EQ:
            return eq
        elif opt == Py_NE:
            return not eq
        else:
            raise ValueError('Cannot compare rules with opt=%d' % opt)

    def __repr__(self):
        return '%s(%s, %s, %s)' % (NewCFGProduction.__name__,
                                   repr(self.lhs),
                                   repr(self.rhs),
                                   repr(self._fmap))

    def __str__(self):
        return '%r ||| %s ||| %s' % (self.lhs,
                                     ' '.join(repr(s) for s in self.rhs),
                                     ' '.join('{0}={1}'.format(k, v) for k, v in sorted(self.fpairs)))

    @classmethod
    def MakeStandardCFGProduction(cls, Nonterminal lhs, rhs, float weight, fname='Prob', transform=float):
        return NewCFGProduction(lhs, rhs, {fname: transform(weight)})


cdef class _CFGProduction(Rule):
    """
    Implements a context-free production.
    """

    def __init__(self, Nonterminal lhs, rhs, weight_t weight):
        self._lhs = lhs
        self._rhs = tuple(rhs)
        self._weight = weight
        self._hash = hash((self._lhs, self._rhs, self._weight))
    
    property lhs:
        def __get__(self):
            """Return the LHS symbol (a Nonterminal) aka the head."""
            return self._lhs

    property rhs:
        def __get__(self):
            """A tuple of symbols (terminals and nonterminals) representing the RHS aka the tail."""
            return self._rhs
    
    property weight:
        def __get__(self):
            return self._weight

    def __hash__(self):
        return self._hash

    def __richcmp__(x, y, opt):
        cdef bint eq = type(x) == type(y) and x.weight == y.weight and x.lhs == y.lhs  and x.rhs == y.rhs
        if opt == Py_EQ:
            return eq
        elif opt == Py_NE:
            return not eq
        else:
            raise ValueError('Cannot compare rules with opt=%d' % opt)

    def __repr__(self):
        return '%s(%s, %s, %s)' % (_CFGProduction.__name__, repr(self.lhs), repr(self.rhs), repr(self.weight))

    def __str__(self):
        return '%r ||| %s ||| %s' % (self.lhs, ' '.join(repr(s) for s in self.rhs), self.weight)

    #def pprint(self, make_symbol):
    #    return '%s ||| %s ||| %s' % (make_symbol(self.lhs), ' '.join(str(make_symbol(s)) for s in self.rhs), self.weight)


def get_oov_cfg_productions(oovs, unk_lhs, fname, fvalue):
    for word in oovs:
        r = NewCFGProduction(Nonterminal(unk_lhs), (Terminal(word),), {fname: fvalue})
        logging.debug('Passthrough rule for %s: %s', word, r)
        yield r