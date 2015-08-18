"""
This module contains class definitions for rules, such as a context-free production.

:Authors: - Wilker Aziz
"""

import logging
from weakref import WeakValueDictionary
from grasp.cfg.symbol import Terminal, Nonterminal


cdef class Rule:

    property lhs:
        def __get__(self):
            raise NotImplementedError()

    property rhs:
        def __get__(self):
            raise NotImplementedError()

    property weight:
        def __get__(self):
            raise NotImplementedError()


class CFGProduction(Rule):
    """
    Implements a context-free production. 
    
    References to productions are managed by the CFGProduction class.
    We use WeakValueDictionary for builtin reference counting.
    The symbols in the production must all be immutable (thus hashable).

    >>> CFGProduction(1, [1,2,3], 0.0)  # integers are hashable
    CFGProduction(1, (1, 2, 3), 0.0)
    >>> CFGProduction(('S', 1, 3), [('X', 1, 2), ('X', 2, 3)], 0.0)  # tuples are hashable
    CFGProduction(('S', 1, 3), (('X', 1, 2), ('X', 2, 3)), 0.0)
    >>> CFGProduction(Nonterminal('S'), [Terminal('<s>'), Nonterminal('X'), Terminal('</s>')], 0.0)  # Terminals and Nonterminals are also hashable
    CFGProduction(Nonterminal('S'), (Terminal('<s>'), Nonterminal('X'), Terminal('</s>')), 0.0)
    """

    #_rules = WeakValueDictionary()
    #_rules = defaultdict(None)

    #def __new__(cls, lhs, rhs, weight):
    #    """The symbols in lhs and in the rhs must be hashable."""
    #    skeleton = (lhs, tuple(rhs), weight)
    #    obj = CFGProduction._rules.get(skeleton, None)
    #    if not obj:
    #        obj = object.__new__(cls)
    #        CFGProduction._rules[skeleton] = obj
    #        obj._skeleton = skeleton
    #    return obj

    def __init__(self, lhs, rhs, weight):
        self._skeleton = (lhs, tuple(rhs), weight)
    
    @property
    def lhs(self):
        """Return the LHS symbol (a Nonterminal) aka the head."""
        return self._skeleton[0]

    @property
    def rhs(self):
        """A tuple of symbols (terminals and nonterminals) representing the RHS aka the tail."""
        return self._skeleton[1]
    
    @property
    def weight(self):
        return self._skeleton[2]

    def __hash__(self):
        return hash(self._skeleton)

    def __eq__(self, other):
        return self._skeleton == other._skeleton

    def __repr__(self):
        return '%s(%s, %s, %s)' % (CFGProduction.__name__, repr(self.lhs), repr(self.rhs), repr(self.weight))

    def __str__(self):
        return '%r ||| %s ||| %s' % (self.lhs, ' '.join(repr(s) for s in self.rhs), self.weight)

    def pprint(self, make_symbol):
        return '%s ||| %s ||| %s' % (make_symbol(self.lhs), ' '.join(str(make_symbol(s)) for s in self.rhs), self.weight)


def get_oov_cfg_productions(oovs, unk_lhs, weight):
    for word in oovs:
        r = CFGProduction(Nonterminal(unk_lhs), [Terminal(word)], weight)
        logging.debug('Passthrough rule for %s: %s', word, r)
        yield r