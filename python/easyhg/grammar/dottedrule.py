"""
This module implements a dotted rule which is common to several logic programs.
A dotted rule is an immutable object and its instances are managed by the class DottedRule.

@author wilkeraziz
"""

from weakref import WeakValueDictionary
from symbol import Terminal
from rule import CFGProduction
from itertools import ifilter
from collections import defaultdict


class DottedRule(object):
    """
    This implements a dotted rule common to several deductive logics used in parsing and intersection.
    It consists of:
        1) a CFG rule
        2) a right-most dot representing a state
        3) a vector of inner dots representing the states a long a path
    Moreover, dotted rules are immutable objects.
    
    This class implements instance management.
    """

    _instances = WeakValueDictionary()
    #_instances = defaultdict(None)

    def __new__(cls, rule, dot, inner=[]):
        """The symbols in lhs and rhs must be hashable"""
        skeleton = (rule, dot, tuple(inner))
        obj = DottedRule._instances.get(skeleton, None)
        if not obj:
            obj = object.__new__(cls)
            DottedRule._instances[skeleton] = obj
            obj._skeleton = skeleton
            obj._start = inner[0] if inner else dot
            obj._next = rule.rhs[len(inner)] if len(inner) < len(rule.rhs) else None
            obj._complete = len(inner) == len(rule.rhs)
        return obj
    
    @property
    def rule(self):
        """returns the underlying CFG rule"""
        return self._skeleton[0]

    @property
    def dot(self):
        """the right-most state intersected"""
        return self._skeleton[1]

    @property
    def inner(self):
        """the inner states that have been intersected"""
        return self._skeleton[2]

    @property
    def start(self):
        """the left-most state intersected"""
        return self._start

    @property
    def next(self):
        """the symbol to the right of the dot (or None if the dot has reached the end of the rule)"""
        return self._next

    def nextsymbols(self):
        """iterates through the symbols ahead of the dot"""
        return self.rule.rhs[len(self.inner):]

    def is_complete(self):
        """whether the dot has reached the end of the rule"""
        return self._complete  # len(self.inner) == len(self.rule.rhs)

    def advance(self, dot):
        """returns a new item whose dot has been advanced"""
        return DottedRule(self.rule, dot, self.inner + (self.dot,))

    def weight(self, wfsa, semiring):
        """
        Computes the total weight (in a given semiring) taking into account the underlying rule's weight
        and the path intersected with the automaton.
        Note that only terminal symbols in the RHS of the underlying rule can contribute to this computation.
        """
        fsa_states = self.inner + (self.dot,)
        fsa_weights = [wfsa.arc_weight(fsa_states[i], fsa_states[i + 1], sym) 
                for i, sym in ifilter(lambda (_, s): isinstance(s, Terminal), enumerate(self.rule.rhs))]
        return reduce(semiring.times, fsa_weights, self.rule.weight)

    def cfg_production(self, wfsa, semiring, make_symbol):
        """
        Create a CFGProduction object from the item's state. Note that only complete items can perform such operation.
        Intersected symbols are renamed according to the given policy `make_symbol` which should expect as input a tuple
        of the kind (symbol, start, end) and return a symbol.
        """
        if not self.is_complete():
            raise ValueError('An incomplete item cannot be converted to a CFG production: %s' % str(self))
        fsa_states = self.inner + (self.dot,)
        fsa_weights = [wfsa.arc_weight(fsa_states[i], fsa_states[i + 1], sym) 
                for i, sym in ifilter(lambda (_, s): isinstance(s, Terminal), enumerate(self.rule.rhs))]
        weight = reduce(semiring.times, fsa_weights, self.rule.weight)
        return CFGProduction(make_symbol(self.rule.lhs, self.start, self.dot), 
                (make_symbol(sym, fsa_states[i], fsa_states[i + 1]) for i, sym in enumerate(self.rule.rhs)),
                weight)
    
    def __str__(self):
        return '%s ||| %s ||| %d' % (str(self.rule), self.inner, self.dot)
