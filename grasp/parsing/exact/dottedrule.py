"""
This module implements a dotted rule which is common to several logic programs.
A dotted rule is an immutable object and its instances are managed by the class DottedRule.

:Authors: - Wilker Aziz
"""

from weakref import WeakValueDictionary
from grasp.cfg.symbol import Terminal, Nonterminal, make_span
from grasp.cfg.rule import CFGProduction
from functools import reduce

from collections import defaultdict


class DottedRule(object):
    """
    This implements a dotted rule common to several deductive logics used in parsing and intersection.
    It consists of:
        1) a CFG rule
        2) a right-most dot representing a state
        3) a vector of inner dots representing the states a long a path
        4) a weight
    Moreover, dotted rules are immutable objects.
    
    This class implements instance management.

    >>> item = DottedRule(CFGProduction(Nonterminal('S'), [Nonterminal('X'), Terminal('a')], 1.0), 0)
    >>> item2 = DottedRule(CFGProduction(Nonterminal('S'), [Nonterminal('X'), Terminal('a')], 1.0), 0)
    >>> item is item2
    True
    >>> item == item2
    True
    >>> item.dot
    0
    >>> item3 = item.advance(1)
    >>> item3 is not item
    True
    >>> item3 != item
    True
    >>> str(item)
    "[S] ||| [X] 'a' ||| 1.0 ||| () ||| 0"
    >>> str(item3)
    "[S] ||| [X] 'a' ||| 1.0 ||| (0,) ||| 1"
    """

    _instances = WeakValueDictionary()
    #_instances = defaultdict(None)

    def __new__(cls, rule, dot, inner=[], weight=1):
        """
        An item with a dot at the beginning of the rule. A rule must be a hashable object.

        :param rule: a CFGProduction
        :param dot: an integer representing the last state intersected
        :param inner: a sequence of states previously intersected
        :param weight: weighted incorporated thus far
        """

        skeleton = (rule, dot, tuple(inner), weight)
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
    def weight(self):
        """weight incorporated thus far"""
        return self._skeleton[3]

    @property
    def start(self):
        """the left-most state intersected"""
        return self._start

    @property
    def next(self):
        """the symbol to the right of the dot (or None if the dot has reached the end of the rule)"""
        return self._next

    @property
    def next_i(self):
        return len(self._skeleton[2])

    def nextsymbols(self):
        """iterates through the symbols ahead of the dot"""
        return self.rule.rhs[len(self.inner):]

    def is_complete(self):
        """whether the dot has reached the end of the rule"""
        return self._complete  # len(self.inner) == len(self.rule.rhs)

    def advance(self, dot, weight=1.0):
        """returns a new item whose dot has been advanced"""
        return DottedRule(self.rule, dot, self.inner + (self.dot,), weight)

    def _weight(self, wfsa, semiring):
        """
        Computes the total weight (in a given semiring) taking into account the underlying rule's weight
        and the path intersected with the automaton.
        Note that only terminal symbols in the RHS of the underlying rule can contribute to this computation.
        """
        fsa_states = self.inner + (self.dot,)
        fsa_weights = [wfsa.arc_weight(fsa_states[i], fsa_states[i + 1], sym) 
                for i, sym in filter(lambda i_s: isinstance(i_s[1], Terminal), enumerate(self.rule.rhs))]
        return reduce(semiring.times, fsa_weights, self.rule.weight)

    def cfg_production(self, wfsa, semiring):
        """
        Create a CFGProduction object from the item's state. Note that only complete items can perform such operation.
        Intersected symbols are renamed according to the given policy `make_symbol` which should expect as input a tuple
        of the kind (symbol, start, end) and return a symbol.
        """
        if not self.is_complete():
            raise ValueError('An incomplete item cannot be converted to a CFG production: %s' % str(self))
        fsa_states = self.inner + (self.dot,)
        fsa_weights = [wfsa.arc_weight(fsa_states[i], fsa_states[i + 1], sym) 
                for i, sym in filter(lambda i_s: isinstance(i_s[1], Terminal), enumerate(self.rule.rhs))]
        weight = reduce(semiring.times, fsa_weights, self.rule.weight)
        return CFGProduction(make_span(self.rule.lhs, self.start, self.dot),
                (make_span(sym, fsa_states[i], fsa_states[i + 1]) for i, sym in enumerate(self.rule.rhs)),
                weight)
    
    def __str__(self):
        return '%s ||| %s ||| %d ||| %f' % (str(self.rule), self.inner, self.dot, self.weight)
