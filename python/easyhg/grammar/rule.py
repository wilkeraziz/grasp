"""
@author wilkeraziz
"""

from weakref import WeakValueDictionary


class CFGProduction(object):
    """
    Implements a context-free production. References to productions are managed by the CFGProduction class.
    We use WeakValueDictionary for builtin reference counting.
    The symbols in the production must all be immutable (thus hashable).

    >>> CFGProduction(1, [1,2,3], 0.0)  # integers are hashable
    CFGProduction(1, (1, 2, 3), 0.0)
    >>> CFGProduction(('S', 1, 3), [('X', 1, 2), ('X', 2, 3)], 0.0)  # tuples are hashable
    CFGProduction(('S', 1, 3), (('X', 1, 2), ('X', 2, 3)), 0.0)
    >>> from symbol import Terminal, Nonterminal
    >>> CFGProduction(Nonterminal('S'), [Terminal('<s>'), Nonterminal('X'), Terminal('</s>')], 0.0)  # Terminals and Nonterminals are also hashable
    CFGProduction(Nonterminal('S'), (Terminal('<s>'), Nonterminal('X'), Terminal('</s>')), 0.0)
    """

    _rules = WeakValueDictionary()

    def __new__(cls, lhs, rhs, weight=0.0):
        """The symbols in lhs and rhs must be hashable"""
        skeleton = (lhs, tuple(rhs), weight)
        obj = CFGProduction._rules.get(skeleton, None)
        if not obj:
            obj = object.__new__(cls)
            CFGProduction._rules[skeleton] = obj
            obj._skeleton = skeleton
        return obj
    
    @property
    def lhs(self):
        return self._skeleton[0]

    @property
    def rhs(self):
        return self._skeleton[1]
    
    @property
    def weight(self):
        return self._skeleton[2]

    def __repr__(self):
        return '%s(%s, %s, %s)' % (CFGProduction.__name__, repr(self.lhs), repr(self.rhs), repr(self.weight))

    def __str__(self):
        return '%s ||| %s ||| %s' % (self.lhs, ' '.join(str(s) for s in self.rhs), self.weight)


