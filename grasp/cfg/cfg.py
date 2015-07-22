"""
This module contains class definitions for weighted context-free grammars and utilitary functions.

:Authors: - Wilker Aziz
"""

from collections import defaultdict
from itertools import chain
from .symbol import Terminal, Nonterminal
from .rule import CFGProduction


class Grammar:

    pass


class CFG(Grammar):
    """A context-free grammar.

    An object which acts much like a dictionary (and sometimes a set).

    >>> cfg = CFG([CFGProduction(Nonterminal('S'), [Terminal('BOS'), Nonterminal('X'), Terminal('EOS')], 1.0)])
    >>> # let's start testing length
    >>> cfg.n_nonterminals()
    2
    >>> cfg.n_terminals()
    2
    >>> len(cfg)
    1
    >>> # then checks
    >>> cfg.is_terminal(Terminal('a'))
    False
    >>> cfg.is_terminal(Terminal('BOS'))
    True
    >>> cfg.is_nonterminal(Nonterminal('Y'))
    False
    >>> cfg.is_nonterminal(Nonterminal('X'))
    True
    >>> cfg.can_rewrite(Nonterminal('X'))
    False
    >>> # then iterators
    >>> set(cfg.iterterminals()) == set([Terminal('BOS'), Terminal('EOS')])
    True
    >>> set(cfg.iternonterminals()) == set([Nonterminal('S'), Nonterminal('X')])
    True
    >>> set(iter(cfg)) == set([CFGProduction(Nonterminal('S'), [Terminal('BOS'), Nonterminal('X'), Terminal('EOS')], 1.0)])
    True
    >>> # then topsort
    >>> cfg.add(CFGProduction(Nonterminal('X'), [Terminal('a')], 1.0))
    True
    >>> topsort = (frozenset([Terminal('BOS'), Terminal('a'), Terminal('EOS')]), frozenset([Nonterminal('X')]), frozenset([Nonterminal('S')]))
    >>> cfg.topsort()
    >>> cfg.add(CFGProduction(Nonterminal('X'), [Nonterminal('Y')], 1.0))
    True
    >>> cfg.add(CFGProduction(Nonterminal('Y'), [Terminal('b')], 1.0))
    True
    >>> cfg.add(CFGProduction(Nonterminal('Y'), [Terminal('b')], 1.0))
    False
    >>> topsort = (frozenset([Terminal('a'), Terminal('b'), Terminal('EOS'), Terminal('Y'), Terminal('BOS')]), frozenset([Nonterminal('X'), Nonterminal('Y')]), frozenset([Nonterminal('S')]))
    >>> cfg.topsort()
    >>> cfg.add(CFGProduction(Nonterminal('Y'), [Nonterminal('X')], 1.0))
    True
    >>> cfg.topsort()
    """

    def __init__(self, rules=[]):
        """
        :params rules:
            an iterable over CFG productions
        """
        super()
        self._rules_by_lhs = defaultdict(set)
        self._nonterminals = set()
        self._sigma = set()  # terminals
        for rule in rules:
            self.add(rule)

    def __len__(self):
        """Count the total number of rules."""
        return sum(len(rules) for rules in self._rules_by_lhs.values())

    @property
    def lexicon(self):
        return self._sigma

    def n_symbols(self):
        return len(self._nonterminals) + len(self._sigma)

    def n_nonterminals(self):
        return len(self._nonterminals)

    def n_terminals(self):
        return len(self._sigma)
    
    def is_terminal(self, terminal):
        """Whether or not a symbol is a terminal of the grammar."""
        return terminal in self._sigma

    def is_nonterminal(self, nonterminal):
        """Whether or not a symbol is a nonterminal of the grammar."""
        return nonterminal in self._nonterminals

    def deadends(self):
        return self._nonterminals - self._rules_by_lhs.keys()
    
    def can_rewrite(self, lhs):
        """Whether a given nonterminal can be rewritten.
        
        This may differ from ``self.is_nonterminal(symbol)`` which returns whether a symbol belongs
        to the set of nonterminals of the grammar.
        """
        return lhs in self._rules_by_lhs
    
    def iterterminals(self):
        """Iterate through terminal symbols in no particular order."""
        return iter(self._sigma)
    
    def iternonterminals(self):
        """Iterate through nonterminal symbols in no particular order."""
        return iter(self._nonterminals)
    
    def itersymbols(self):
        """Iterate though all symbols of the grammar in no particular order (except that terminals come first)."""
        return chain(self._sigma, self._nonterminals)
    
    def __iter__(self):
        """Iterate through rules in no particular order."""
        return chain(*iter(self._rules_by_lhs.values()))
    
    def iterrules(self, lhs):
        """Iterate through rules rewriting a given LHS symbol."""
        return chain(*iter(self._rules_by_lhs.values())) if lhs is None else iter(self._rules_by_lhs.get(lhs, frozenset()))

    def iteritems(self):
        """Iterate through pairs ``(lhs, rules)`` in no particular order."""
        return iter(self._rules_by_lhs.items()) 

    def __getitem__(self, lhs):
        """Return the set of rules (or an empty set) rewriting the given LHS symbol."""
        return self._rules_by_lhs.get(lhs, frozenset())
    
    def get(self, lhs, default=None):
        """Return the set of rules (or the default argument) rewritting the given LHS symbol."""
        return self._rules_by_lhs.get(lhs, default)
    
    def add(self, rule):
        """Add a rule if not yet in the grammar and return the result of the operation.
        
        This method invalidates partial ordering information.
        """

        group = self._rules_by_lhs[rule.lhs]
        n = len(group)
        group.add(rule)
        if len(group) > n:
            self._nonterminals.add(rule.lhs)
            self._nonterminals.update(filter(lambda s: isinstance(s, Nonterminal), rule.rhs))
            self._sigma.update(filter(lambda s: isinstance(s, Terminal), rule.rhs))
            return True
        else:
            return False

    def update(self, rules):
        """Adds multiple rules"""
        [self.add(r) for r in rules]

    def __str__(self):
        """String representation of the (top-sorted) CFG."""
        lines = []
        for lhs, rules in sorted(iter(self._rules_by_lhs.items()), key=lambda pair: str(pair[0])):
            for rule in sorted(rules, key=lambda r: str(r)):
                lines.append(str(rule))
        return '\n'.join(lines)

    def pprint(self, make_symbol):
        """String representation of the (top-sorted) CFG."""
        lines = []
        for lhs, rules in sorted(iter(self._rules_by_lhs.items()), key=lambda pair: str(pair[0])):
            for rule in sorted(rules, key=lambda r: str(r)):
                lines.append(rule.pprint(make_symbol))
        return '\n'.join(lines)


def stars(cfg):
    """Compute the backward-star and the forward-star of the forest.

    Suppose a set of nodes V and a set of edges E

    backward-star
        BS(v) = {e in E: head(e) == v}

    forward-star
        FS(v) = {(e, i): e in E and tail(e)[i] == v}
        
    :returns:
        a pair of dictionaries representing BS and FS, respectively.
    """

    backward = defaultdict(set)
    forward = defaultdict(set)
    for r in cfg:
        backward[r.lhs].add(r)
        [forward[sym].add((r, i)) for i, sym in enumerate(frozenset(r.rhs))]
    return backward, forward