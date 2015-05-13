"""
@author wilkeraziz
"""

from collections import defaultdict
from itertools import chain, groupby, ifilter
from symbol import Terminal, Nonterminal


class CFG(object):

    def __init__(self, rules=[]):


        self._rules_by_lhs = defaultdict(set)
        self._nonterminals = set()
        self._sigma = set()  # terminals
        for rule in rules:
            self._rules_by_lhs[rule.lhs].add(rule)
            self._nonterminals.add(rule.lhs)
            self._nonterminals.update(ifilter(lambda s: isinstance(s, Nonterminal), rule.rhs))
            self._sigma.update(ifilter(lambda s: isinstance(s, Terminal), rule.rhs))

    @property
    def terminals(self):
        return self._sigma

    @property
    def nonterminals(self):
        return self._nonterminals

    def add(self, rule):
        self._rules_by_lhs[rule.lhs].add(rule)

    def __contains__(self, lhs):
        """Tests whether a given nonterminal can be rewritten"""
        return lhs in self._rules_by_lhs

    def __getitem__(self, lhs):
        return self._rules_by_lhs.get(lhs, frozenset())

    def iterrules(self, lhs):
        return iter(self._rules_by_lhs.get(lhs, frozenset()))

    def __iter__(self):
        return chain(*self._rules_by_lhs.itervalues())

    def get(self, lhs, default=None):
        return self._rules_by_lhs.get(lhs, default)
    
    def iteritems(self):
        return self._rules_by_lhs.iteritems()
    
    def __str__(self):
        lines = []
        for lhs, rules in self.iteritems():
            for rule in rules:
                lines.append(str(rule))
        return '\n'.join(lines)


class _FrozenCFG(object):

    def __init__(self, rules):
        self._rules_by_lhs = defaultdict(None,
                {lhs: frozenset(group) 
                    for lhs, group in groupby(sorted(rules, key=lambda r: r.lhs), key=lambda r: r.lhs)}
                )

        self._nonterminals = set()
        self._terminals = set()
        for r in self:
            self._nonterminals.add(r.lhs)
            for s in r.rhs:
                if isinstance(s, Terminal):
                    self._terminals.add(s)
                else:
                    self._nonterminals.add(s)
        self._nonterminals = frozenset(self._nonterminals)
        self._terminals = frozenset(self._terminals)
    
    def unknown_words(self, words):
        return set(words) - self._terminals

    def __contains__(self, lhs):
        return lhs in self._rules_by_lhs

    def __getitem__(self, lhs):
        return self._rules_by_lhs.get(lhs, frozenset())

    def __iter__(self):
        return chain(*self._rules_by_lhs.itervalues())
    
    def get(self, lhs, default=None):
        return self._rules_by_lhs.get(lhs, default)

    def iteritems(self):
        return self._rules_by_lhs.iteritems()
    
    def __str__(self):
        lines = []
        for lhs, rules in self.iteritems():
            for rule in rules:
                lines.append(str(rule))
        return '\n'.join(lines)

def stars(rules):
    backward = defaultdict(set)
    forward = defaultdict(set)
    for r in rules:
        backward[r.lhs].add(r)
        [forward[sym].add(r) for sym in frozenset(r.rhs)]
    return backward, forward


