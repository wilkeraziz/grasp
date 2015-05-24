"""
@author wilkeraziz
"""

from collections import defaultdict, deque
from itertools import chain, groupby, ifilter
from symbol import Terminal, Nonterminal
from topsort import topsort


class CFG(object):

    def __init__(self, rules=[]):
        self._rules_by_lhs = defaultdict(set)
        self._nonterminals = set()
        self._sigma = set()  # terminals
        for rule in rules:
            self.add(rule)

    def __len__(self):
        return sum(len(rules) for rules in self._rules_by_lhs.itervalues())

    @property
    def terminals(self):
        return self._sigma

    def is_terminal(self, terminal):
        return terminal in self._sigma

    @property
    def nonterminals(self):
        return self._nonterminals

    def add(self, rule):
        self._rules_by_lhs[rule.lhs].add(rule)
        self._nonterminals.add(rule.lhs)
        self._nonterminals.update(ifilter(lambda s: isinstance(s, Nonterminal), rule.rhs))
        self._sigma.update(ifilter(lambda s: isinstance(s, Terminal), rule.rhs))
        
    def __contains__(self, lhs):
        """Tests whether a given nonterminal can be rewritten"""
        return lhs in self._rules_by_lhs

    def __getitem__(self, lhs):
        return self._rules_by_lhs.get(lhs, frozenset())

    def iterrules(self, lhs=None):
        return chain(*self._rules_by_lhs.itervalues()) if lhs is None else iter(self._rules_by_lhs.get(lhs, frozenset()))

    def itersymbols(self, terminals=True, nonterminals=True):
        return chain(self._sigma, self._nonterminals)

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
    
    def cleanup(self, roots):
        seen = set(roots)
        Q = deque(seen)
        while Q:
            lhs = Q.popleft()
            rules = self.get(lhs, set())
            if not rules:
                self._rules_by_lhs.pop(lhs, None)
            else:
                for rule in rules:
                    for nt in ifilter(lambda s: isinstance(s, Nonterminal) and s not in seen, rule.rhs):
                        Q.append(nt)
                        seen.add(nt)

    def iterrules_topdown(self, consistent=True):
        for group in reversed(list(topsort_cfg(self))):
            for sym in sorted(group, key=lambda s: str(s)):
                for rule in sorted(self._rules_by_lhs.get(sym, set()), key=lambda r: str(r)):
                    yield rule


def stars(rules):
    backward = defaultdict(set)
    forward = defaultdict(set)
    for r in rules:
        backward[r.lhs].add(r)
        [forward[sym].add(r) for sym in frozenset(r.rhs)]
    return backward, forward


def topsort_cfg(cfg):
    # make dependencies
    D = defaultdict(set)  
    for v in cfg.nonterminals:
        deps = D[v]
        for r in cfg.iterrules(v):
            deps.update(r.rhs)
    return topsort(D, cfg.terminals)

