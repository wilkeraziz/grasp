"""
@author wilkeraziz
"""

import numpy as np
from collections import defaultdict
from itertools import chain, groupby
from .rule import CFGProduction
from .cfg import CFG
from .symbol import Terminal, Nonterminal
from functools import reduce


class SCFG(object):

    def __init__(self, syncrules=[]):
        """
        """
        self._syncrules_by_lhs = defaultdict(set)
        self._srules = defaultdict(lambda : defaultdict(set))
        self._nonterminals = set()
        self._sigma = set()  # source terminals
        self._delta = set()  # target terminals
        for srule in syncrules:
            self._syncrules_by_lhs[srule.lhs].add(srule)
            self._srules[srule.lhs][srule.f_rhs].add(srule)
            self._nonterminals.add(srule.lhs)
            self._nonterminals.update(filter(lambda s: isinstance(s, Nonterminal), srule.f_rhs))
            self._sigma.update(filter(lambda s: isinstance(s, Terminal), srule.f_rhs))
            self._delta.update(filter(lambda s: isinstance(s, Terminal), srule.e_rhs))
    
    @property
    def sigma(self):
        return self._sigma

    @property
    def delta(self):
        return self._delta

    @property
    def nonterminals(self):
        return self._nonterminals

    def add(self, srule):
        self._syncrules_by_lhs[srule.lhs].add(srule)
        self._srules[srule.lhs][srule.f_rhs].add(srule)
        self._nonterminals.add(srule.lhs)
        self._nonterminals.update(filter(lambda s: isinstance(s, Nonterminal), srule.f_rhs))
        self._sigma.update(filter(lambda s: isinstance(s, Terminal), srule.f_rhs))
        self._delta.update(filter(lambda s: isinstance(s, Terminal), srule.e_rhs))


    def __contains__(self, lhs):
        return lhs in self._syncrules_by_lhs

    def __getitem__(self, lhs):
        return self._syncrules_by_lhs.get(lhs, frozenset())

    def __iter__(self):
        return chain(*iter(self._syncrules_by_lhs.values()))

    def get(self, lhs, default=None):
        return self._syncrules_by_lhs.get(lhs, default)
    
    def iteritems(self):
        return iter(self._syncrules_by_lhs.items())
    
    def __str__(self):
        lines = []
        for lhs, rules in self.items():
            for rule in rules:
                lines.append(str(rule))
        return '\n'.join(lines)

    def f_projection(self, semiring, marginalise=False):
        """
        A source projection is the grammar resulting from marginalising over the target rules.
        You can set plus=None if you want the projection to be unweighted, 
        or you can specify an operator (and a zero element).
        """
        # group rules by projection
        aux = defaultdict(list)
        for syncr in chain(*iter(self._syncrules_by_lhs.values())):
            aux[(syncr.lhs, syncr.f_rhs)].append(syncr.weight)
        if not marginalise:
            return CFG(CFGProduction(lhs, f_rhs, semiring.one) for (lhs, f_rhs), weights in aux.items())
        else: 
            return CFG(CFGProduction(lhs, f_rhs, reduce(semiring.plus, weights)) for (lhs, f_rhs), weights in aux.items())

    def e_projection(self, semiring, marginalise=False):
        """
        A target projection is the grammar resulting from marginalising over the source rules.
        You can set plus=None if you want the projection to be unweighted, 
        or you can specify an operator (and a zero element).
        """
        # group rules by projection
        aux = defaultdict(list)
        for syncr in chain(*iter(self._syncrules_by_lhs.values())):
            aux[(syncr.lhs, syncr.project_rhs())].append(syncr.weight)
        if not marginalise:
            return CFG(CFGProduction(lhs, e_rhs, semiring.one) for (lhs, e_rhs), weights in aux.items())
        else: 
            return CFG(CFGProduction(lhs, e_rhs, reduce(semiring.plus, weights)) for (lhs, e_rhs), weights in aux.items())

    def iterrulesbyf(self, lhs, f_rhs):
        srules = self._srules.get(lhs, None)
        if srules is None:
            return iter([])
        return iter(srules.get(f_rhs, []))
