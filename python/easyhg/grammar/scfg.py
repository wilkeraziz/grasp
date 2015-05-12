"""
@author wilkeraziz
"""

from collections import defaultdict
from itertools import chain, groupby
from rule import CFGProduction
from cfg import FrozenCFG
import numpy as np


class SCFG(object):

    def __init__(self, syncrules=[]):
        """
        """
        self._syncrules_by_lhs = defaultdict(set, 
                {lhs: set(group) 
                    for lhs, group in groupby(sorted(syncrules, key=lambda syncr: syncr.lhs), key=lambda syncr: syncr.lhs)})

        self._syncrules_by_f = defaultdict(set,
                {(lhs, f_rhs): set(group)
                    for (lhs, f_rhs), group in groupby(sorted(syncrules, key=lambda syncr: (syncr.lhs, syncr.f_rhs)), 
                        key=lambda syncr: (syncr.lhs, syncr.f_rhs))})
            
    #def add(self, syncrule):
    #    self._syncrules_by_lhs[syncrule.lhs].add(syncrule)

    def __contains__(self, lhs):
        return lhs in self._syncrules_by_lhs

    def __getitem__(self, lhs):
        return self._syncrules_by_lhs.get(lhs, frozenset())

    def __iter__(self):
        return chain(*self._syncrules_by_lhs.itervalues())

    def get(self, lhs, default=None):
        return self._syncrules_by_lhs.get(lhs, default)
    
    def iteritems(self):
        return self._syncrules_by_lhs.iteritems()
    
    def __str__(self):
        lines = []
        for lhs, rules in self.iteritems():
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
        for syncr in chain(*self._syncrules_by_lhs.itervalues()):
            aux[(syncr.lhs, syncr.f_rhs)].append(syncr.weight)
        if not marginalise:
            return FrozenCFG(CFGProduction(lhs, f_rhs, semiring.one) for (lhs, f_rhs), weights in aux.iteritems())
        else: 
            return FrozenCFG(CFGProduction(lhs, f_rhs, reduce(semiring.plus, weights)) for (lhs, f_rhs), weights in aux.iteritems())

    def e_projection(self, semiring, marginalise=False):
        """
        A target projection is the grammar resulting from marginalising over the source rules.
        You can set plus=None if you want the projection to be unweighted, 
        or you can specify an operator (and a zero element).
        """
        # group rules by projection
        aux = defaultdict(list)
        for syncr in chain(*self._syncrules_by_lhs.itervalues()):
            aux[(syncr.lhs, syncr.project_rhs())].append(syncr.weight)
        if not marginalise:
            return FrozenCFG(CFGProduction(lhs, e_rhs, semiring.one) for (lhs, e_rhs), weights in aux.iteritems())
        else: 
            return FrozenCFG(CFGProduction(lhs, e_rhs, reduce(semiring.plus, weights)) for (lhs, e_rhs), weights in aux.iteritems())
        

    def iterprojections(self, lhs, fi_rhs, fo_rhs=None, weight=None, semiring=None): 
        """
        Returns an iterator for the target projections of a source rule
        >>> from symbol import Nonterminal
        >>> from rule import CFGProduction, SCFGProduction
        >>> r1 = SCFGProduction(Nonterminal('S'), [Nonterminal('X1'), Nonterminal('X2')], [Nonterminal('1'), Nonterminal('2')])
        >>> r2 = SCFGProduction(Nonterminal('S'), [Nonterminal('X1'), Nonterminal('X2')], [Nonterminal('2'), Nonterminal('1')])
        >>> G = SCFG([r1, r2])
        >>> projs = sorted((G.iterprojections(Nonterminal('S'), [Nonterminal('X1'), Nonterminal('X2')])))
        >>> set(projs) == set([CFGProduction(Nonterminal('S'), (Nonterminal('X2'), Nonterminal('X1')), 0.0), CFGProduction(Nonterminal('S'), (Nonterminal('X1'), Nonterminal('X2')), 0.0)])
        True
        """

        if weight is None:
            weight = one

        if fo_rhs is None:
            fo_rhs = fi_rhs

        syncrules = self._syncrules_by_f.get((lhs, tuple(fi_rhs)), frozenset())
        
        for e_rhs, group in groupby(sorted(syncrules, key=lambda syncr: syncr.project_rhs(fo_rhs)), 
                key=lambda syncr: syncr.project_rhs(fo_rhs)):
            if semiring is None:
                yield CFGProduction(lhs, e_rhs)
            else:
                yield CFGProduction(lhs, e_rhs, semiring.times(weight, reduce(semiring.plus, (syncr.weight for syncr in group), semiring.zero)))

    def iterrulesbyf(self, lhs, f_rhs):
        return iter(self._syncrules_by_f.get((lhs, tuple(f_rhs)), frozenset()))
