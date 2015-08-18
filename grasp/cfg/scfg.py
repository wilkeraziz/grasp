"""
:Authors: - Wilker Aziz
"""

from collections import defaultdict
from itertools import chain
from .rule import CFGProduction
from .cfg import CFG
from .symbol import Terminal, Nonterminal
from functools import reduce


def _new_defdict_set():
    return defaultdict(set)


class _SCFG(object):

    def __init__(self, syncrules=[]):
        """
        """
        self._syncrules_by_lhs = defaultdict(set)
        self._srules = defaultdict(_new_defdict_set)
        self._nonterminals = set()
        self._sigma = set()  # source terminals
        self._delta = set()  # target terminals
        for srule in syncrules:
            self.add(srule)

    @property
    def sigma(self):
        return self._sigma

    @property
    def delta(self):
        return self._delta

    @property
    def nonterminals(self):
        return self._nonterminals

    def n_nonterminals(self):
        return len(self._nonterminals)

    def n_terminals(self):
        return len(self._sigma) + len(self._delta)

    def __len__(self):
        return sum(len(rules) for rules in self._syncrules_by_lhs.values())

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


class SCFG(object):
    """
    A Synchronous CFG. Note that SCFG does not inherit from CFG and should not be assumed
    to have a similar interface.

    We treat a SCFG pretty much as a hash table between
    from (lhs, irhs) to a list of synchronous rules.

    You can use SCFG objects to manage synchronous rules and to get input/output projections as CFG objects.
    """

    def __init__(self, syncrules=[]):
        """

        :param syncrules: an iterable sequence of synchronous rules.
        :return:
        """
        self._srules = defaultdict(lambda: defaultdict(list))
        self._sigma = set()
        self._delta = set()
        self._rules = []
        for srule in syncrules:
            self.add(srule)

    def __len__(self):
        """The total number of synchronous rules in the container."""
        return sum(sum(len(rules) for rules in by_irhs.values()) for by_irhs in self._srules.values())

    def in_ivocab(self, word):
        return word in self._sigma

    def in_ovocab(self, word):
        return word in self._delta

    def __iter__(self):
        return iter(self._rules)

    def add(self, srule):
        """Add a synchronous rule to the container."""
        self._rules.append(srule)
        self._srules[srule.lhs][srule.irhs].append(srule)
        self._sigma.update(filter(lambda s: isinstance(s, Terminal), srule.irhs))
        self._delta.update(filter(lambda s: isinstance(s, Terminal), srule.orhs))

    def __str__(self):
        lines = []
        for lhs, by_irhs in self._srules.items():
            for i_rhs, rules in by_irhs.items():
                for rule in rules:
                    lines.append(str(rule))
        return '\n'.join(lines)

    def input_projection(self, semiring, weighted=False):
        """
        A projection is the grammar resulting from marginalising over the alternative rules.
        You can set `weighted` to False if you want the projection to be unweighted.
        :param semiring: must provide `zero`, `one` and `plus`.
        :param weighted:
        :return: a CFG.
        """
        if not weighted:
            def make_rule(lhs, rhs, srules):
                return CFGProduction(lhs, rhs, semiring.one)
        else:
            def make_rule(lhs, rhs, srules):
                return CFGProduction(lhs, rhs, reduce(semiring.plus, (r.weight for r in srules), semiring.zero))

        def iterrules():
            for lhs, by_irhs in self._srules.items():
                for f_rhs, srules in by_irhs.items():
                    yield make_rule(lhs, f_rhs, srules)

        return CFG(iterrules())

    def iteroutputrules(self, lhs, irhs):
        """Iterate through synchronous rules matching given LHS symbol and a given input RHS sequence."""
        srules = self._srules.get(lhs, None)
        if srules is None:
            return iter([])
        return iter(srules.get(irhs, []))
