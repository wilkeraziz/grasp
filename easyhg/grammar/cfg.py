"""
This module contains class definitions for weighted context-free grammars and utilitary functions.

:Authors: - Wilker Aziz
"""

from collections import defaultdict, deque
from itertools import chain
from .symbol import Terminal, Nonterminal
from .topsort import robust_topological_sort
from .rule import CFGProduction
from .grammar import Grammar

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


class TopSortTable(object):

    def __init__(self, forest):  # TODO: implement callback to update the table when the forest changes
        # gathers the dependencies between nonterminals
        deps = defaultdict(set)
        for lhs, rules in forest.iteritems():
            syms = deps[lhs]
            for rule in rules:
                syms.update(filter(lambda s: isinstance(s, Nonterminal), rule.rhs))
        order = robust_topological_sort(deps)
        # adds terminals to the bottom-level
        order.appendleft(frozenset(frozenset([t]) for t in forest.iterterminals()))
        self._topsort = order

    def n_levels(self):
        return len(self._topsort)

    def n_top_symbols(self):
        return sum(len(b) for b in self.itertopbuckets())

    def n_top_buckets(self):
        return len(self._topsort[-1])

    def n_loopy_symbols(self):
        return sum(len(buckets) for buckets in filter(lambda b: len(b) > 1, self.iterbuckets()))

    def n_cycles(self):
        return sum(1 for _ in filter(lambda b: len(b) > 1, self.iterbuckets()))

    def topsort(self):
        return self._topsort

    def itertopbuckets(self):
        """Iterate over the top buckets of the grammar/forest/hypergraph"""
        return iter(self.topsort()[-1])

    def iterbottombuckets(self):
        return iter(self.topsort()[0])

    def root(self):
        """
        Return the start/root/goal symbol/node of the grammar/forest/hypergraph
        :return: node
        """
        toplevel = self.topsort()[-1]  # top-most set of buckets
        if len(toplevel) > 1:  # more than one bucket
            raise ValueError('I expected a single bucket instead of %d' % len(toplevel))
        top = next(iter(toplevel))  # at this point we know there is only one top-level bucket
        if len(top) > 1:  # sometimes this is a loopy bucket (more than one node)
            raise ValueError('I expected a single start symbol instead of %d' % len(top))
        return next(iter(top))  # here we know there is only one start symbol

    def __iter__(self):
        """Iterates over all buckets in bottom-up order"""
        return self.iterbuckets()

    def iterlevels(self, reverse=False, skip=0):
        """
        Iterate level by level (a level is a set of buckets sharing a ranking).
        In Goodman (1999), a bucket which is not a singleton is called a "loopy bucket".

        :param reverse: bottom-up if False, top-down if True
        :param skip: skip a number of levels
        :return: iterator over levels
        """
        # bottom-up vs top-down
        if not reverse:
            iterator = iter(self.topsort())
        else:
            iterator = reversed(self.topsort())
        # skipping n levels
        for n in range(skip):
            next(iterator)
        return iterator

    def iterbuckets(self, reverse=False, skip=0):
        """
        Iterate bucket by bucket (a bucket is a set of strongly connected nodes).
        In Goodman (1999), a bucket which is not a singleton is called a "loopy bucket".

        :param reverse: bottom-up if False, top-down if True
        :param skip: skip a number of levels
        :return: iterator over buckets
        """
        iterator = self.iterlevels(reverse, skip)
        for buckets in iterator:
            for bucket in buckets:
                yield bucket

    def __str__(self):
        lines = []
        for i, level in enumerate(self.iterlevels()):
            lines.append('level=%d' % i)
            for bucket in level:
                if len(bucket) > 1:
                    lines.append(' (loopy) {0}'.format(' '.join(str(x) for x in bucket)))
                else:
                    lines.append(' {0}'.format(' '.join(str(x) for x in bucket)))
        return '\n'.join(lines)








