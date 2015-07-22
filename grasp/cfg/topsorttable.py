"""
:Authors: - Wilker Aziz
"""

from collections import defaultdict
from .topsort import robust_topological_sort
from .symbol import Nonterminal


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
            raise ValueError('I expected a single bucket instead of %d\n%s' % (len(toplevel), '\n'.join(str(s) for s in toplevel)))
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


class LazyTopSortTable(object):

    def __init__(self, forest):
        self._forest = forest
        self._tsort = None

    def do(self):
        if self._tsort is None:
            self._tsort = TopSortTable(self._forest)
        return self._tsort