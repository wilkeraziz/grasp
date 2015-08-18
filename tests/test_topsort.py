"""
:Authors: - Wilker Aziz
"""

import unittest

from grasp.formal.hg import Hypergraph

from grasp.formal.topsort import AcyclicTopSortTable, RobustTopSortTable
from grasp.cfg import CFGProduction, Terminal, Nonterminal


class TopSortTableTestCase(unittest.TestCase):

    def test_acyclic_topsort(self):
        hg = Hypergraph()
        r1 = CFGProduction(Nonterminal('TOP'), [Nonterminal('S:0-1'), Nonterminal('X:1-2')], 0.5)
        r2 = CFGProduction(Nonterminal('S:0-1'), [Nonterminal('X:0-1')], 0.5)
        r3 = CFGProduction(Nonterminal('X:0-1'), [Terminal('a')], 0.5)
        r4 = CFGProduction(Nonterminal('X:1-2'), [Terminal('b')], 0.5)
        r5 = CFGProduction(Nonterminal('Z'), [], 0.5)
        e1 = hg.add_edge(r1)
        e2 = hg.add_edge(r2)
        e3 = hg.add_edge(r3)
        e4 = hg.add_edge(r4)
        e5 = hg.add_edge(r5)
        TOP = hg.fetch(Nonterminal('TOP'))
        S01 = hg.fetch(Nonterminal('S:0-1'))
        X01 = hg.fetch(Nonterminal('X:0-1'))
        X12 = hg.fetch(Nonterminal('X:1-2'))
        Z = hg.fetch(Nonterminal('Z'))
        a = hg.fetch(Terminal('a'))
        b = hg.fetch(Terminal('b'))
        tsort = AcyclicTopSortTable(hg)

        self.assertEqual(tsort.n_levels(), 5)  # 5 levels
        self.assertEqual(tsort.level(TOP), 4)
        self.assertEqual(tsort.level(S01), 3)
        self.assertEqual(tsort.level(X01), 2)
        self.assertEqual(tsort.level(X12), 2)
        self.assertEqual(tsort.level(a), 1)
        self.assertEqual(tsort.level(b), 1)
        self.assertEqual(tsort.level(Z), 0)
        self.assertEqual(tsort.root(), TOP)

        self.assertEqual(list(tsort.iternodes()), [Z, a, b, X12, X01, S01, TOP])

    def test_tarjan(self):
        hg = Hypergraph()
        rules = [
            CFGProduction(Nonterminal('S'), [Nonterminal('S'), Nonterminal('X')], 0.5),
            CFGProduction(Nonterminal('S'), [Nonterminal('X')], 0.5),
            CFGProduction(Nonterminal('X'), [Terminal('a')], 0.5),
            CFGProduction(Nonterminal('T'), [Nonterminal('X')], 0.5),
            CFGProduction(Nonterminal('ROOT'), [Nonterminal('ROOT'), Nonterminal('S')], 0.1),
            CFGProduction(Nonterminal('ROOT'), [Nonterminal('S')], 0.5),
            CFGProduction(Nonterminal('ROOT'), [Nonterminal('T')], 0.5),
            CFGProduction(Nonterminal('S'), [Nonterminal('T')], 0.5),
            CFGProduction(Nonterminal('T'), [Nonterminal('S')], 0.5),
            CFGProduction(Nonterminal('ROOT2'), [Nonterminal('D')], 0.5),
            CFGProduction(Nonterminal('X'), [Terminal('b')], 0.5),
            CFGProduction(Nonterminal('ROOT2'), [Terminal('c')], 0.5),
            CFGProduction(Nonterminal('S2'), [Nonterminal('T2')], 0.5),
            CFGProduction(Nonterminal('T2'), [Nonterminal('S2')], 0.5),
            CFGProduction(Nonterminal('T2'), [Nonterminal('X')], 0.5)]

        for r in rules:
            hg.add_edge(r)

        tsort = RobustTopSortTable(hg)

        self.assertEqual(tsort.n_levels(), 5)  # 5 levels

        expected_order = [
            {frozenset({hg.fetch(Nonterminal('D'))})},  # level 0
            {frozenset({hg.fetch(Terminal('a'))}),
             frozenset({hg.fetch(Terminal('b'))}),
             frozenset({hg.fetch(Terminal('c'))})}, # level 1
            {frozenset({hg.fetch(Nonterminal('X'))}),
             frozenset({hg.fetch(Nonterminal('ROOT2'))})}, # level 2
            {frozenset({hg.fetch(Nonterminal('S')), hg.fetch(Nonterminal('T'))}),
             frozenset({hg.fetch(Nonterminal('S2')), hg.fetch(Nonterminal('T2'))})}, # level 3
            {frozenset({hg.fetch(Nonterminal('ROOT'))})}  # level 4
        ]

        loopy = [False, False, False, False, False, False, True, True, True]
        order = []
        i = 0
        for level in tsort.iterlevels():
            order.append(set())
            for bucket in level:
                order[-1].add(frozenset(bucket))
                self.assertEqual(tsort.is_loopy(bucket), loopy[i])
                i += 1

        self.assertEqual(expected_order, order)

    def test_tarjan_no0(self):
        hg = Hypergraph()
        rules = [
            CFGProduction(Nonterminal('S'), [Nonterminal('S'), Nonterminal('X')], 0.5),
            CFGProduction(Nonterminal('S'), [Nonterminal('X')], 0.5),
            CFGProduction(Nonterminal('X'), [Terminal('a')], 0.5),
            CFGProduction(Nonterminal('T'), [Nonterminal('X')], 0.5),
            CFGProduction(Nonterminal('ROOT'), [Nonterminal('S')], 0.5),
            CFGProduction(Nonterminal('ROOT'), [Nonterminal('T')], 0.5),
            CFGProduction(Nonterminal('S'), [Nonterminal('T')], 0.5),
            CFGProduction(Nonterminal('T'), [Nonterminal('S')], 0.5),
            CFGProduction(Nonterminal('ROOT2'), [Terminal('d')], 0.5),
            CFGProduction(Nonterminal('X'), [Terminal('b')], 0.5),
            CFGProduction(Nonterminal('ROOT2'), [Terminal('c')], 0.5),
            CFGProduction(Nonterminal('S2'), [Nonterminal('T2')], 0.5),
            CFGProduction(Nonterminal('T2'), [Nonterminal('S2')], 0.5),
            CFGProduction(Nonterminal('T2'), [Nonterminal('X')], 0.5)]

        for r in rules:
            hg.add_edge(r)

        tsort = RobustTopSortTable(hg)

        self.assertEqual(tsort.n_levels(), 5)  # 5 levels

        expected_order = [
            set(),  # level 0
            {frozenset({hg.fetch(Terminal('a'))}),
             frozenset({hg.fetch(Terminal('b'))}),
             frozenset({hg.fetch(Terminal('c'))}),
             frozenset({hg.fetch(Terminal('d'))})}, # level 1
            {frozenset({hg.fetch(Nonterminal('X'))}),
             frozenset({hg.fetch(Nonterminal('ROOT2'))})}, # level 2
            {frozenset({hg.fetch(Nonterminal('S')), hg.fetch(Nonterminal('T'))}),
             frozenset({hg.fetch(Nonterminal('S2')), hg.fetch(Nonterminal('T2'))})}, # level 3
            {frozenset({hg.fetch(Nonterminal('ROOT'))})}  # level 4
        ]

        order = []
        for level in tsort.iterlevels():
            order.append(set())
            for bucket in level:
                order[-1].add(frozenset(bucket))

        self.assertEqual(expected_order, order)


if __name__ == '__main__':
    unittest.main()
