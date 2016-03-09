"""
:Authors: - Wilker Aziz
"""

import unittest

from grasp.formal.hg import Hypergraph
from grasp.cfg import CFG, Terminal, Nonterminal
from grasp.cfg.rule import NewCFGProduction as CFGProduction
from grasp.formal.scfgop import cfg_to_hg
from grasp.cfg.model import PCFG


def make_production(lhs, rhs, weight):
    return CFGProduction(lhs, rhs, {'Prob': weight})


class HypergraphTestCase(unittest.TestCase):

    def setUp(self):
        self.cfg = CFG()
        self.cfg.add(make_production(Nonterminal('S'), [Nonterminal('S'), Nonterminal('X')], 0.9))
        self.cfg.add(make_production(Nonterminal('S'), [Nonterminal('X')], 0.1))
        self.cfg.add(make_production(Nonterminal('X'), [Terminal('a')], 1.0))
        self.model = PCFG('Prob')

    def test_construct(self):
        hg = Hypergraph()
        self.assertEqual(hg.n_nodes(), 0)
        self.assertEqual(hg.n_edges(), 0)

    def test_update(self):
        hg = cfg_to_hg([self.cfg], [], self.model)
        self.assertEqual(hg.n_nodes(), 3)
        self.assertEqual(hg.n_edges(), 3)

    def test_nonterminal(self):
        hg = Hypergraph()
        hg.add_node(Nonterminal('S'))
        S = hg.fetch(Nonterminal('S'))
        self.assertNotEqual(S, -1)
        self.assertEqual(hg.label(S), Nonterminal('S'))
        self.assertTrue(hg.is_nonterminal(S))

    def test_terminal(self):
        hg = Hypergraph()
        hg.add_node(Terminal('a'))
        a = hg.fetch(Terminal('a'))
        self.assertNotEqual(a, -1)
        self.assertEqual(hg.label(a), Terminal('a'))
        self.assertTrue(hg.is_terminal(a))

    def test_rule(self):
        hg = Hypergraph()
        rule = make_production(Nonterminal('S'), [Nonterminal('X'), Terminal('a')], 1.0)
        e = hg.add_xedge(rule.lhs, rule.rhs, self.model(rule), rule, False)
        self.assertEqual(hg.rule(e), rule)
        self.assertEqual(hg.n_nodes(), 3)
        self.assertEqual(hg.n_edges(), 1)

    def test_cfg(self):
        hg = cfg_to_hg([self.cfg], [], self.model)

        S = hg.fetch(Nonterminal('S'))
        X = hg.fetch(Nonterminal('X'))
        a = hg.fetch(Terminal('a'))

        self.assertTrue(hg.is_source(a))
        self.assertFalse(hg.is_source(X))
        self.assertFalse(hg.is_source(S))
        self.assertEqual(len(list(hg.iterbs(S))), 2)
        self.assertEqual(len(list(hg.iterbs(X))), 1)
        self.assertEqual(len(list(hg.iterbs(a))), 0)
        self.assertEqual(hg.label(S), Nonterminal('S'))
        self.assertEqual(hg.label(X), Nonterminal('X'))
        self.assertEqual(hg.label(a), Terminal('a'))

    def test_stars(self):
        hg = Hypergraph()
        r1 = make_production(Nonterminal('S'), [Nonterminal('S'), Nonterminal('X')], 0.5)
        r2 = make_production(Nonterminal('S'), [Nonterminal('X')], 0.5)
        r3 = make_production(Nonterminal('X'), [Terminal('a')], 0.5)
        e1 = hg.add_xedge(r1.lhs, r1.rhs, self.model(r1), r1, False)
        e2 = hg.add_xedge(r2.lhs, r2.rhs, self.model(r2), r2, False)
        e3 = hg.add_xedge(r3.lhs, r3.rhs, self.model(r3), r3, False)
        S = hg.fetch(Nonterminal('S'))
        X = hg.fetch(Nonterminal('X'))
        a = hg.fetch(Terminal('a'))

        self.assertSequenceEqual(set(hg.iterfs(S)), {e1})
        self.assertSequenceEqual(set(hg.iterfs(X)), {e1, e2})
        self.assertSequenceEqual(set(hg.iterfs(a)), {e3})

        self.assertSequenceEqual(set(hg.iterbs(S)), {e1, e2})
        self.assertSequenceEqual(set(hg.iterbs(X)), {e3})
        self.assertSequenceEqual(set(hg.iterbs(a)), set())

        self.assertSequenceEqual(set(hg.iterdeps(S)), {S, X})
        self.assertSequenceEqual(set(hg.iterdeps(X)), {a})
        self.assertSequenceEqual(set(hg.iterdeps(a)), set())


if __name__ == '__main__':
    unittest.main()
