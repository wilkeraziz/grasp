"""
:Authors: - Wilker Aziz
"""

import unittest


from grasp.formal.topsort import AcyclicTopSortTable, RobustTopSortTable
from grasp.alg.value import acyclic_value_recursion, robust_value_recursion
from grasp.cfg import CFG, Terminal, Nonterminal
from grasp.cfg.rule import NewCFGProduction as CFGProduction
import grasp.semiring as semiring
from grasp.formal.scfgop import cfg_to_hg
from grasp.cfg.model import PCFG


def get_rule(lhs, rhs, prob):
    return CFGProduction.MakeStandardCFGProduction(lhs, rhs, prob, fname='Prob', transform=float)


class AcyclicValueRecursionTestCase(unittest.TestCase):

    def setUp(self):
        self.cfg = CFG()
        self.cfg.add(get_rule(Nonterminal('S02'),
                              [Nonterminal('S01'), Nonterminal('X12'), Nonterminal('PUNC')],
                              0.5))
        self.cfg.add(get_rule(Nonterminal('S01'), [Nonterminal('X01')], 0.1))
        self.cfg.add(get_rule(Nonterminal('X01'), [Terminal('Hello')], 0.7))
        self.cfg.add(get_rule(Nonterminal('X01'), [Terminal('hello')], 0.1))
        self.cfg.add(get_rule(Nonterminal('X12'), [Terminal('World')], 0.6))
        self.cfg.add(get_rule(Nonterminal('X12'), [Terminal('world')], 0.2))
        self.cfg.add(get_rule(Nonterminal('PUNC'), [Terminal('!')], 0.1))
        self.cfg.add(get_rule(Nonterminal('PUNC'), [Terminal('!!!')], 0.3))
        self.cfg.add(get_rule(Nonterminal('A'), [Terminal('dead')], 0.3))
        self.cfg.add(get_rule(Nonterminal('B'), [], 0.3))
        self.forest = cfg_to_hg([self.cfg], [], PCFG('Prob'))
        self.tsort = AcyclicTopSortTable(self.forest)

    def test_root(self):
        self.assertEqual(self.tsort.root(), self.forest.fetch(Nonterminal('S02')))

    def test_value(self):
        values = acyclic_value_recursion(self.forest, self.tsort, semiring.prob)
        self.assertAlmostEqual(values[self.tsort.root()], 0.0128)
        self.assertAlmostEqual(values[self.forest.fetch(Nonterminal('S01'))], 0.08)
        self.assertAlmostEqual(values[self.forest.fetch(Nonterminal('X01'))], 0.8)
        self.assertAlmostEqual(values[self.forest.fetch(Nonterminal('X12'))], 0.8)
        self.assertAlmostEqual(values[self.forest.fetch(Nonterminal('PUNC'))], 0.4)
        self.assertAlmostEqual(values[self.forest.fetch(Terminal('Hello'))], 1.0)
        self.assertAlmostEqual(values[self.forest.fetch(Terminal('hello'))], 1.0)
        self.assertAlmostEqual(values[self.forest.fetch(Terminal('world'))], 1.0)
        self.assertAlmostEqual(values[self.forest.fetch(Terminal('World'))], 1.0)
        self.assertAlmostEqual(values[self.forest.fetch(Terminal('!'))], 1.0)
        self.assertAlmostEqual(values[self.forest.fetch(Terminal('!!!'))], 1.0)
        self.assertAlmostEqual(values[self.forest.fetch(Nonterminal('A'))], 0.3)
        self.assertAlmostEqual(values[self.forest.fetch(Nonterminal('B'))], 0)

    def test_robust_value(self):
        tsort = RobustTopSortTable(self.forest)
        values = robust_value_recursion(self.forest, tsort, semiring.prob)
        self.assertAlmostEqual(values[self.tsort.root()], 0.0128)
        self.assertAlmostEqual(values[self.forest.fetch(Nonterminal('S01'))], 0.08)
        self.assertAlmostEqual(values[self.forest.fetch(Nonterminal('X01'))], 0.8)
        self.assertAlmostEqual(values[self.forest.fetch(Nonterminal('X12'))], 0.8)
        self.assertAlmostEqual(values[self.forest.fetch(Nonterminal('PUNC'))], 0.4)
        self.assertAlmostEqual(values[self.forest.fetch(Terminal('Hello'))], 1.0)
        self.assertAlmostEqual(values[self.forest.fetch(Terminal('hello'))], 1.0)
        self.assertAlmostEqual(values[self.forest.fetch(Terminal('world'))], 1.0)
        self.assertAlmostEqual(values[self.forest.fetch(Terminal('World'))], 1.0)
        self.assertAlmostEqual(values[self.forest.fetch(Terminal('!'))], 1.0)
        self.assertAlmostEqual(values[self.forest.fetch(Terminal('!!!'))], 1.0)
        self.assertAlmostEqual(values[self.forest.fetch(Nonterminal('A'))], 0.3)
        self.assertAlmostEqual(values[self.forest.fetch(Nonterminal('B'))], 0)


if __name__ == '__main__':
    unittest.main()
