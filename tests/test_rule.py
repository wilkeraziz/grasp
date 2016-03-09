__author__ = 'waziz'

import unittest
from grasp.cfg.symbol import Terminal, Nonterminal
from grasp.cfg.rule import NewCFGProduction as CFGProduction
from grasp.cfg.model import PCFG


def make_production(lhs, rhs, weight):
    return CFGProduction(lhs, rhs, {'Prob': weight, 'Dummy': 0})


class CFGProductionTestCase(unittest.TestCase):

    def setUp(self):
        self.S_SX = make_production(Nonterminal('S'), [Nonterminal('S'), Nonterminal('X')], 0.9)
        self.S_X = make_production(Nonterminal('S'), [Nonterminal('X')], 0.1)
        self.X_a = make_production(Nonterminal('X'), [Terminal('a')], 1.0)
        self.model = PCFG('Prob')

    def test_lhs(self):
        self.assertEqual(self.S_SX.lhs, Nonterminal('S'))
        self.assertEqual(self.S_X.lhs, Nonterminal('S'))
        self.assertEqual(self.X_a.lhs, Nonterminal('X'))

    def test_rhs(self):
        self.assertEqual(self.S_SX.rhs, (Nonterminal('S'), Nonterminal('X')))
        self.assertEqual(self.S_X.rhs, (Nonterminal('X'),))
        self.assertEqual(self.X_a.rhs, (Terminal('a'),))

    def test_weight_property(self):
        with self.assertRaises(AttributeError):
            self.X_a.weight

    def test_weight(self):
        self.assertEqual(self.model(self.S_SX), 0.9)
        self.assertEqual(self.model(self.S_X), 0.1)
        self.assertEqual(self.model(self.X_a), 1.0)

    def test_dummy_weight(self):
        model = PCFG(fname='Dummy')
        self.assertEqual(model(self.S_SX), 0)
        self.assertEqual(model(self.S_X), 0)
        self.assertEqual(model(self.X_a), 0)


if __name__ == '__main__':
    unittest.main()
