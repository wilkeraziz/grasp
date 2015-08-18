"""
:Authors: - Wilker Aziz
"""

import unittest
from grasp.formal.fsa import DFA, make_dfa
from grasp.cfg.symbol import Terminal


class DFATestCase(unittest.TestCase):

    def setUp(self):
        self.dfa = DFA()
        self.dfa.add_state()
        self.dfa.add_state()
        self.dfa.add_state()
        self.dfa.make_initial(0)
        self.dfa.make_final(2)
        self.dfa.add_arc(0, 1, Terminal('a'), 1)
        self.dfa.add_arc(1, 2, Terminal('b'), 1)

    def test_construct(self):
        self.assertEqual(DFA().n_states(), 0)
        self.assertEqual(DFA().n_arcs(), 0)

    def test_add_state(self):
        dfa = DFA()
        dfa.add_state()
        self.assertEqual(dfa.n_states(), 1)

    def test_add_arc(self):
        dfa = DFA()
        with self.assertRaises(IndexError):
            dfa.add_arc(0, 0, Terminal('a'), 0)
        dfa.add_state()
        dfa.add_arc(0, 0, Terminal('a'), 0)
        self.assertEqual(dfa.n_arcs(), 1)

    def test_fetch(self):
        dfa = DFA()
        with self.assertRaises(IndexError):
            dfa.fetch(0, Terminal('a'))
        self.assertEqual(self.dfa.fetch(0, Terminal('a')), 0)
        self.assertEqual(self.dfa.fetch(0, Terminal('b')), -1)
        self.assertEqual(self.dfa.fetch(1, Terminal('a')), -1)
        self.assertEqual(self.dfa.fetch(1, Terminal('b')), 1)
        self.assertEqual(self.dfa.fetch(2, Terminal('c')), -1)

    def test_arc(self):
        dfa = DFA()
        with self.assertRaises(IndexError):
            dfa.arc(0)
        arc = self.dfa.arc(self.dfa.fetch(0, Terminal('a')))
        self.assertEqual(arc.origin, 0)
        self.assertEqual(arc.destination, 1)
        self.assertEqual(arc.label, Terminal('a'))
        self.assertEqual(arc.weight, 1)

    def test_is_initial(self):
        self.assertTrue(self.dfa.is_initial(0))
        self.assertFalse(self.dfa.is_initial(1))
        self.assertFalse(self.dfa.is_initial(2))

    def test_is_final(self):
        self.assertTrue(self.dfa.is_final(2))
        self.assertFalse(self.dfa.is_final(0))
        self.assertFalse(self.dfa.is_final(1))

    def test_n_states(self):
        self.assertEqual(self.dfa.n_states(), 3)

    def test_n_arcs(self):
        self.assertEqual(self.dfa.n_arcs(), 2)

    def test_make_dfa(self):
        dfa = make_dfa('this is cool'.split(), 1.0)
        self.assertEqual(dfa.n_states(), 4)
        self.assertEqual(dfa.n_arcs(), 3)
        self.assertSequenceEqual(set(dfa.iterinitial()), {0})
        self.assertSequenceEqual(set(dfa.iterfinal()), {3})
        self.assertListEqual([str(arc.label) for arc in dfa.iterarcs()], 'this is cool'.split())
        self.assertEqual(sum(arc.weight for arc in dfa.iterarcs()), 3)


if __name__ == '__main__':
    unittest.main()
