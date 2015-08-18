"""
This module contains unit tests for classes and functions in grasp.cfg.symbol

:Authors: - Wilker Aziz
"""

import unittest

from grasp.cfg.symbol import Symbol, Terminal, Nonterminal, Span, StaticSymbolFactory, SymbolFactory


class TerminalTestCase(unittest.TestCase):

    def setUp(self):
        self.a = Terminal('a')
        self.a2 = Terminal('a')

    def test_inheritance(self):
        self.assertIsInstance(self.a, Symbol)

    def test_underlying(self):
        self.assertEqual(self.a.underlying, 'a')

    def test_underlying_type(self):
        self.assertIs(type(self.a.underlying), str)

    def test_surface(self):
        self.assertEqual(self.a.surface, self.a.underlying)

    def test_instance_management(self):
        self.assertIsNot(self.a, self.a2)

    def test_str(self):
        self.assertEqual(str(self.a), 'a')

    def test_repr(self):
        self.assertEqual(repr(self.a), "'a'")


class NonterminalTestCase(unittest.TestCase):

    def setUp(self):
        self.X = Nonterminal('X')
        self.X2 = Nonterminal('X')

    def test_inheritance(self):
        self.assertIsInstance(self.X, Symbol)

    def test_underlying(self):
        self.assertEqual(self.X.underlying, 'X')

    def test_underlying_type(self):
        self.assertIs(type(self.X.underlying), str)

    def test_label(self):
        self.assertEqual(self.X.label, self.X.underlying)

    def test_instance_management(self):
        self.assertIsNot(self.X, self.X2)

    def test_str(self):
        self.assertEqual(str(self.X), 'X')

    def test_repr(self):
        self.assertEqual(repr(self.X), "[X]")


class SpanTestCase(unittest.TestCase):

    def setUp(self):
        self.X = Nonterminal('X')
        self.X01 = Span(Nonterminal('X'), 0, 1)
        self.X01b = Span(Nonterminal('X'), 0, 1)

    def test_inheritance(self):
        self.assertIsInstance(self.X01, Nonterminal)

    def test_construct(self):
        with self.assertRaises(TypeError):
            Span('X', 0, 1)

    def test_underlying(self):
        self.assertEqual(self.X01.underlying, (self.X, 0, 1))

    def test_underlying_type(self):
        self.assertIs(type(self.X01.underlying), tuple)

    def test_base(self):
        self.assertEqual(self.X01.base, self.X)

    def test_start(self):
        self.assertEqual(self.X01.start, 0)

    def test_end(self):
        self.assertEqual(self.X01.end, 1)

    def test_instance_management(self):
        self.assertIsNot(self.X01, self.X01b)

    def test_str(self):
        self.assertEqual(str(self.X01), 'X:0-1')

    def test_repr(self):
        self.assertEqual(repr(self.X01), "[X:0-1]")


class ComparisonTestCase(unittest.TestCase):

    def setUp(self):
        self.tX1 = Terminal('X')
        self.tX2 = Terminal('X')
        self.nX1 = Nonterminal('X')
        self.nX2 = Nonterminal('X')

    def test_type_matters(self):
        self.assertNotEqual(self.tX1, self.nX1, msg='The specific type of the symbol matters to decide for equality.')

    def test_underlying_identity(self):
        self.assertEqual(self.tX1, self.tX2)
        self.assertIsNot(self.tX1, self.tX2)
        self.assertEqual(self.nX1, self.nX2)
        self.assertIsNot(self.nX1, self.nX2)


class StaticSymbolFactoryTestCase(unittest.TestCase):

    def setUp(self):
        self.factory = StaticSymbolFactory
        self.factory2 = StaticSymbolFactory()

    def test_terminal(self):
        self.assertIs(type(self.factory.terminal('x')), Terminal)

    def test_nonterminal(self):
        self.assertIs(type(self.factory.nonterminal('X')), Nonterminal)

    def test_span(self):
        self.assertIs(type(self.factory.span(self.factory.nonterminal('X'), 0, 1)), Span)

    def test_instance_management(self):
        self.assertIs(self.factory.terminal('x'), self.factory.terminal('x'))
        self.assertIsNot(self.factory.terminal('x'), self.factory.nonterminal('x'))
        self.assertIsNot(self.factory.terminal('x'), self.factory.terminal('X'))
        self.assertIs(self.factory.nonterminal('X'), self.factory.nonterminal('X'))
        self.assertIsNot(self.factory.nonterminal('X'), self.factory.nonterminal('Y'))
        self.assertIs(self.factory.span(self.factory.nonterminal('X'), 0, 1),
                      self.factory.span(self.factory.nonterminal('X'), 0, 1))
        self.assertIsNot(self.factory.span(self.factory.nonterminal('X'), 0, 1),
                         self.factory.span(self.factory.nonterminal('X'), 1, 0))

    def test_static(self):
        self.assertIs(self.factory.terminal('x'), self.factory2.terminal('x'))


class SymbolFactoryTestCase(unittest.TestCase):

    def setUp(self):
        self.factory = SymbolFactory()
        self.factory2 = SymbolFactory()

    def test_terminal(self):
        self.assertIs(type(self.factory.terminal('x')), Terminal)

    def test_nonterminal(self):
        self.assertIs(type(self.factory.nonterminal('X')), Nonterminal)

    def test_span(self):
        self.assertIs(type(self.factory.span(self.factory.nonterminal('X'), 0, 1)), Span)

    def test_instance_management(self):
        self.assertIs(self.factory.terminal('x'), self.factory.terminal('x'))
        self.assertIsNot(self.factory.terminal('x'), self.factory.nonterminal('x'))
        self.assertIsNot(self.factory.terminal('x'), self.factory.terminal('X'))
        self.assertIs(self.factory.nonterminal('X'), self.factory.nonterminal('X'))
        self.assertIsNot(self.factory.nonterminal('X'), self.factory.nonterminal('Y'))
        self.assertIs(self.factory.span(self.factory.nonterminal('X'), 0, 1),
                      self.factory.span(self.factory.nonterminal('X'), 0, 1))
        self.assertIsNot(self.factory.span(self.factory.nonterminal('X'), 0, 1),
                         self.factory.span(self.factory.nonterminal('X'), 1, 0))

    def test_nonstatic(self):
        self.assertIsNot(self.factory.terminal('x'), self.factory2.terminal('x'))
        self.assertEqual(self.factory.terminal('x'), self.factory2.terminal('x'))
        self.assertIsNot(self.factory.nonterminal('X'), self.factory2.nonterminal('X'))
        self.assertEqual(self.factory.nonterminal('X'), self.factory2.nonterminal('X'))


if __name__ == '__main__':
    unittest.main()
