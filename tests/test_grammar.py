import unittest
import doctest
from easyhg.grammar import symbol
from easyhg.grammar import rule
from easyhg.grammar import topsort
from easyhg.grammar import cfg
from easyhg.grammar import fsa
from easyhg.grammar import semiring
from easyhg.grammar import dottedrule
from easyhg.grammar import agenda
from easyhg.grammar import inference
from easyhg.grammar import kbest
from easyhg.grammar import ply_cfg
from easyhg.grammar import sentence
from easyhg.grammar import earley
from easyhg.grammar import nederhof
#from easyhg.grammar import cky
from easyhg.grammar import slicednederhof
from easyhg.grammar import slicevars


def load_tests(loader, tests, ignore):
    # Grammar
    tests.addTests(doctest.DocTestSuite(symbol))
    tests.addTests(doctest.DocTestSuite(rule))
    tests.addTests(doctest.DocTestSuite(topsort))
    tests.addTests(doctest.DocTestSuite(cfg))
    # FSA
    tests.addTests(doctest.DocTestSuite(fsa))
    # Input/Output
    tests.addTests(doctest.DocTestSuite(ply_cfg))
    tests.addTests(doctest.DocTestSuite(sentence))
    # inference algorithms
    tests.addTests(doctest.DocTestSuite(semiring))
    tests.addTests(doctest.DocTestSuite(inference))  # TODO: write tests
    tests.addTests(doctest.DocTestSuite(kbest))  # TODO: write tests
    # intersection algorithms
    tests.addTests(doctest.DocTestSuite(dottedrule))
    tests.addTests(doctest.DocTestSuite(agenda))
    tests.addTests(doctest.DocTestSuite(nederhof))
    tests.addTests(doctest.DocTestSuite(earley))
    tests.addTests(doctest.DocTestSuite(slicednederhof))
    # sampling algorithms
    tests.addTests(doctest.DocTestSuite(slicevars))
    return tests


if __name__ == '__main__':
    unittest.main()
