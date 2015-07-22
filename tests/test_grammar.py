import unittest
import doctest
from grasp.cfg import symbol
from grasp.cfg import rule
from grasp.cfg import topsort
from grasp.cfg import cfg
from grasp.cfg import fsa
from grasp.cfg import semiring
from grasp.cfg import dottedrule
from grasp.cfg import agenda
from grasp.cfg import inference
from grasp.cfg import kbest
from grasp.cfg import ply_cfg
from grasp.cfg import sentence
from grasp.cfg import earley
from grasp.cfg import nederhof
#from easyhg.grammar import cky
from grasp.cfg import slicednederhof
from grasp.cfg import slicevars


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
