import unittest
import doctest
from . import symbol
from . import rule
from . import topsort
from . import cfg
from . import fsa
from . import semiring
from . import dottedrule
from . import agenda
from . import inference
from . import kbest
from . import ply_cfg
from . import sentence
from . import earley
from . import nederhof
#from . import cky
from . import slicednederhof
from . import slicevars


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
