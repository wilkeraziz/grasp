"""
Authors: - Wilker Aziz
"""

import unittest

from grasp.scoring.state import StateMapper
from grasp.scoring.frepr import FValue, FVec, FMap, FCSR
from grasp.scoring.lookup import RuleTable, CDEC_DEFAULT
from grasp.semiring.operator import ProbPlus, ProbTimes
import numpy as np
import grasp.ptypes as ptypes


class StateTestCase(unittest.TestCase):

    def test_hashable(self):
        mapper = StateMapper()
        with self.assertRaises(TypeError):
            mapper.id([1,2,3])

    def test_state(self):
        mapper = StateMapper()
        with self.assertRaises(IndexError):
            mapper.state(1)

    def test_final(self):
        mapper = StateMapper()
        self.assertEqual(len(mapper), 1)
        self.assertEqual(mapper.final, 0)

    def test_map(self):
        mapper = StateMapper()
        a = mapper.id((1,))
        b = mapper.id((1, 2))
        c = mapper.id((1, 2, 3))
        self.assertEqual(len(mapper), 4)
        self.assertEqual(mapper.state(a), (1,))
        self.assertEqual(mapper.state(b), (1, 2))
        self.assertEqual(mapper.state(c), (1, 2, 3))


class FReprTestCase(unittest.TestCase):

    def test_value(self):
        x = FValue(10)
        w= FValue(0.5)
        self.assertEqual(x.value, 10.0)
        self.assertEqual(x.dot(w), 5.0)
        with self.assertRaises(TypeError):
            x.dot(10)

    def test_vec(self):
        x = FVec([1.0, 2.0, 3.0])
        w = FVec([0.5, 1.0, 2.0])
        self.assertSequenceEqual(list(x.vec), [1.0, 2.0, 3.0])
        self.assertSequenceEqual(list(x), [1.0, 2.0, 3.0])
        self.assertEqual(x.dot(w), 8.5)
        with self.assertRaises(TypeError):
            x.dot(FValue(10))

    def test_map(self):
        d = {'a': 1.0, 'b': 2.0, 'c': 3.0}
        x = FMap(d)
        w = FMap([('a', 0.5), ('b', 1.0), ('c', 2.0), ('d', 3.0)])
        self.assertDictEqual(x.map, d)
        self.assertDictEqual(dict(x), d)
        self.assertEqual(x.dot(w), 8.5)
        with self.assertRaises(TypeError):
            x.dot(FValue(10))

    def test_csr(self):
        csr1 = FCSR.construct([1, 2, 3], [0, 2, 5], 10)
        self.assertEqual(len(csr1), 3)
        self.assertSequenceEqual(list(csr1.densify()), [1, 0, 2, 0, 0, 3, 0, 0, 0, 0])
        self.assertSequenceEqual(list(iter(csr1)), [(0, 1), (2, 2), (5, 3)])
        csr2 = csr1.prod(2)
        self.assertEqual(len(csr2), 3)
        self.assertSequenceEqual(list(csr2.densify()), [2, 0, 4, 0, 0, 6, 0, 0, 0, 0])
        self.assertSequenceEqual(list(iter(csr2)), [(0, 2), (2, 4), (5, 6)])
        self.assertEqual(csr1.dot(csr2), 28)
        csr3 = FCSR.construct([10, 20], [0, 1], 10)
        self.assertEqual(csr1.dot(csr3), 10)
        csr4 = csr1.hadamard(csr3, ProbPlus())
        self.assertSequenceEqual(list(iter(csr4)), [(0, 11), (1, 20), (2, 2), (5, 3)])
        csr5 = csr1.hadamard(csr3, ProbTimes())
        self.assertSequenceEqual(list(iter(csr5)), [(0, 10)])



class LookupTableTestCase(unittest.TestCase):

    def test_weights(self):
        rt = RuleTable(0, 'RuleTable')
        with self.assertRaises(KeyError):
            rt.weights({})
        wmap = {}
        for i, f in enumerate(CDEC_DEFAULT):
            wmap[f] = i
        self.assertSequenceEqual(list(rt.weights(wmap)), range(9))



if __name__ == '__main__':
    unittest.main()
