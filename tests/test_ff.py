"""
Authors: - Wilker Aziz
"""

import unittest

from grasp.scoring.state import StateMapper
from grasp.scoring.extractor import FValue, FVec, FMap
from grasp.scoring.lookup import RuleTable, CDEC_DEFAULT
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
