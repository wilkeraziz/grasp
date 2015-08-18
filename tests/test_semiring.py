"""
:Authors: - Wilker Aziz
"""

import unittest
import grasp.semiring as semiring
import numpy as np
from collections import Counter


class ProbTestCase(unittest.TestCase):

    def setUp(self):
        self.semiring = semiring.prob

    def test_attributes(self):
        self.assertFalse(self.semiring.idempotent)
        self.assertFalse(self.semiring.LOG)

    def test_identities(self):
        self.assertEqual(self.semiring.one, 1.0)
        self.assertEqual(self.semiring.zero, 0.0)
        self.assertSequenceEqual(self.semiring.zeros(10), [0.0] * 10)
        self.assertSequenceEqual(self.semiring.ones(10), [1.0] * 10)

    def test_conversions(self):
        self.assertEqual(self.semiring.as_real(0.5), 0.5)
        self.assertEqual(self.semiring.from_real(0.5), 0.5)

    def test_times(self):
        self.assertEqual(self.semiring.times.identity, self.semiring.one)
        self.assertEqual(self.semiring.times(self.semiring.one, self.semiring.one), self.semiring.one)
        self.assertEqual(self.semiring.times(self.semiring.one, self.semiring.zero), self.semiring.zero)
        self.assertAlmostEqual(self.semiring.times(0.2, 0.4), 0.08)
        self.assertAlmostEqual(self.semiring.times.reduce([0.1, 0.5, 0.2]), 0.01)
        self.assertEqual(self.semiring.times.reduce([]), self.semiring.one)
        self.assertAlmostEqual(self.semiring.divide(0.5, 0.2), 2.5)

    def test_plus(self):
        self.assertEqual(self.semiring.plus.identity, self.semiring.zero)
        self.assertEqual(self.semiring.plus(self.semiring.one, self.semiring.zero), self.semiring.one)
        self.assertEqual(self.semiring.plus(self.semiring.zero, self.semiring.zero), self.semiring.zero)
        self.assertEqual(self.semiring.plus(self.semiring.one, self.semiring.one), 2.0)
        self.assertAlmostEqual(self.semiring.plus(0.2, 0.4), 0.6)
        self.assertAlmostEqual(self.semiring.plus.reduce([0.1, 0.5, 0.2]), 0.8)
        self.assertEqual(self.semiring.plus.reduce([]), self.semiring.zero)

    def test_choice(self):

        with self.assertRaises(ValueError):
            self.semiring.plus.choice(np.array([0.1, 0.2, 0.3, 0.6]))
        c = Counter(self.semiring.plus.choice(np.array([0.1, 0.6, 0.3])) for i in range(1000))
        self.assertSequenceEqual([i for i, n in c.most_common()], [1, 2, 0])


class InsideTestCase(unittest.TestCase):

    def setUp(self):
        self.semiring = semiring.inside

    def test_attributes(self):
        self.assertFalse(self.semiring.idempotent)
        self.assertTrue(self.semiring.LOG)

    def test_identities(self):
        self.assertEqual(self.semiring.one, 0.0)
        self.assertEqual(self.semiring.zero, -float('inf'))
        self.assertSequenceEqual(self.semiring.zeros(10), [-float('inf')] * 10)
        self.assertSequenceEqual(self.semiring.ones(10), [0.0] * 10)

    def test_conversions(self):
        self.assertEqual(self.semiring.as_real(-0.69314718055994529), 0.5)
        self.assertEqual(self.semiring.from_real(0.5), -0.69314718055994529)
        self.assertEqual(self.semiring.as_real(0.69314718055994529), 2.0)
        self.assertEqual(self.semiring.from_real(2.0), 0.69314718055994529)

    def test_invariants(self):
        self.assertEqual(self.semiring.times.identity, self.semiring.one)
        self.assertEqual(self.semiring.times(self.semiring.one, self.semiring.one), self.semiring.one)
        self.assertEqual(self.semiring.times(self.semiring.one, self.semiring.zero), self.semiring.zero)
        self.assertEqual(self.semiring.times(self.semiring.zero, self.semiring.zero), self.semiring.zero)
        self.assertEqual(self.semiring.plus.identity, self.semiring.zero)
        self.assertEqual(self.semiring.plus(self.semiring.one, self.semiring.zero), self.semiring.one)
        self.assertEqual(self.semiring.plus(self.semiring.zero, self.semiring.zero), self.semiring.zero)

    def test_times(self):
        self.assertAlmostEqual(self.semiring.times(1, 2), 3)
        self.assertAlmostEqual(self.semiring.times.reduce([1, 2, 3]), 6)
        self.assertEqual(self.semiring.times.reduce([]), self.semiring.one)
        self.assertAlmostEqual(self.semiring.divide(2, 1), 1)

    def test_plus(self):
        self.assertEqual(self.semiring.plus(self.semiring.one, self.semiring.one), 0.69314718055994529)
        self.assertAlmostEqual(self.semiring.plus.reduce([-0.69314718055994529, 0.0]), 0.40546510810816438)
        self.assertEqual(self.semiring.plus.reduce([]), self.semiring.zero)

    def test_choice(self):

        with self.assertRaises(ValueError):
            self.semiring.plus.choice(np.array([self.semiring.from_real(0.1),
                                                self.semiring.from_real(0.2),
                                                self.semiring.from_real(0.3),
                                                self.semiring.from_real(0.6)]))
        c = Counter(self.semiring.plus.choice(np.array([self.semiring.from_real(0.1),
                                                        self.semiring.from_real(0.6),
                                                        self.semiring.from_real(0.3)])) for i in range(1000))
        self.assertSequenceEqual([i for i, n in c.most_common()], [1, 2, 0])


class ViterbiTestCase(unittest.TestCase):

    def setUp(self):
        self.semiring = semiring.viterbi

    def test_attributes(self):
        self.assertTrue(self.semiring.idempotent)
        self.assertTrue(self.semiring.LOG)

    def test_identities(self):
        self.assertEqual(self.semiring.one, 0.0)
        self.assertEqual(self.semiring.zero, -float('inf'))
        self.assertSequenceEqual(self.semiring.zeros(10), [-float('inf')] * 10)
        self.assertSequenceEqual(self.semiring.ones(10), [0.0] * 10)

    def test_conversions(self):
        self.assertEqual(self.semiring.as_real(-0.69314718055994529), 0.5)
        self.assertEqual(self.semiring.from_real(0.5), -0.69314718055994529)
        self.assertEqual(self.semiring.as_real(0.69314718055994529), 2.0)
        self.assertEqual(self.semiring.from_real(2.0), 0.69314718055994529)

    def test_invariants(self):
        self.assertEqual(self.semiring.times.identity, self.semiring.one)
        self.assertEqual(self.semiring.times(self.semiring.one, self.semiring.one), self.semiring.one)
        self.assertEqual(self.semiring.times(self.semiring.one, self.semiring.zero), self.semiring.zero)
        self.assertEqual(self.semiring.times(self.semiring.zero, self.semiring.zero), self.semiring.zero)
        self.assertEqual(self.semiring.plus.identity, self.semiring.zero)
        self.assertEqual(self.semiring.plus(self.semiring.one, self.semiring.zero), self.semiring.one)
        self.assertEqual(self.semiring.plus(self.semiring.zero, self.semiring.zero), self.semiring.zero)

    def test_times(self):
        self.assertAlmostEqual(self.semiring.times(1, 2), 3)
        self.assertAlmostEqual(self.semiring.times.reduce([1, 2, 3]), 6)
        self.assertEqual(self.semiring.times.reduce([]), self.semiring.one)
        self.assertAlmostEqual(self.semiring.divide(2, 1), 1)

    def test_plus(self):
        self.assertEqual(self.semiring.plus(self.semiring.one, self.semiring.one), self.semiring.one)
        self.assertAlmostEqual(self.semiring.plus.reduce([-0.69314718055994529, 0.0]), 0.0)
        self.assertEqual(self.semiring.plus.reduce([]), self.semiring.zero)

    def test_choice(self):
        c = Counter(self.semiring.plus.choice(np.array([self.semiring.from_real(0.1),
                                                        self.semiring.from_real(0.9),
                                                        self.semiring.from_real(0.3)])) for i in range(1000))
        self.assertSequenceEqual(c.most_common(), [(1, 1000)])


if __name__ == '__main__':
    unittest.main()
