"""
These are Inside-style semirings.

We have SumTimes (for log probabilities) and Prob (for "normal" probabilities).

:Authors: - Wilker Aziz
"""

import numpy as np


class SumTimes(object):
    """
    >>> semi = SumTimes
    >>> semi.one
    0.0
    >>> semi.zero
    -inf
    >>> semi.plus(semi.one, semi.zero)  # additive identity
    0.0
    >>> semi.plus(semi.one, semi.one)  # doctest: +ELLIPSIS
    0.6931...
    >>> semi.times(semi.one, semi.zero)  # multiplicative annihilator
    -inf
    >>> semi.times(semi.one, semi.from_real(0.5))  # multiplicative identity # doctest: +ELLIPSIS
    -0.6931...
    >>> semi.divide(semi.from_real(0.5), semi.from_real(0.2))  # doctest: +ELLIPSIS
    0.9162...
    >>> semi.gt(semi.from_real(0.6), semi.from_real(0.1))
    True
    >>> semi.gt(semi.heapify(semi.from_real(0.6)), semi.heapify(semi.from_real(0.1)))
    False
    >>> semi.as_real(semi.from_real(0.5))
    0.5
    """

    LOG = True
    IDEMPOTENT = False

    one = 0.0
    zero = -np.inf
    plus = np.logaddexp
    times = np.add
    divide = np.subtract
    as_real = np.exp
    from_real = np.log
    gt = np.greater
    heapify = np.negative
    choice = lambda items, values: items[np.random.choice(len(items), p=[SumTimes.as_real(v) for v in values])]


class Prob(object):
    """
    >>> semi = Prob
    >>> semi.one
    1.0
    >>> semi.zero
    0.0
    >>> semi.plus(semi.one, semi.zero)  # additive identity
    1.0
    >>> semi.plus(semi.one, semi.one)
    2.0
    >>> semi.times(semi.one, semi.zero)  # multiplicative annihilator
    0.0
    >>> semi.times(semi.one, semi.from_real(0.5))  # multiplicative identity
    0.5
    >>> semi.divide(semi.from_real(0.5), semi.from_real(0.2))
    2.5
    >>> semi.gt(semi.from_real(0.6), semi.from_real(0.1))
    True
    >>> semi.gt(semi.heapify(semi.from_real(0.6)), semi.heapify(semi.from_real(0.1)))
    False
    >>> semi.as_real(semi.from_real(0.5))
    0.5
    """

    LOG = False
    IDEMPOTENT = False

    one = 1.0
    zero = 0.0
    plus = np.add
    times = np.multiply
    divide = np.divide
    as_real = float
    from_real = float
    gt = np.greater
    heapify = np.negative
    choice = lambda items, values: items[np.random.choice(len(items), p=[SumTimes.as_real(v) for v in values])]