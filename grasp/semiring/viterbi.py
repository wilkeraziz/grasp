"""
These are Viterbi-like semirings.
I prefer calling it MaxTimes to avoid confusion.

:Authors: - Wilker Aziz
"""

import numpy as np


class MaxTimes(object):
    """
    >>> semi = MaxTimes
    >>> semi.one
    0.0
    >>> semi.zero
    -inf
    >>> semi.plus(semi.one, semi.zero)  # additive identity
    0.0
    >>> semi.plus(semi.one, semi.one)  # max
    0.0
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
    IDEMPOTENT = True

    one = 0.0
    zero = -np.inf
    plus = max
    times = np.add
    divide = np.subtract
    as_real = np.exp
    from_real = np.log
    gt = np.greater
    heapify = np.negative
    choice = lambda items, values: max(zip(items, values), key=lambda pair: pair[1])[0]
