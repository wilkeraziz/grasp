"""
:Authors: - Wilker Aziz
"""

import numpy as np


class Counting(object):
    """
    >>> semi = Count
    >>> semi.one
    1
    >>> semi.zero
    0
    >>> semi.plus(semi.one, semi.zero)  # additive identity
    1
    >>> semi.plus(semi.one, semi.one)  # sum
    2
    >>> semi.times(semi.one, semi.zero)  # multiplicative annihilator
    0
    >>> semi.from_real(0.5)
    0
    >>> semi.from_real(2.5)
    2
    >>> semi.times(semi.one, semi.from_real(2.5))  # multiplicative identity # doctest: +ELLIPSIS
    2
    >>> semi.gt(semi.from_real(0.6), semi.from_real(0.1))
    False
    >>> semi.gt(semi.heapify(semi.from_real(0.6)), semi.heapify(semi.from_real(0.1)))
    False
    >>> semi.as_real(semi.from_real(1.5))
    1.0
    """

    LOG = False

    one = 1
    zero = 0
    plus = np.add
    times = np.multiply
    divide = None
    as_real = float
    from_real = float
    gt = np.greater
    heapify = np.negative

    convert = lambda x, semiring: Counting.one if semiring.gt(x, semiring.zero) else Counting.zero



