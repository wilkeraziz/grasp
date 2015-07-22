"""
:Authors: - Wilker Aziz
"""

import numpy as np


class Boolean(object):
    """
    >>> semi = Boolean
    >>> semi.one
    True
    >>> semi.zero
    False
    >>> semi.plus(semi.one, semi.zero)  # additive identity
    True
    >>> semi.plus(semi.one, semi.one)  # sum
    True
    >>> semi.times(semi.one, semi.zero)  # multiplicative annihilator
    False
    >>> semi.from_real(1.0)
    True
    >>> semi.from_real(0.0)
    False
    >>> semi.times(semi.one, semi.from_real(0))  # multiplicative identity # doctest: +ELLIPSIS
    False
    >>> semi.gt(semi.from_real(0), semi.from_real(1))
    False
    >>> semi.gt(semi.heapify(semi.from_real(0)), semi.heapify(semi.from_real(1)))
    True
    >>> semi.as_real(semi.from_real(2))
    1.0
    """

    LOG = False

    one = True
    zero = False
    plus = np.logical_or
    times = np.logical_and
    divide = None
    as_real = float
    from_real = bool
    gt = np.greater
    heapify = np.logical_not

    convert = lambda x, semiring: Boolean.one if semiring.gt(x, semiring.zero) else Boolean.zero
