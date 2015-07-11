"""
This module specifies several semirings. 

A semiring must define the following 

    one => the multiplicative identity
    zero => the additive identity (and multiplicative annihilator)
    plus => addition
    times => multiplication

A semiring may define the following 

    divide => division
    as_real => return a Real number
    from_real => constructs from a Real number
    gt => comparison '>'
    heapify => return a value compatible with the logic of a heap (smaller first)

"""
__author__ = 'wilkeraziz'

import numpy as np
import operator


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

    one = 1.0
    zero = 0.0
    plus = np.add
    times = np.multiply
    divide = np.divide
    as_real = float
    from_real = float
    gt = np.greater
    heapify = operator.neg 


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

    one = 0.0
    zero = -np.inf
    plus = np.logaddexp
    times = np.add
    divide = np.subtract
    as_real = np.exp
    from_real = np.log
    gt = np.greater
    heapify = operator.neg 


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

    one = 0.0
    zero = -np.inf
    plus = max
    times = np.add
    divide = np.subtract
    as_real = np.exp
    from_real = np.log
    gt = np.greater
    heapify = operator.neg


class Count(object):
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
    plus = operator.add  # np.add
    times = operator.mul  # np.multiply
    divide = None
    as_real = float
    from_real = int
    gt = np.greater
    heapify = operator.neg

    convert = lambda x, semiring: Count.one if semiring.gt(x, semiring.zero) else Count.zero



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
    plus = operator.or_  # np.add
    times = operator.and_  # np.multiply
    divide = None
    as_real = float
    from_real = bool
    gt = np.greater
    heapify = operator.neg

