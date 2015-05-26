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

    one = 1.0
    zero = 0
    plus = np.add
    times = np.multiply
    divide = np.divide
    as_real = float
    from_real = float
    gt = np.greater
    heapify = operator.neg 

class SumTimes(object):

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

    one = 1L
    zero = 0L
    plus = operator.add  # np.add
    times = operator.mul  # np.multiply
    divide = None
    as_real = float
    from_real = lambda x: long(bool(x))
    heapify = operator.neg 

