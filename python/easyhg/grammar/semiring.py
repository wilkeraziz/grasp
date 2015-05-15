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

class SumTimes(object):

    one = 0.0
    zero = -np.inf
    plus = np.logaddexp
    times = np.add
    divide = np.subtract
    as_real = np.exp
    from_real = np.log
    gt = np.greater

class MaxTimes(object):

    one = 0.0
    zero = -np.inf
    plus = max
    times = np.add
    divide = np.subtract
    as_real = np.exp
    from_real = np.log
    gt = np.greater

class Count(object):

    one = 1.0
    zero = 0.0
    plus = np.add
    times = np.multiply
    divide = None
    as_real = float
    from_real = lambda x: int(bool(x))
