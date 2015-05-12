import numpy as np


class Prob(object):

    one = 1.0
    zero = 0
    plus = np.add
    times = np.multiply
    as_real = float
    from_real = float

class SumTimes(object):

    one = 0.0
    zero = -np.inf
    plus = np.logaddexp
    times = np.add
    as_real = np.exp
    from_real = np.log

class MaxTimes(object):

    one = 0.0
    zero = -np.inf
    plus = max
    times = np.add
    as_real = np.exp
    from_real = np.log

