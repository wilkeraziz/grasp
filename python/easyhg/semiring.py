from numpy import log
from nympy import exp


class MaxTimes(object):

    @staticmethod
    def one():
        return 0

    @staticmethod
    def zero():
        return float('-inf')

    @staticmethod
    def plus(a, b):
        return max(a, b)

    @staticmethod
    def times(a, b):
        return a + b

    @staticmethod
    def as_real(a):
        return exp(a)

    @staticmethod
    def construct(a):
        return log(a)

class SumTimes(object):

    @staticmethod
    def one():
        return 0

    @staticmethod
    def zero():
        return float('-inf')

    @staticmethod
    def plus(a, b):
        return log(exp(a) + exp(b))

    @staticmethod
    def times(a, b):
        return a + b

    @staticmethod
    def as_real(a):
        return exp(a)

    @staticmethod
    def construct(a):
        return log(a)
