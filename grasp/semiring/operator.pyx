cimport numpy as np
import numpy as np
from grasp.ptypes cimport weight_t
import grasp.ptypes as ptypes


cdef class UnaryOperator:

    cdef weight_t evaluate(self, weight_t x): pass

    def __call__(self, weight_t x):
        return self.evaluate(x)

    def __repr__(self):
        return '%s()' % (self.__class__.__name__)


cdef class BinaryOperator:

    def __init__(self, weight_t identity):
        self.identity = identity

    def __call__(self, weight_t x, weight_t y):
        return self.evaluate(x, y)

    cdef weight_t evaluate(self, weight_t x, weight_t y): pass

    cpdef weight_t reduce(self, iterable) except *: pass

    def __repr__(self):
        return '%s(identity=%s)' % (self.__class__.__name__, self.identity)


cdef class FixedLHS(UnaryOperator):

    def __init__(self, weight_t lhs, BinaryOperator op):
        self.lhs = lhs
        self.op = op

    cpdef weight_t evaluate(self, weight_t rhs):
        return self.op.evaluate(self.lhs, rhs)


cdef class FixedRHS(UnaryOperator):

    def __init__(self, weight_t rhs, BinaryOperator op):
        self.lhs = rhs
        self.op = op

    cpdef weight_t evaluate(self, weight_t lhs):
        return self.op.evaluate(lhs, self.rhs)


cdef class Plus(BinaryOperator):

    def __init__(self, weight_t identity, bint idempotent):
        super(Plus, self).__init__(identity)
        self.idempotent = idempotent

    cpdef int choice(self, weight_t[::1] values) except -1: pass

    def __repr__(self):
        return '%s(identity=%r, idempotent=%r)' % (self.__class__.__name__, self.identity, self.idempotent)


cdef class Times(BinaryOperator):

    def __init__(self, weight_t identity,  UnaryOperator inverse, BinaryOperator power):
        super(Times, self).__init__(identity)
        self.inverse = inverse
        self.power = power



cdef class ProbPower(BinaryOperator):

    def __init__(self):
        super(ProbPower, self).__init__(1.0)

    cpdef weight_t  evaluate(self, weight_t base, weight_t power):
        return np.power(base, power)

    cpdef weight_t reduce(self, sequence) except *:
        return np.power.reduce(sequence)


cdef class LogProbPower(BinaryOperator):

    def __init__(self):
        super(LogProbPower, self).__init__(1.0)

    cpdef weight_t evaluate(self, weight_t base, weight_t power):
        return power * base

    cpdef weight_t reduce(self, sequence) except *:
        return np.multiply.reduce(sequence)


cdef class ProbInverse(UnaryOperator):

    cpdef weight_t evaluate(self, weight_t value):
        return 1.0 / value


cdef class LogProbInverse(UnaryOperator):

    cpdef weight_t evaluate(self, weight_t value):
        return -value


cdef class ProbTimes(Times):

    def __init__(self):
        super(ProbTimes, self).__init__(1.0, ProbInverse(), ProbPower())

    cdef weight_t evaluate(self, weight_t a, weight_t b):
        return np.multiply(a, b)

    cpdef weight_t reduce(self, sequence) except *:
        return np.multiply.reduce(sequence)


cdef class LogProbTimes(Times):

    def __init__(self):
        super(LogProbTimes, self).__init__(0.0, LogProbInverse(), LogProbPower())

    cdef weight_t evaluate(self, weight_t a, weight_t b):
        return np.add(a, b)

    cpdef weight_t reduce(self, sequence) except *:
        return np.add.reduce(sequence)


cdef class ViterbiTimes(Times):

    def __init__(self):
        super(ViterbiTimes, self).__init__(0.0, LogProbInverse(), LogProbPower())

    cdef weight_t evaluate(self, weight_t a, weight_t b):
        return np.add(a, b)

    cpdef weight_t reduce(self, sequence) except *:
        return np.add.reduce(sequence)


cdef class ProbPlus(Plus):

    def __init__(self):
        super(ProbPlus, self).__init__(0.0, False)

    cdef weight_t evaluate(self, weight_t a, weight_t b):
        return np.add(a, b)

    cpdef weight_t reduce(self, sequence) except *:
        return np.add.reduce(sequence)

    cpdef int choice(self, weight_t[::1] values) except -1:
        cdef:
            weight_t threshold = np.random.uniform()
            weight_t acc = 0
            weight_t p
            size_t i = 0
        for p in values:
            acc += p
            if acc > threshold:
                return i
            i += 1
        return i - 1  # last


cdef class LogProbPlus(Plus):

    def __init__(self):
        super(LogProbPlus, self).__init__(-np.inf, False)

    cdef weight_t evaluate(self, weight_t a, weight_t b):
        return np.logaddexp(a, b)

    cpdef weight_t reduce(self, sequence) except *:
        try:
            return np.logaddexp.reduce(sequence)
        except ValueError:
            return self.identity

    cpdef int choice(self, weight_t[::1] values) except -1:
        cdef:
            weight_t threshold = np.random.uniform()
            weight_t[::1] probs = np.exp(values)
            weight_t acc = 0
            weight_t p
            size_t i = 0
        for p in probs:
            acc += p
            if acc > threshold:
                return i
            i += 1
        return i - 1  # last


cdef class ViterbiPlus(Plus):

    def __init__(self):
        super(ViterbiPlus, self).__init__(-np.inf, True)

    cdef weight_t evaluate(self, weight_t a, weight_t b):
        return np.max([a, b])

    cpdef weight_t reduce(self, sequence) except *:
        try:
            return np.max(sequence)
        except ValueError:
            return self.identity

    cpdef int choice(self, weight_t[::1] values) except -1:
        return np.argmax(values)

