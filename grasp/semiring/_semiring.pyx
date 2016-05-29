cimport numpy as np
import numpy as np
from grasp.ptypes cimport weight_t
import grasp.ptypes as ptypes
from grasp.semiring.operator cimport ProbPlus, ProbTimes
from grasp.semiring.operator cimport LogProbPlus, LogProbTimes
from grasp.semiring.operator cimport ViterbiPlus, ViterbiTimes
cimport libc.math as cppmath

cdef class Semiring:

    def __init__(self, Plus plus, Times times, bint LOG):
        self.plus = plus
        self.times = times
        self.one = times.identity
        self.zero = plus.identity
        self.LOG = LOG
        self.idempotent = plus.idempotent

    cpdef weight_t as_real(self, weight_t x): pass

    cpdef weight_t from_real(self, weight_t x): pass

    cpdef bint gt(self, weight_t x, weight_t y): pass

    cpdef weight_t heapify(self, weight_t x): pass

    cpdef weight_t divide(self, weight_t num, weight_t den):
        return self.times.evaluate(num, self.times.inverse.evaluate(den))

    cpdef weight_t power(self, weight_t base, weight_t power):
        return self.times.power.evaluate(base, power)

    cpdef weight_t[::1] zeros(self, size_t size):
        return np.full(size, self.zero, ptypes.weight)

    cpdef weight_t[::1] ones(self, size_t size):
        return np.full(size, self.one, ptypes.weight)

    cpdef weight_t[::1] normalise(self, weight_t[::1] values):
        cdef weight_t v
        cdef weight_t total = self.plus.reduce(values)
        return np.array([self.divide(v, total) for v in values], dtype=ptypes.weight)

    def __repr__(self):
        return '%s(plus=%r, times=%r, LOG=%r)' % (self.__class__.__name__, self.plus, self.times, self.LOG)


cdef class Prob(Semiring):

    def __init__(self):
        super(Prob, self).__init__(ProbPlus(), ProbTimes(), LOG=False)

    cpdef weight_t as_real(self, weight_t x):
        return x

    cpdef weight_t from_real(self, weight_t x):
        return x

    cpdef bint gt(self, weight_t x, weight_t y):
        return x > y

    cpdef weight_t heapify(self, weight_t x):
        return -x


cdef class LogProb(Semiring):

    def __init__(self):
        super(LogProb, self).__init__(LogProbPlus(), LogProbTimes(), LOG=True)

    cpdef weight_t as_real(self, weight_t x):
        return cppmath.exp(x)

    cpdef weight_t from_real(self, weight_t x):
        return cppmath.log(x)

    cpdef bint gt(self, weight_t x, weight_t y):
        return x > y

    cpdef weight_t heapify(self, weight_t x):
        return -x


cdef class Viterbi(Semiring):

    def __init__(self):
        super(Viterbi, self).__init__(ViterbiPlus(), ViterbiTimes(), LOG=True)

    cpdef weight_t as_real(self, weight_t x):
        return cppmath.exp(x)

    cpdef weight_t from_real(self, weight_t x):
        return cppmath.log(x)

    cpdef bint gt(self, weight_t x, weight_t y):
        return x > y

    cpdef weight_t heapify(self, weight_t x):
        return -x