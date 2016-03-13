cimport numpy as np
import numpy as np
from grasp.ptypes cimport weight_t
import grasp.ptypes as ptypes


cdef class Operator:

    def __call__(self, weight_t a, weight_t b):
        return self.evaluate(a, b)

    cdef weight_t evaluate(self, weight_t a, weight_t b): pass

    cpdef weight_t reduce(self, iterable) except *: pass

    def __repr__(self):
        return '%s(identity=%s)' % (self.__class__.__name__, self.identity)


cdef class Plus(Operator):

    cpdef int choice(self, weight_t[::1] values) except -1: pass

    def __repr__(self):
        return '%s(identity=%r, idempotent=%r)' % (self.__class__.__name__, self.identity, self.idempotent)


cdef class Times(Operator):

    cpdef weight_t inverse(self, weight_t a): pass


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

    cpdef weight_t divide(self, weight_t x, weight_t y):
        return self.times.evaluate(x, self.times.inverse(y))

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


cdef class ProbTimes(Times):

    def __init__(self):
        self.identity = 1.0

    cdef weight_t evaluate(self, weight_t a, weight_t b):
        return np.multiply(a, b)

    cpdef weight_t reduce(self, sequence) except *:
        return np.multiply.reduce(sequence)

    cpdef weight_t inverse(self, weight_t a):
        return np.divide(1.0, a)


cdef class LogProbTimes(Times):

    def __init__(self):
        self.identity = 0.0

    cdef weight_t evaluate(self, weight_t a, weight_t b):
        return np.add(a, b)

    cpdef weight_t reduce(self, sequence) except *:
        return np.add.reduce(sequence)

    cpdef weight_t inverse(self, weight_t a):
        return np.negative(a)


cdef class ViterbiTimes(Times):

    def __init__(self):
        self.identity = 0.0

    cdef weight_t evaluate(self, weight_t a, weight_t b):
        return np.add(a, b)

    cpdef weight_t reduce(self, sequence) except *:
        return np.add.reduce(sequence)

    cpdef weight_t inverse(self, weight_t a):
        return np.negative(a)


cdef class ProbPlus(Plus):

    def __init__(self):
        self.identity = 0.0
        self.idempotent = False

    cdef weight_t evaluate(self, weight_t a, weight_t b):
        return np.add(a, b)

    cpdef weight_t reduce(self, sequence) except *:
        return np.add.reduce(sequence)

    cpdef int choice(self, weight_t[::1] values) except -1:
        return np.random.choice(values.shape[0], p=values)


cdef class LogProbPlus(Plus):

    def __init__(self):
        self.identity = -np.inf
        self.idempotent = False

    cdef weight_t evaluate(self, weight_t a, weight_t b):
        return np.logaddexp(a, b)

    cpdef weight_t reduce(self, sequence) except *:
        try:
            return np.logaddexp.reduce(sequence)
        except ValueError:
            return self.identity

    cpdef int choice(self, weight_t[::1] values) except -1:
        return np.random.choice(values.shape[0], p=np.exp(values))


cdef class ViterbiPlus(Plus):

    def __init__(self):
        self.identity = -np.inf
        self.idempotent = True

    cdef weight_t evaluate(self, weight_t a, weight_t b):
        return np.max([a, b])

    cpdef weight_t reduce(self, sequence) except *:
        try:
            return np.max(sequence)
        except ValueError:
            return self.identity

    cpdef int choice(self, weight_t[::1] values) except -1:
        return np.argmax(values)


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
        return np.exp(x)

    cpdef weight_t from_real(self, weight_t x):
        return np.log(x)

    cpdef bint gt(self, weight_t x, weight_t y):
        return x > y

    cpdef weight_t heapify(self, weight_t x):
        return -x


cdef class Viterbi(Semiring):

    def __init__(self):
        super(Viterbi, self).__init__(ViterbiPlus(), ViterbiTimes(), LOG=True)

    cpdef weight_t as_real(self, weight_t x):
        return np.exp(x)

    cpdef weight_t from_real(self, weight_t x):
        return np.log(x)

    cpdef bint gt(self, weight_t x, weight_t y):
        return x > y

    cpdef weight_t heapify(self, weight_t x):
        return -x