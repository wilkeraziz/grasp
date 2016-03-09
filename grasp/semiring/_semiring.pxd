from grasp.ptypes cimport weight_t


cdef class Operator:

    cdef readonly weight_t identity

    cdef weight_t evaluate(self, weight_t a, weight_t b)

    cpdef weight_t reduce(self, sequence) except *


cdef class Plus(Operator):

    cdef readonly bint idempotent

    cpdef int choice(self, weight_t[::1] values) except -1


cdef class Times(Operator):

    cpdef weight_t inverse(self, weight_t a)


cdef class Semiring:

    cdef readonly bint LOG
    cdef readonly bint idempotent
    cdef readonly weight_t one
    cdef readonly weight_t zero
    cdef readonly Plus plus
    cdef readonly Times times

    cpdef weight_t as_real(self, weight_t x)

    cpdef weight_t from_real(self, weight_t x)

    cpdef bint gt(self, weight_t x, weight_t y)

    cpdef weight_t heapify(self, weight_t x)

    cpdef weight_t divide(self, weight_t x, weight_t y)

    cpdef weight_t[::1] zeros(self, size_t size)

    cpdef weight_t[::1] ones(self, size_t size)

    cpdef weight_t[::1] normalise(self, weight_t[::1] values)


cdef class ProbTimes(Times): pass

cdef class LogProbTimes(Times): pass

cdef class ViterbiTimes(Times): pass

cdef class ProbPlus(Plus): pass

cdef class LogProbPlus(Plus): pass

cdef class ViterbiPlus(Plus): pass

cdef class Prob(Semiring): pass

cdef class LogProb(Semiring): pass

cdef class Viterbi(Semiring): pass

