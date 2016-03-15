from grasp.ptypes cimport weight_t
from grasp.semiring.operator cimport Plus, Times


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

    cpdef weight_t divide(self, weight_t num, weight_t den)

    cpdef weight_t power(self, weight_t base, weight_t power)

    cpdef weight_t[::1] zeros(self, size_t size)

    cpdef weight_t[::1] ones(self, size_t size)

    cpdef weight_t[::1] normalise(self, weight_t[::1] values)


cdef class Prob(Semiring): pass


cdef class LogProb(Semiring): pass


cdef class Viterbi(Semiring): pass

