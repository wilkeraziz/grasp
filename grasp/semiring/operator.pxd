from grasp.ptypes cimport weight_t


cdef class UnaryOperator:

    cdef weight_t evaluate(self, weight_t x)


cdef class BinaryOperator:

    cdef readonly weight_t identity

    cdef weight_t evaluate(self, weight_t x, weight_t y)

    cpdef weight_t reduce(self, sequence) except *


cdef class FixedLHS(UnaryOperator):

    cdef weight_t lhs
    cdef BinaryOperator op


cdef class FixedRHS(UnaryOperator):

    cdef weight_t rhs
    cdef BinaryOperator op


cdef class Plus(BinaryOperator):

    cdef readonly bint idempotent

    cpdef int choice(self, weight_t[::1] values) except -1


cdef class Times(BinaryOperator):

    cdef readonly UnaryOperator inverse
    cdef readonly BinaryOperator power


cdef class ProbPower(BinaryOperator): pass


cdef class ProbInverse(UnaryOperator): pass


cdef class LogProbPower(BinaryOperator): pass


cdef class LogProbInverse(UnaryOperator): pass


cdef class ProbTimes(Times): pass


cdef class LogProbTimes(Times): pass


cdef class ViterbiTimes(Times): pass


cdef class ProbPlus(Plus): pass


cdef class LogProbPlus(Plus): pass


cdef class ViterbiPlus(Plus): pass


# TODO: make unary operators for Heapify, AsReal, FromReal
# TODO: make binary operators for GreaterThan

