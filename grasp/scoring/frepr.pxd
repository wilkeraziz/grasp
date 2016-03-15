"""
:Authors: - Wilker Aziz
"""
from grasp.ptypes cimport weight_t
from grasp.semiring.operator cimport BinaryOperator
from grasp.semiring.operator cimport UnaryOperator
from grasp.semiring._semiring cimport Semiring


cdef class FRepr:

    cpdef FRepr prod(self, weight_t scalar)

    cpdef weight_t dot(self, FRepr w) except *

    cpdef FRepr hadamard(self, FRepr rhs, BinaryOperator op)

    cpdef FRepr elementwise(self, UnaryOperator op)

    cpdef FRepr elementwise_b(self, weight_t rhs, BinaryOperator op)

    cpdef FRepr densify(self)

    cpdef FRepr power(self, weight_t power, Semiring semiring)


cdef class FValue(FRepr):

    cdef readonly weight_t value


cdef class FVec(FRepr):

    cdef readonly weight_t[::1] vec


cdef class FMap(FRepr):

    cdef readonly dict map


cdef class FComponents(FRepr):

    cdef readonly list components

    cpdef FComponents concatenate(self, FComponents other)