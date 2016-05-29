"""
Weight function over edges and derivations.

:Authors: - Wilker Aziz
"""

from grasp.ptypes cimport weight_t, id_t, boolean_t
from grasp.formal.hg cimport Hypergraph
from grasp.semiring._semiring cimport Semiring
from grasp.semiring.operator cimport BinaryOperator
cimport numpy as np


cdef class WeightFunction:

    cpdef weight_t value(self, id_t e)

    cpdef weight_t reduce(self, BinaryOperator op, iterable)


cpdef weight_t derivation_weight(Hypergraph forest, tuple edges, Semiring semiring, WeightFunction omega=?)


cdef class ConstantFunction(WeightFunction):

    cdef weight_t constant


cdef class ReducedFunction(WeightFunction):

    cdef tuple functions
    cdef BinaryOperator op


cdef class TableLookupFunction(WeightFunction):

    cdef weight_t[::1] table


cdef class BooleanFunction(WeightFunction):

    cdef boolean_t[::1] table
    cdef weight_t one
    cdef weight_t zero


cdef class HypergraphLookupFunction(WeightFunction):

    cdef Hypergraph hg


cdef class ScaledFunction(WeightFunction):

    cdef WeightFunction func
    cdef weight_t scalar


cdef class ThresholdFunction(WeightFunction):

    cdef:
        WeightFunction func
        Semiring input_semiring
        WeightFunction thresholdfunc
        Semiring output_semiring
