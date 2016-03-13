"""
This implements the value recursion for numerical semirings.

    V(v) = \bigoplus_{e \in BS(v)} \omega(e) \bigotimes_{u \in tail(e)} V(u)

We also have an implementation which is robust to the presence of cycles.

:Authors: - Wilker Aziz
"""

from grasp.ptypes cimport weight_t, id_t
from grasp.formal.hg cimport Hypergraph
from grasp.formal.topsort cimport AcyclicTopSortTable, RobustTopSortTable
from grasp.semiring._semiring cimport Semiring, Operator


cdef class ValueFunction:

    cpdef weight_t value(self, id_t e)

    cpdef weight_t reduce(self, Operator op, iterable)


cdef class ConstantFunction(ValueFunction):

    cdef weight_t constant


cdef class CascadeValueFunction(ValueFunction):

    cdef tuple functions
    cdef Operator op


cdef class LookupFunction(ValueFunction):

    cdef weight_t[::1] table


cdef class EdgeWeight(ValueFunction):

    cdef Hypergraph hg


cdef class ScaledEdgeWeight(ValueFunction):

    cdef Hypergraph hg
    cdef weight_t scalar


cdef class ScaledValue(ValueFunction):

    cdef ValueFunction func
    cdef weight_t scalar


cdef class ThresholdValueFunction(ValueFunction):

    cdef:
        ValueFunction f
        Semiring input_semiring
        weight_t threshold
        Semiring output_semiring


cdef class BinaryEdgeWeight(ValueFunction):

    cdef Hypergraph hg
    cdef Semiring input_semiring
    cdef Semiring output_semiring


cpdef weight_t derivation_value(Hypergraph forest, tuple edges, Semiring semiring, ValueFunction omega=?)


cdef weight_t node_value(Hypergraph forest,
                         ValueFunction omega,
                         Semiring semiring,
                         weight_t[::1] values,
                         id_t parent)


cpdef weight_t[::1] acyclic_value_recursion(Hypergraph forest,
                                            AcyclicTopSortTable tsort,
                                            Semiring semiring,
                                            ValueFunction omega=?)

cpdef weight_t[::1] acyclic_reversed_value_recursion(Hypergraph forest,
                                            AcyclicTopSortTable tsort,
                                            Semiring semiring,
                                            weight_t[::1] values,
                                            ValueFunction omega=?)

cpdef weight_t[::1] robust_value_recursion(Hypergraph forest,
                                           RobustTopSortTable tsort,
                                           Semiring semiring,
                                           ValueFunction omega=?)

cdef weight_t[::1] approximate_supremum(Hypergraph forest,
                                        ValueFunction omega,
                                        Semiring semiring,
                                        weight_t[::1] values,
                                        list bucket)

cpdef weight_t[::1] compute_edge_values(Hypergraph forest,
                                        Semiring semiring,
                                        weight_t[::1] node_values,
                                        ValueFunction omega=?,
                                        bint normalise=?)

cpdef weight_t[::1] compute_edge_expectation(Hypergraph forest,
                                        Semiring semiring,
                                        weight_t[::1] node_values,
                                        weight_t[::1] node_reversed_values,
                                        ValueFunction omega=?,
                                        bint normalise=?)

cdef class EdgeValues(ValueFunction):

    cdef Hypergraph _forest
    cdef Semiring _semiring
    cdef weight_t[::1] _node_values
    cdef weight_t[::1] _edge_values
    cdef ValueFunction _omega
    cdef bint _normalise


cdef class LazyEdgeValues(ValueFunction):

    cdef Hypergraph _forest
    cdef Semiring _semiring
    cdef weight_t[::1] _node_values
    cdef list _edge_values
    cdef ValueFunction _omega
    cdef bint _normalise

    cdef weight_t _unnormalised(self, id_t e)

    cdef _normalised(self, id_t e)
