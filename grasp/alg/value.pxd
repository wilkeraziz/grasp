"""
This implements the value recursion for numerical semirings.

    V(v) = \bigoplus_{e \in BS(v)} \omega(e) \bigotimes_{u \in tail(e)} V(u)

We also have an implementation which is robust to the presence of cycles.

:Authors: - Wilker Aziz
"""

from grasp.ptypes cimport weight_t, id_t
from grasp.formal.hg cimport Hypergraph
from grasp.formal.topsort cimport AcyclicTopSortTable, RobustTopSortTable
from grasp.semiring._semiring cimport Semiring
from grasp.formal.wfunc cimport WeightFunction


cdef weight_t node_value(Hypergraph forest,
                         WeightFunction omega,
                         Semiring semiring,
                         weight_t[::1] values,
                         id_t parent)


cpdef weight_t[::1] acyclic_value_recursion(Hypergraph forest,
                                            AcyclicTopSortTable tsort,
                                            Semiring semiring,
                                            WeightFunction omega=?)

cpdef weight_t[::1] acyclic_reversed_value_recursion(Hypergraph forest,
                                            AcyclicTopSortTable tsort,
                                            Semiring semiring,
                                            weight_t[::1] values,
                                            WeightFunction omega=?)

cpdef weight_t[::1] robust_value_recursion(Hypergraph forest,
                                           RobustTopSortTable tsort,
                                           Semiring semiring,
                                           WeightFunction omega=?)

cdef weight_t[::1] approximate_supremum(Hypergraph forest,
                                        WeightFunction omega,
                                        Semiring semiring,
                                        weight_t[::1] values,
                                        list bucket)

cpdef weight_t[::1] compute_edge_values(Hypergraph forest,
                                        Semiring semiring,
                                        weight_t[::1] node_values,
                                        WeightFunction omega=?,
                                        bint normalise=?)

cpdef weight_t[::1] compute_edge_expectation(Hypergraph forest,
                                        Semiring semiring,
                                        weight_t[::1] node_values,
                                        weight_t[::1] node_reversed_values,
                                        WeightFunction omega=?,
                                        bint normalise=?)

cdef class EdgeValues(WeightFunction):

    cdef Hypergraph _forest
    cdef Semiring _semiring
    cdef weight_t[::1] _node_values
    cdef weight_t[::1] _edge_values
    cdef WeightFunction _omega
    cdef bint _normalise


cdef class LazyEdgeValues(WeightFunction):

    cdef Hypergraph _forest
    cdef Semiring _semiring
    cdef weight_t[::1] _node_values
    cdef list _edge_values
    cdef WeightFunction _omega
    cdef bint _normalise

    cdef weight_t _unnormalised(self, id_t e)

    cdef _normalised(self, id_t e)
