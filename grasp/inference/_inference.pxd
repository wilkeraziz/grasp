"""
:Authors: - Wilker Aziz
"""

from grasp.semiring._semiring cimport Semiring
from grasp.formal.hg cimport Hypergraph
from grasp.formal.topsort cimport TopSortTable
from grasp.inference._value cimport ValueFunction
from grasp.ptypes cimport weight_t, id_t


cpdef tuple sample(Hypergraph forest,
                   id_t root,
                   Semiring semiring,
                   ValueFunction omega)


cpdef list batch_sample(Hypergraph forest,
                        TopSortTable tsort,
                        Semiring semiring,
                        size_t size,
                        ValueFunction omega=?,
                        weight_t[::1] node_values=?,
                        weight_t[::1] edge_values=?)

cpdef tuple viterbi_derivation(Hypergraph forest,
                               TopSortTable tsort,
                               ValueFunction omega=?,
                               weight_t[::1] node_values=?,
                               weight_t[::1] edge_values=?)


cpdef tuple sample_derivation(Hypergraph forest,
                              TopSortTable tsort,
                              ValueFunction omega=?,
                              weight_t[::1] node_values=?,
                              weight_t[::1] edge_values=?)


cpdef list sample_derivations(Hypergraph forest,
                              TopSortTable tsort,
                              size_t size,
                              ValueFunction omega=?,
                              weight_t[::1] node_values=?,
                              weight_t[::1] edge_values=?)


cdef class DerivationCounter:

    cdef:
        Hypergraph _forest
        TopSortTable _tsort
        ValueFunction _omega
        id_t _root
        bint _counts_computed
        weight_t[::1] _count_values

    cdef void do(self)

    cpdef id_t count(self, id_t node)

    cpdef id_t n_derivations(self)


cdef class AncestralSampler:

    cdef:
        Hypergraph _forest
        TopSortTable _tsort
        ValueFunction _omega
        weight_t[::1] _node_values
        weight_t[::1] _edge_values
        id_t _root
        DerivationCounter _counter

    cpdef list sample(self, size_t n)

    cpdef weight_t prob(self, tuple d)

    cpdef int n_derivations(self)