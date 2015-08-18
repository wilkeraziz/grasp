cimport numpy as np

from libcpp.stack cimport stack
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from grasp.ptypes cimport id_t
from grasp.formal.hg cimport Hypergraph


ctypedef struct tarjan_node_t:
    id_t index
    id_t low
    bint stacked


cdef np.int_t[::1] acyclic_topsort(Hypergraph hg)

cdef strong_connect(id_t v,
                      vector[tarjan_node_t]& nodes,
                      id_t* index,
                      stack[id_t]& agenda,
                      vector[unordered_set[id_t]]& deps,
                      vector[vector[id_t]]& order)

cdef void c_tarjan(Hypergraph hg, vector[vector[id_t]]& order)

cpdef list tarjan(Hypergraph hg)

cdef np.int_t[::1] robust_topsort(Hypergraph hg, vector[vector[id_t]]& components)


cdef class TopSortTable: pass

cdef class AcyclicTopSortTable(TopSortTable):

    cdef Hypergraph _hg
    cdef np.int_t[::1] _levels
    cdef list _tsort

    cpdef size_t n_levels(self)

    cpdef size_t n_top(self)

    cpdef size_t level(self, id_t node)

    cpdef id_t root(self) except -1

    cpdef itertop(self)

    cpdef iterlevels(self, bint reverse=?, size_t skip=?)

    cpdef iternodes(self, bint reverse=?, size_t skip=?)


cdef class RobustTopSortTable(TopSortTable):

    cdef Hypergraph _hg
    cdef np.int_t[::1] _levels
    cdef vector[vector[id_t]] _components
    cdef list _tsort

    cpdef size_t n_levels(self)

    cpdef size_t n_top(self)

    cpdef size_t level(self, id_t node)

    cpdef id_t root(self) except -1

    cpdef itertopbuckets(self)

    cdef itertopnodes(self)

    cpdef iterlevels(self, bint reverse=?, size_t skip=?)

    cpdef iterbuckets(self, bint reverse=?, size_t skip=?)

    cpdef iternodes(self, bint reverse=?, size_t skip=?)

    cpdef bint is_loopy(self, list bucket)