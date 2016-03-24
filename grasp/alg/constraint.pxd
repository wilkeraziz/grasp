from grasp.ptypes cimport id_t
from grasp.formal.hg cimport Hypergraph
from grasp.formal.fsa cimport DFA
cimport numpy as np


cdef class Constraint:

    cdef bint connected(self, id_t edge, id_t origin, id_t destination)


cdef class ConstraintContainer(Constraint):

    cdef list constraints


cdef class GlueConstraint(Constraint):


    cdef Hypergraph hg
    cdef DFA dfa


cdef class HieroConstraints(Constraint):

    cdef int longest
    cdef Hypergraph hg
    cdef DFA dfa
    cdef np.int_t[:,::1] shortest_paths

