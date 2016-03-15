from grasp.scoring.extractor cimport Stateless
from grasp.ptypes cimport weight_t


cdef class WordPenalty(Stateless):

    cdef weight_t _penalty


cdef class ArityPenalty(Stateless):

    cdef weight_t _penalty
    cdef int _max_arity
