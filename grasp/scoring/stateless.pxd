from grasp.scoring.extractor cimport FRepr, Extractor
from grasp.ptypes cimport weight_t


cdef class Stateless(Extractor):

    cpdef FRepr featurize(self, edge)


cdef class WordPenalty(Stateless):

    cdef weight_t _penalty

cdef class ArityPenalty(Stateless):

    cdef weight_t _penalty
