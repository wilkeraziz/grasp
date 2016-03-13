"""
:Authors: - Wilker Aziz
"""
from grasp.ptypes cimport weight_t
from grasp.scoring.frepr cimport FRepr


cdef class Extractor:

    cdef int _uid
    cdef str _name

    cpdef FRepr weights(self, dict wmap)

    cpdef weight_t dot(self, FRepr frepr, FRepr wrepr)

    cpdef FRepr constant(self, weight_t value)


cdef class TableLookup(Extractor):

    cpdef FRepr featurize(self, rule)


cdef class Stateless(Extractor):

    cpdef FRepr featurize(self, edge)


cdef class StatefulFRepr:

    cdef readonly FRepr frepr
    cdef readonly object state


cdef class Stateful(Extractor):

    cpdef object initial(self)

    cpdef object final(self)

    cpdef FRepr featurize_initial(self)

    cpdef FRepr featurize_final(self, context)

    cpdef StatefulFRepr featurize(self, word, context)

    cpdef FRepr featurize_yield(self, derivation_yield)
