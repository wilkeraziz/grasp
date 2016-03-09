"""
:Authors: - Wilker Aziz
"""
from grasp.ptypes cimport weight_t


cdef class FRepr:

    cpdef weight_t dot(self, FRepr w)


cdef class FValue(FRepr):

    cdef readonly weight_t value


cdef class FVec(FRepr):

    cdef readonly weight_t[::1] vec


cdef class FMap(FRepr):

    cdef readonly dict map


cdef class Extractor:

    cdef int _uid
    cdef str _name

    cpdef FRepr weights(self, dict wmap)

    cpdef weight_t dot(self, FRepr frepr, FRepr wrepr)
