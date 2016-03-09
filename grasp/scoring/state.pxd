"""
:Authors: - Wilker Aziz
"""

from grasp.ptypes cimport id_t


cdef class StateMapper:

    cdef dict _state2int
    cdef list _int2state

    cpdef id_t id(self, object state) except -1

    cpdef object state(self, id_t i)