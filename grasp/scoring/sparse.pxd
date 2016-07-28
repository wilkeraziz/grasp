from grasp.scoring.extractor cimport Stateless
from grasp.ptypes cimport weight_t


cdef class RuleIndicator(Stateless):

    cdef int _n_features
    cdef object _hasher




