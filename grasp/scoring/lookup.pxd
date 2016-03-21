from grasp.scoring.extractor cimport TableLookup
from grasp.ptypes cimport weight_t


cdef class NamedFeature(TableLookup):

    cdef str _fname
    cdef weight_t _default


cdef class RuleTable(TableLookup):

    cdef tuple _fnames
    cpdef weight_t _default


cdef class LogTransformedRuleTable(RuleTable):

    pass