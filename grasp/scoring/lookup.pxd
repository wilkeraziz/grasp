from grasp.scoring.extractor cimport TableLookup


cdef class RuleTable(TableLookup):

    cdef tuple _fnames


cdef class LogTransformedRuleTable(RuleTable):

    pass