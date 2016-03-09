from grasp.scoring.extractor cimport FRepr, Extractor


cdef class TableLookup(Extractor):

    cpdef FRepr featurize(self, rule)


cdef class RuleTable(TableLookup):

    cdef tuple _fnames