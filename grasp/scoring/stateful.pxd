from grasp.scoring.extractor cimport Extractor, FRepr


cdef class StatefulFRepr:

    cdef readonly FRepr frepr
    cdef readonly object state


cdef class Stateful(Extractor):

    cpdef initial(self)

    cpdef final(self)

    cpdef FRepr featurize_initial(self)

    cpdef FRepr featurize_final(self, context)

    cpdef StatefulFRepr featurize(self, word, context)

    cpdef FRepr featurize_derivation(self, derivation)
