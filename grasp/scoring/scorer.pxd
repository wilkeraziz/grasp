"""
:Authors: - Wilker Aziz
"""

from grasp.ptypes cimport weight_t, id_t
from grasp.scoring.state cimport StateMapper


cdef class LogLinearModel:

    cdef dict _wmap
    cdef tuple _extractors
    cdef tuple _lookup
    cdef tuple _stateless
    cdef tuple _stateful
    cdef tuple _lookup_weights
    cdef tuple _stateless_weights
    cdef tuple _stateful_weights

    cpdef weight_t lookup_score(self, list freprs)

    cpdef weight_t stateless_score(self, list freprs)

    cpdef weight_t stateful_score(self, list freprs)


cdef class Scorer: pass


cdef class TableLookupScorer(Scorer):

    cdef LogLinearModel _model

    cpdef weight_t score(self, rule)


cdef class StatelessScorer(Scorer):

    cdef LogLinearModel _model
    cdef tuple _extractors

    cpdef weight_t score(self, edge)


cdef class StatefulScorer(Scorer):

    cdef LogLinearModel _model
    cdef tuple _extractors
    cdef StateMapper _mapper
    cdef id_t _initial
    cdef id_t _final

    cpdef id_t initial(self)

    cpdef id_t final(self)

    cpdef weight_t initial_score(self)

    cpdef weight_t final_score(self, context)

    cpdef tuple score(self, word, context)

    cpdef weight_t score_derivation(self, derivation)
