"""
:Authors: - Wilker Aziz
"""

from grasp.ptypes cimport weight_t, id_t
from grasp.scoring.state cimport StateMapper
from grasp.scoring.frepr cimport FComponents
from grasp.scoring.model cimport Model
from grasp.semiring._semiring cimport Semiring
from grasp.cfg.rule cimport Rule


cdef class Scorer:

    cdef Model _model

    cpdef tuple extractors(self)

    cpdef Model model(self)

    cpdef FComponents constant(self, weight_t value)


cdef class TableLookupScorer(Scorer):

    cpdef FComponents featurize(self, rule)

    cpdef weight_t score(self, rule)

    cpdef tuple featurize_and_score(self, rule)

    cpdef tuple featurize_and_score_derivation(self, tuple rules, Semiring semiring)


cdef class StatelessScorer(Scorer):

    cpdef FComponents featurize(self, edge)

    cpdef weight_t score(self, edge)

    # TODO: here an edge shoul be represented by (head label, tail labels, rule)
    # currently it is just a rule
    cpdef tuple featurize_and_score(self, edge)

    # TODO: here a derivation should be represented by a sequence of edges
    # where each edge is represented by (head label, tail labels, rule)
    # currently is a sequence of rules
    cpdef tuple featurize_and_score_derivation(self, tuple edges, Semiring semiring)


cdef class StatefulScorer(Scorer):

    cdef StateMapper _mapper
    cdef id_t _initial
    cdef id_t _final

    cpdef id_t initial(self)

    cpdef id_t final(self)

    cpdef FComponents featurize_initial(self)

    cpdef FComponents featurize_final(self, context)

    cpdef tuple featurize(self, word, context)

    cpdef FComponents featurize_yield(self, derivation_yield)

    cpdef tuple featurize_and_score(self, word, context)

    cpdef weight_t initial_score(self)

    cpdef weight_t final_score(self, context)

    cpdef tuple score(self, word, context)

    cpdef weight_t score_yield(self, derivation_yield)

    cpdef tuple featurize_and_score_yield(self, derivation_yield)
