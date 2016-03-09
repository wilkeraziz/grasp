"""
:Authors: - Wilker Aziz
"""

from grasp.cfg.symbol cimport Nonterminal
from grasp.cfg.rule cimport Rule
from grasp.ptypes cimport weight_t


cdef class SCFGProduction(Rule):

    cdef:
        Nonterminal _lhs
        tuple _irhs
        tuple _orhs
        tuple _nt_alignment
        dict _fmap

    cpdef weight_t fvalue(self, fname, weight_t default=?)


cdef class InputView(Rule):

    cdef SCFGProduction _srule


cdef class OutputView(Rule):

    cdef SCFGProduction _srule


cdef class InputGroupView(Rule):

    cdef tuple _srules


cdef class OutputGroupView(Rule):

    cdef tuple _srules
