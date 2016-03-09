from grasp.cfg.symbol cimport Nonterminal
from grasp.ptypes cimport weight_t


cdef class Rule:

    cpdef weight_t fvalue(self, fname, weight_t default=?)


cdef class NewCFGProduction(Rule):

    cdef:
        Nonterminal _lhs  # TODO: generalise Nonterminal to LHS (e.g. compatible with MCFGs)
        tuple _rhs        # TODO: generalise tuple to RHS (e.g. compatible with MCFGs)
        int _hash
        dict _fmap


cdef class _CFGProduction(Rule):

    cdef:
        Nonterminal _lhs
        tuple _rhs
        weight_t _weight
        int _hash