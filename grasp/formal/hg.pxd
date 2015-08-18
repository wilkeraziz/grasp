from grasp.cfg.symbol cimport Symbol
from grasp.cfg.rule cimport Rule
from grasp.ptypes cimport id_t, weight_t


ctypedef tuple tail_t

cdef class Node:

    cdef readonly Symbol label


cdef class Edge:

    cdef readonly id_t head
    cdef readonly tail_t tail
    cdef readonly weight_t weight


cdef class Hypergraph:

    cdef list _nodes
    cdef list _edges
    cdef list _bs
    cdef list _fs
    cdef list _deps

    cdef dict _symbol_map
    cdef list _symbols
    cdef list _rules
    cdef set _glue

    cdef _add_node(self, Node node)

    cdef _add_edge(self, Edge edge)

    cpdef id_t add_node(self, Symbol label)

    cpdef id_t add_edge(self, Rule rule, bint glue=?)

    cpdef id_t add_xedge(self, Symbol lhs, tuple rhs, weight_t weight, Rule rule, bint glue=?)

    cpdef id_t fetch(self, Symbol sym, id_t default=?)

    cpdef id_t head(self, id_t e)

    cpdef tail_t tail(self, id_t e)

    cpdef weight_t weight(self, id_t e)

    cpdef size_t arity(self, id_t e)

    cpdef id_t child(self, id_t e, id_t i)

    cpdef bint is_terminal(self, id_t n)

    cpdef bint is_nonterminal(self, id_t n)

    cpdef bint is_source(self, id_t head)

    cpdef Symbol label(self, id_t n)

    cpdef Rule rule(self, id_t e)

    cpdef update(self, rules, bint glue=?)

    cpdef size_t n_nodes(self)

    cpdef size_t n_edges(self)

    cpdef size_t n_incoming(self, id_t n)

    cpdef iterglue(self)

    cpdef iterbs(self, id_t head)

    cpdef id_t bs_i(self, id_t head, size_t i)

    cpdef iterfs(self, id_t node)

    cpdef iterdeps(self, id_t node)

    cpdef bint self_depends(self, id_t node)


cpdef Hypergraph cfg_to_hg(grammars, glue_grammars)
