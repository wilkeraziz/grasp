"""
Contains class definitions for symbols (e.g. Terminal and Nonterminal).

:Authors: - Wilker Aziz
"""
from grasp.ptypes cimport id_t


cdef class Symbol:
    """
    Any immutable hashable object.
    """

    cdef object __weakref__
    cdef object _obj


cdef class Terminal(Symbol):
    """
    A terminal symbol is a literal.
    """

    pass


cdef class Nonterminal(Symbol):
    """
    A nonterminal symbol is a variable.
    """

    pass


cdef class Span(Nonterminal):
    """
    A nonterminal symbol spanning a path between two states.
    """

    pass


cdef class SymbolFactory:

    cdef object _terminals
    cdef object _nonterminals

    cpdef Terminal terminal(self, surface)
    cpdef Nonterminal nonterminal(self, label)
    cpdef Span span(self, Nonterminal base, id_t start, id_t end)


cpdef Symbol make_span(Symbol symbol, int sfrom=?, int sto=?)