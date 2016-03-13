"""
Contains class definitions for symbols (e.g. Terminal and Nonterminal) and other utilitary functions involving them.

:Authors: - Wilker Aziz
"""

from cpython.object cimport Py_EQ, Py_NE
from weakref import WeakValueDictionary
from grasp.ptypes cimport id_t


cdef class Symbol:

    def __init__(self, obj):
        self._obj = obj

    def __hash__(self):
        return hash(self._obj)

    def __richcmp__(x, y, opt):
        if opt == Py_EQ:
            return type(x) == type(y) and x.underlying == y.underlying
        elif opt == Py_NE:
            return type(x) != type(y) or x.underlying != y.underlying
        else:
            raise ValueError('Cannot compare symbol with opt=%d' % opt)

    def __str__(self):
        return str(self._obj)

    def __repr__(self):
        return repr(self._obj)

    def __getstate__(self):
        return {'obj': self._obj}

    def __setstate__(self, d):
        self._obj = d['obj']

    property underlying:
        def __get__(self):
            """The underlying object that uniquely represents the symbol."""
            return self._obj


cdef class Terminal(Symbol):

    def __init__(self, surface):
        super(Terminal, self).__init__(surface)

    def __repr__(self):
        """Return the string associated with the underlying object wrapped with single quotes."""
        return "'{0}'".format(str(self))

    property surface:
        def __get__(self):
            """The surface word (syntactic sugar for the underlying object)."""
            return self.underlying


cdef class Nonterminal(Symbol):

    def __init__(self, label):
        super(Nonterminal, self).__init__(label)

    def __repr__(self):
        return '[{0}]'.format(str(self))

    property label:
        def __get__(self):
            """The nonterminal category (syntactic sugar for the underlying object)."""
            return self.underlying


cdef class Span(Nonterminal):

    def __init__(self, Nonterminal base, id_t start, id_t end):
        super(Span, self).__init__((base, start, end))

    def __str__(self):
        if self.underlying[1] >= 0 and self.underlying[2] >= 0:
            return '{0}:{1}-{2}'.format(self.underlying[0],
                                        self.underlying[1],
                                        self.underlying[2])
        else:
            return '{0}:-'.format(self.underlying[0])

    def __repr__(self):
        return '[{0}]'.format(str(self))

    property base:
        def __get__(self):
            """The nonterminal category."""
            return <Nonterminal>self.underlying[0]

    property start:
        def __get__(self):
            """The start position."""
            return <id_t>self.underlying[1]

    property end:
        def __get__(self):
            """The end position."""
            return <id_t>self.underlying[2]


cdef class SymbolFactory:

    def __init__(self):
        self._terminals = WeakValueDictionary()
        self._nonterminals = WeakValueDictionary()

    cpdef Terminal terminal(self, surface):
        cdef Terminal sym = self._terminals.get(surface, None)
        if sym is None:
            sym = Terminal(surface)
            self._terminals[surface] = sym
        return sym

    cpdef Nonterminal nonterminal(self, label):
        cdef Nonterminal sym = self._nonterminals.get(label, None)
        if sym is None:
            sym = Nonterminal(label)
            self._nonterminals[label] = sym
        return sym

    cpdef Span span(self, Nonterminal base, id_t start, id_t end):
        cdef tuple key = (base, start, end)
        cdef Span sym = self._nonterminals.get(key, None)
        if sym is None:
            sym = Span(base, start, end)
            self._nonterminals[key] = sym
        return sym


cpdef Symbol make_span(Symbol symbol, int sfrom=-1, int sto=-1):
    return symbol if isinstance(symbol, Terminal) else Span(symbol, sfrom, sto)


cpdef flatten_symbol(Symbol symbol):
    return Nonterminal(str(symbol)) if isinstance(symbol, Span) else symbol


class StaticSymbolFactory:

    _terminals = WeakValueDictionary()
    _nonterminals = WeakValueDictionary()

    @classmethod
    def terminal(cls, surface):
        cdef Terminal sym = cls._terminals.get(surface, None)
        if sym is None:
            sym = Terminal(surface)
            cls._terminals[surface] = sym
        return sym

    @classmethod
    def nonterminal(cls, label):
        cdef Nonterminal sym = cls._nonterminals.get(label, None)
        if sym is None:
            sym = Nonterminal(label)
            cls._nonterminals[label] = sym
        return sym

    @classmethod
    def span(cls, Nonterminal base, id_t start, id_t end):
        cdef tuple key = (base, start, end)
        cdef Span sym = cls._nonterminals.get(key, None)
        if sym is None:
            sym = Span(base, start, end)
            cls._nonterminals[key] = sym
        return sym