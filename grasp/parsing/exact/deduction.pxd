from grasp.ptypes cimport id_t, weight_t
from grasp.formal.hg cimport Hypergraph
from grasp.formal.fsa cimport DFA
from grasp.cfg.symbol cimport Symbol, Terminal, Nonterminal, make_span
from libcpp.pair cimport pair
cimport numpy as np
from grasp.semiring._semiring cimport Semiring


cdef class Item:
    
    cdef readonly id_t edge
    cdef readonly tuple states
    cdef readonly weight_t weight
    

cdef class ItemFactory:
    
    cdef list _items
    cdef list _items_by_key
    
    cpdef Item item(self, id_t i)
        
    cpdef Item get_item(self, id_t edge, tuple states, weight_t weight)

    cdef pair[id_t, bint] insert(self, id_t edge, tuple states, weight_t weight)

    cdef pair[id_t, bint] advance(self, Item item, id_t to, weight_t weight)


cdef class Agenda:

    cdef Hypergraph _hg
    cdef public list _waiting
    cdef public list _generating
    cdef public dict _complete
    cdef list _active

    cdef Item pop(self)

    cdef void push(self, Item item)

    cdef void make_generating(self, Item item)

    cdef void make_waiting(self, Item item)

    cdef void discard(self, Item item)

    cdef set waiting(self, id_t head, id_t start)

    cdef set destinations(self, id_t node, id_t start)

    cdef Hypergraph make_output(self, id_t root, Nonterminal goal, DFA dfa)


cdef class SliceVariables:

    cpdef bint is_inside(self, key, p)

    cpdef weight_t logpdf(self, key, p)

    cpdef weight_t pdf(self, key, p)


cdef class DeductiveParser:

    cdef Hypergraph _hg
    cdef DFA _dfa
    cdef object _semiring
    cdef SliceVariables _slicevars

    cdef Agenda _agenda
    cdef ItemFactory _ifactory
    cdef np.int8_t[::1] _glue

    cdef bint is_glue(self, id_t e)

    cdef Symbol next_symbol(self, Item item)

    cdef bint is_complete(self, Item item)

    cdef void axioms(self, id_t root)

    cdef void inference(self, Item item)

    cdef bint scan(self, Item item)

    cdef bint complete_itself(self, Item item)

    cdef bint complete_others(self, Item item)

    cdef void process_incomplete(self, Item item)

    cdef void process_complete(self, Item item)

    cdef bint push(self, const pair[id_t,bint]& insertion)

    cdef Item pop(self)

    cpdef Hypergraph do(self, id_t root, Symbol goal)


cdef class Earley(DeductiveParser):

    cdef set _predictions

    cdef void axioms(self, id_t root)

    cdef bint _predict(self, Item item)

    cdef void process_complete(self, Item item)

    cdef void process_incomplete(self, Item item)


cdef class Nederhof(DeductiveParser):

    cdef list _edges_by_tail0

    cdef void axioms(self, id_t root)

    cdef void _complete_tail0(self, id_t node, id_t start, id_t end, weight_t weight)

    cdef void process_complete(self, Item item)

    cdef void process_incomplete(self, Item item)


cpdef weight_t[::1] reweight(Hypergraph forest, SliceVariables slicevars, Semiring semiring)