from grasp.alg.slicevars cimport SliceVariables

from grasp.ptypes cimport id_t, weight_t
from grasp.formal.hg cimport Hypergraph
from grasp.formal.fsa cimport DFA, Arc
from grasp.cfg.symbol cimport Symbol, Terminal, Nonterminal, make_span
from grasp.cfg.rule cimport Rule
from libcpp.pair cimport pair
cimport numpy as np
from grasp.semiring._semiring cimport Semiring
from grasp.scoring.scorer cimport TableLookupScorer, StatelessScorer, StatefulScorer

from grasp.scoring.frepr cimport FComponents

from grasp.alg.value cimport ValueFunction, EdgeWeight


cdef class Item:
    
    cdef readonly id_t edge
    cdef readonly tuple states
    cdef readonly weight_t weight
    # TODO: change this tuple to FComponents
    cdef readonly tuple frepr
    

cdef class ItemFactory:
    
    cdef list _items
    cdef list _items_by_key
    
    cpdef Item item(self, id_t i)
        
    cpdef Item get_item(self, id_t edge, tuple states, weight_t weight, tuple frepr)

    cdef pair[id_t, bint] insert(self, id_t edge, tuple states, weight_t weight, tuple frepr)

    cdef pair[id_t, bint] advance(self, Item item, id_t to, weight_t weight, tuple frepr)


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

    cdef Hypergraph make_output(self, id_t root,
                                Rule goal_rule,
                                set initial,
                                set final,
                                list mapping=?,
                                list components=?,
                                FComponents comp_one=?)


cdef class DeductiveIntersection:

    cdef Hypergraph _hg
    cdef ValueFunction _omega
    cdef Semiring _semiring
    cdef SliceVariables _slicevars

    cdef Agenda _agenda
    cdef ItemFactory _ifactory
    cdef np.int8_t[::1] _glue

    cdef bint is_glue(self, id_t e)

    cdef Symbol next_symbol(self, Item item)

    cdef bint is_complete(self, Item item)

    cdef void inference(self, id_t root)

    cdef void process(self, Item item)

    cdef bint complete_itself(self, Item item)

    cdef bint complete_others(self, Item item)

    cdef bint push(self, const pair[id_t,bint]& insertion)

    cdef Item pop(self)

    # The following methods need to be implemented by subclasses

    cdef void axioms(self, id_t root)

    cdef bint scan(self, Item item)

    cdef void process_incomplete(self, Item item)

    cdef void process_complete(self, Item item)

    cpdef Hypergraph do(self, id_t root, Rule goal_rule)


cdef class Parser(DeductiveIntersection):
    """
    A parser is an instance of intersection where the forest is intersected with a DFA.
    This DFA is an explicitly instantiated automaton.
    """

    cdef DFA _dfa

    cdef bint scan(self, Item item)

    cpdef Hypergraph do(self, id_t root, Rule goal_rule)


cdef class EarleyParser(Parser):
    """
    A top-down parser.
    """

    cdef set _predictions

    cdef void axioms(self, id_t root)

    cdef bint _predict(self, Item item)

    cdef void process_complete(self, Item item)

    cdef void process_incomplete(self, Item item)


cdef class NederhofParser(Parser):
    """
    A bottom-up parser.
    """

    cdef list _edges_by_tail0

    cdef void axioms(self, id_t root)

    cdef void _complete_tail0(self, id_t node, id_t start, id_t end, weight_t weight)

    cdef void process_complete(self, Item item)

    cdef void process_incomplete(self, Item item)


cdef class Rescorer(DeductiveIntersection):
    """
    A rescorer is an instance of intersection where the forest is intersected with an implicit automaton.
    States are generated by a stateful scorer (as opposed to being observed in an explicitly instantiated DFA).
    As opposed to a Parser, a Rescorer cannot leave derivations out of the forest, instead it must reweight them all.
    """

    cdef:
        TableLookupScorer _lookup
        StatelessScorer _stateless
        StatefulScorer _stateful
        id_t _initial
        id_t _final
        list _mapping
        bint _keep_frepr
        list _components
        #FComponents _stateless_one, _stateful_one, _comp_one
        list _skeleton_frepr



    cdef weight_t score_on_creation(self, id_t e, list parts)

    cdef bint scan(self, Item item)

    cpdef Hypergraph do(self, id_t root, Rule goal_rule)

    cpdef id_t maps_to(self, id_t e)

    cpdef list components(self)

    cpdef list skeleton_components(self)

cdef class EarleyRescorer(Rescorer):
    """
    An Earley rescorer implements a top-down rescorer based on the Earley algorithm for parsing.
    """

    cdef set _predictions

    cdef void axioms(self, id_t root)

    cdef bint _predict(self, Item item)

    cdef void process_complete(self, Item item)

    cdef void process_incomplete(self, Item item)


cpdef weight_t[::1] reweight(Hypergraph forest, SliceVariables slicevars, Semiring semiring, ValueFunction omega=?)