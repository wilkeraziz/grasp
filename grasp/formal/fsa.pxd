from grasp.ptypes cimport id_t, weight_t
from grasp.cfg.symbol cimport Terminal as Label
cimport numpy as np


cdef class Arc:
    """
    An Arc can be seen as a special type of edge. It connects a single origin node to a destination node.
    Instead of an arbitrary rule, it contains a label (a Symbol) and it carries a weight.
    """
    
    cdef readonly id_t origin
    cdef readonly id_t destination
    cdef readonly Label label
    cdef readonly weight_t weight


cdef class DFA:
    
    cdef list _arcs
    cdef list _FS    # due to determinism, FS is a function of (origin, label)
    cdef set _initial
    cdef set _final
    
    cpdef id_t add_state(self)
    
    cpdef id_t add_arc(self, id_t origin, id_t destination, Label label, weight_t weight) except -100
    
    cpdef id_t fetch(self, id_t origin, Label label) except -100
    
    cpdef Arc arc(self, id_t i)
    
    cpdef make_initial(self, id_t state)
        
    cpdef make_final(self, id_t state)
    
    cpdef bint is_initial(self, id_t state)
    
    cpdef bint is_final(self, id_t state)
    
    cpdef size_t n_states(self)
    
    cpdef size_t n_arcs(self)
    
    cpdef iterarcs(self)
    
    cpdef iterinitial(self)
    
    cpdef iterfinal(self)


cpdef np.int_t[:,::1] floyd_warshall(DFA dfa, np.int_t inf=?)

cpdef DFA make_dfa(words, weight_t w=?)

cpdef DFA make_dfa_set(list sentences, weight_t w=?)

from grasp.scoring.scorer cimport StatefulScorer
cpdef DFA make_dfa_set2(list sentences, StatefulScorer stateful)