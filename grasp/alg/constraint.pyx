from grasp.ptypes cimport id_t
from grasp.formal.fsa cimport floyd_warshall


cdef class Constraint:
    """
    Constrain intersection procedures.
    This is itself a dummy constraint, in the sense that it does not constain anything.
    """

    cdef bint connected(self, id_t edge, id_t origin, id_t destination):
        return True


cdef class ConstraintContainer(Constraint):

    def __init__(self, iterable):
        self.constraints = []
        for constraint in iterable:
            if not isinstance(constraint, Constraint):
                raise ValueError('A ConstraintContainer can only host Constraint objects: %s' % type(constraint))
            self.constraints.append(constraint)

    cdef bint connected(self, id_t edge, id_t origin, id_t destination):
        cdef Constraint c
        for c in self.constraints:
            if not c.connected(edge, origin, destination):
                return False
        return True


cdef class GlueConstraint(Constraint):

    def __init__(self, Hypergraph hg, DFA dfa):
        self.hg = hg
        self.dfa = dfa

    cdef bint connected(self, id_t edge, id_t origin, id_t destination):
        # we only prune glue edges headed by non-initial states
        return not self.hg.is_glue(edge) or self.dfa.is_initial(origin)


cdef class HieroConstraints(Constraint):
    """
    In SMT hierarchical decoding, one typically constrains
        1. the span of X-rules to a maximum length
        2. the origin state of S-rules to being initial
    """

    def __init__(self, Hypergraph hg, DFA dfa, int longest):
        self.hg = hg
        self.dfa = dfa
        self.longest = longest
        if longest >= 0:
            self.shortest_paths = floyd_warshall(dfa, inf=-1)

    cdef bint connected(self, id_t edge, id_t origin, id_t destination):
        """
        Check whether we have a path between two connected states in the automaton which is short enough given
        possible constraints.
        """
        if self.hg.is_glue(edge):  # glue rules are S-rules
            # they are only constrained wrt to origin state being initial
            return self.dfa.is_initial(origin)
        # X-rules can only be constrained wrt to the length of their spans
        if self.longest < 0:
            return True
        return 0 <= self.shortest_paths[origin, destination] <= self.longest