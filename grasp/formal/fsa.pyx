from grasp.ptypes cimport id_t, real_t


cdef class Arc:
    
    def __init__(self, id_t origin, id_t destination, Label label, real_t weight):
        self._origin = origin
        self._destination = destination
        self._label = label
        self._weight = weight
    
    def __str__(self):
        return 'from={0} to={1} label={2} weight={3}'.format(self._origin, 
                                                             self._destination, 
                                                             repr(self._label),
                                                             self._weight)
    
    def __repr__(self):
        return 'Arc(%r, %r, %r, %r)' % (self._origin, 
                                        self._destination, 
                                        self._label, 
                                        self._weight)

    property origin:
        def __get__(self):
            return self._origin
        
    property destination:
        def __get__(self):
            return self._destination
    
    property label:
        def __get__(self):
            return self._label
        
    property weight:
        def __get__(self):
            return self._weight
        
    
cdef class DFA:
    
    def __init__(self):
        self._FS = []  # origin => label => (destination, weight) 
        self._arcs = []
        self._initial = set()
        self._final = set()

        
    cpdef id_t add_state(self):
        self._FS.append(dict())
        return len(self._FS) - 1
    
    cpdef id_t add_arc(self, id_t origin, id_t destination, Label label, real_t weight) except -100:
        cdef id_t i = self._FS[origin].get(label, -1)
        if i < 0:
            i = len(self._arcs)
            self._arcs.append(Arc(origin, destination, label, weight))
            self._FS[origin][label] = i
        return i
    
    cpdef id_t fetch(self, id_t origin, Label label) except -100:
        return self._FS[origin].get(label, -1)
    
    cpdef Arc arc(self, id_t arc):
        return self._arcs[arc]
    
    cpdef make_initial(self, id_t state):
        self._initial.add(state)
        
    cpdef make_final(self, id_t state):
        self._final.add(state)
    
    cpdef bint is_initial(self, id_t state):
        return state in self._initial
    
    cpdef bint is_final(self, id_t state):
        return state in self._final
    
    cpdef size_t n_states(self):
        return len(self._FS)
    
    cpdef size_t n_arcs(self):
        return len(self._arcs)
    
    cpdef iterarcs(self):
        return iter(self._arcs)
    
    cpdef iterinitial(self):
        return iter(self._initial)
    
    cpdef iterfinal(self):
        return iter(self._final)
    
    def __str__(self):
        return 'states={0} arcs={1}\n{2}\ninitial=({3})\nfinal=({4})'.format(self.n_states(),
                                                 self.n_arcs(),
                                                 '\n'.join(str(arc) for arc in self.iterarcs()),
                                                 ' '.join(str(i) for i in self.iterinitial()),
                                                 ' '.join(str(i) for i in self.iterfinal()))
        

cpdef DFA make_dfa(words, real_t w=0.0):
    cdef DFA dfa = DFA()
    cdef id_t i = dfa.add_state()
    cdef str word
    dfa.make_initial(i)
    for word in words:
        i = dfa.add_state()
        dfa.add_arc(i - 1, i, Label(word), w)
    dfa.make_final(i)
    return dfa
