from grasp.ptypes cimport id_t, weight_t
import numpy as np


cdef class Arc:
    
    def __init__(self, id_t origin, id_t destination, Label label, weight_t weight):
        self.origin = origin
        self.destination = destination
        self.label = label
        self.weight = weight
    
    def __str__(self):
        return 'from={0} to={1} label={2} weight={3}'.format(self.origin,
                                                             self.destination,
                                                             repr(self.label),
                                                             self.weight)
    
    def __repr__(self):
        return 'Arc(%r, %r, %r, %r)' % (self.origin,
                                        self.destination,
                                        self.label,
                                        self.weight)
        
    
cdef class DFA:
    
    def __init__(self):
        self._FS = []  # origin => label => (destination, weight) 
        self._arcs = []
        self._initial = set()
        self._final = set()

        
    cpdef id_t add_state(self):
        self._FS.append(dict())
        return len(self._FS) - 1
    
    cpdef id_t add_arc(self, id_t origin, id_t destination, Label label, weight_t weight) except -100:
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


cpdef np.int_t[:,::1] floyd_warshall(DFA dfa, np.int_t inf=-1):
    """
    Compute the lenght of the minimum path between every node pair.

    :param dfa: DFA with |V| nodes and |E| edges
    :param inf: a representation of infinity
    :return: a |V|x|V| memoryview with distances (inf represents lack of path between nodes)
    """
    cdef np.int_t[:,::1] dist = np.full((dfa.n_states(), dfa.n_states()), inf, dtype=np.int)
    cdef size_t a, i, j, k
    cdef Arc arc
    for v in range(dfa.n_states()):
        dist[v,v] = 0
    for a in range(dfa.n_arcs()):
        arc = dfa.arc(a)
        dist[arc.origin,arc.destination] = 1  # TODO generalise to weight function
    for k in range(dfa.n_states()):
        for i in range(dfa.n_states()):
            for j in range(dfa.n_states()):
                if dist[i,k] == inf or dist[k,j] == inf:  # inf + anything is inf
                    continue
                if dist[i,j] == inf or dist[i,j] > dist[i,k] + dist[k,j]:  # inf is bigger than anything
                    dist[i,j] = dist[i,k] + dist[k,j]
    return dist



cpdef DFA make_dfa(words, weight_t w=0.0):
    cdef DFA dfa = DFA()
    cdef id_t i = dfa.add_state()
    cdef str word
    dfa.make_initial(i)
    for word in words:
        i = dfa.add_state()
        dfa.add_arc(i - 1, i, Label(word), w)
    dfa.make_final(i)
    return dfa


cpdef DFA make_dfa_set(list sentences, weight_t w=0.0):
    cdef DFA dfa = DFA()
    cdef id_t s0 = dfa.add_state()
    cdef id_t sfrom, sto
    cdef str word, key
    cdef list sentence
    cdef states = {'': s0}
    dfa.make_initial(s0)

    for sentence in sentences:
        sfrom = s0
        for i, word in enumerate(sentence):
            key = '{0}-{1}'.format(sfrom, word)
            sto = states.get(key, -1)
            if sto == -1:
                sto = dfa.add_state()
                states[key] = sto
                dfa.add_arc(sfrom, sto, Label(word), w)
            sfrom = sto
        dfa.make_final(sto)

    return dfa


cpdef DFA make_dfa_set2(list sentences, StatefulScorer stateful):
    cdef DFA dfa = DFA()
    cdef id_t s0 = dfa.add_state()
    cdef id_t sfrom, sto
    cdef str word, key
    cdef list sentence
    cdef states = {'': s0}
    cdef Label label
    cdef dfa2scorer= {s0: stateful.initial()}
    dfa.make_initial(s0)

    for sentence in sentences:
        sfrom = s0
        for i, word in enumerate(sentence):
            key = '{0}-{1}'.format(sfrom, word)
            sto = states.get(key, -1)
            if sto == -1:
                sto = dfa.add_state()
                states[key] = sto
                label = Label(word)
                weight, scorer_to = stateful.score(label, context=dfa2scorer[sfrom])
                dfa.add_arc(sfrom, sto, label, weight)
                dfa2scorer[sto] = scorer_to
            sfrom = sto
        dfa.make_final(sto)

    return dfa
