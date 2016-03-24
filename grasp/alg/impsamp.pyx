from cpython.object cimport Py_EQ, Py_NE
from grasp.scoring.frepr cimport FComponents
from grasp.ptypes cimport weight_t


cdef class ISDerivation:

    cdef readonly:
        tuple edges
        FComponents q_comps
        FComponents p_comps
        weight_t count

    def __init__(self, tuple edges, FComponents q_comps, FComponents p_comps, weight_t count):
        self.edges = edges
        self.q_comps = q_comps
        self.p_comps = p_comps
        self.count = count

    def __getstate__(self):
        return {'edges': self.edges,
                'q_comps': self.q_comps,
                'p_comps': self.p_comps,
                'count': self.count}

    def __setstate__(self, state):
        self.edges = state['edges']
        self.q_comps = state['q_comps']
        self.p_comps = state['p_comps']
        self.count = state['count']

    def __hash__(self):
        return hash(self.edges)

    def __richcmp__(ISDerivation x, ISDerivation y, int opt):
        if opt == Py_EQ:
            return x.edges == y.edges
        elif opt == Py_NE:
            return x.eges != y.edges
        else:
            raise ValueError('Cannot compare ISDerivation objects with opt=%d' % opt)


cdef class ISYield:

    cdef readonly:
        str y
        tuple D
        weight_t count

    def __init__(self, str y, derivations, weight_t count):
        self.y = y
        self.D = tuple(derivations)
        self.count = count

    def __getstate__(self):
        return {'y': self.y,
                'D': self.D,
                'count': self.count}

    def __setstate__(self, state):
        self.y = state['y']
        self.D = state['D']
        self.count = state['count']

    def __hash__(self):
        return hash(self.y)

    def __richcmp__(ISYield x, ISYield y, int opt):
        if opt == Py_EQ:
            return x.y == y.y
        elif opt == Py_NE:
            return x.y != y.y
        else:
            raise ValueError('Cannot compare ISYield objects with opt=%d' % opt)
