from grasp.scoring.frepr cimport FComponents
from grasp.ptypes cimport weight_t


cdef class Model:

    cdef tuple _extractors
    cdef dict _wmap
    cdef FComponents _weights

    cpdef tuple extractors(self)

    cpdef weight_t score(self, FComponents freprs)

    cpdef tuple fnames(self, wkey=?)

    cpdef FComponents weights(self)

    cpdef FComponents constant(self, weight_t value)


cdef class DummyModel(Model):

    pass


cdef class ModelContainer(Model):  # TODO: rename it to LogLinearModel

    cdef public Model lookup, stateless, stateful, dummy

    cpdef itercomponents(self)


