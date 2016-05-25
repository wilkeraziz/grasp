from grasp.scoring.frepr cimport FComponents
from grasp.ptypes cimport weight_t
from grasp.scoring.extractor cimport Extractor


cdef class Model:

    cdef tuple _extractors
    cdef dict _wmap
    cdef FComponents _weights
    cdef dict _name_to_position

    cpdef tuple extractors(self)

    cpdef weight_t score(self, FComponents freprs) except *

    cpdef tuple fnames(self, wkey=?)

    cpdef FComponents weights(self)

    cpdef FComponents constant(self, weight_t value)

    cpdef int get_position(self, name)

    cpdef Extractor get_extractor(self, name)


cdef class DummyModel(Model):

    pass


cdef class ModelContainer(Model):  # TODO: rename it to LogLinearModel

    cdef public Model lookup, stateless, stateful, dummy

    cpdef itercomponents(self)


cdef class ModelView(ModelContainer):

    cdef:
        ModelContainer _local
        ModelContainer _nonlocal

    cpdef FComponents merge(self, FComponents local_comps, FComponents nonlocal_comps)

    cpdef ModelContainer local_model(self)
    cpdef ModelContainer nonlocal_model(self)