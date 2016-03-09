from grasp.ptypes cimport weight_t


cdef class Model:

    pass


cdef class DummyConstant(Model):


    cdef weight_t _value


cdef class PCFG(Model):

    cdef str _fname
    cdef object _transform