

cdef class SGD:

    cpdef update(self, parameters, gradients)


cdef class FlatLearningRateSGD(SGD):

    cdef float _gamma0
    cdef size_t _t


cdef class DecayingLearningRateSGD(SGD):

    cdef object _variances
    cdef float _gamma0
    cdef size_t _t


cdef class AdaGrad(SGD):

    cdef object _accumulator
    cdef float _gamma0
    cdef size_t _t

