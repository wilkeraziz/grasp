

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

    cdef object _squared_gradient
    cdef float _gamma0
    cdef size_t _t
    cdef float _epsilon


cdef class AdaDelta(SGD):

    cdef object _squared_gradient
    cdef object _squared_delta
    cdef float _gamma0
    cdef size_t _t
    cdef float _epsilon
    cdef float _rho

