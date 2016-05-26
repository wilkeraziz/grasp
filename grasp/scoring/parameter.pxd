from grasp.ptypes cimport weight_t


cdef class GaussianPrior:

    cpdef float mean(self, key)

    cpdef float var(self)

    cpdef weight_t[::1] mean_vector(self, keys)


cdef class SymmetricGuassianPrior(GaussianPrior):

    cdef:
        float _mean
        float _var


cdef class AsymmetricGuassianPrior(GaussianPrior):

    cdef:
        dict _mean
        float _var
        float _default_mean

