import numpy as np
cimport numpy as np
import grasp.ptypes as ptypes


cdef class GaussianPrior:

    cpdef float mean(self, key):
        return 0.0

    cpdef float var(self):
        return 1.0

    cpdef weight_t[::1] mean_vector(self, keys):
        return np.ones(len(keys), dtype=ptypes.weight)

cdef class SymmetricGuassianPrior(GaussianPrior):

    def __init__(self, float mean=0.0, float var=1.0):
        self._mean = mean
        self._var = var

    cpdef float mean(self, key):
        return self._mean

    cpdef float var(self):
        return self._var

    cpdef weight_t[::1] mean_vector(self, keys):
        return np.full(len(keys), self._mean)


cdef class AsymmetricGuassianPrior(GaussianPrior):

    def __init__(self, dict mean={}, float var=1.0, float default_mean=0.0):
        self._default_mean = default_mean
        self._var = var
        self._mean = mean

    cpdef float mean(self, key):
        return self._mean.get(key, self._default_mean)

    cpdef float var(self):
        return self._var

    cpdef weight_t[::1] mean_vector(self, keys):
        return np.array([self.mean(key) for key in keys], dtype=ptypes.weight)
