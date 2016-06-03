import grasp.ptypes as ptypes
import numpy as np
cimport numpy as np


cdef class SGD:

    cpdef update(self, parameters, gradients):
        pass


cdef class FlatLearningRateSGD(SGD):

    def __init__(self, float gamma0, size_t t=0):
        """

        :param gamma0: initial learning rate
        :param t: time step
        """
        self._gamma0 = gamma0
        self._t = t

    cpdef update(self, parameters, gradients):
        self._t += 1
        return parameters + self._gamma0 * gradients


cdef class DecayingLearningRateSGD(SGD):

    def __init__(self, prior_variance, float gamma0, size_t t=0):
        """

        :param prior variance:
        :param gamma0: initial learning rate
        :param t: time step
        """
        self._variances = prior_variance
        self._t = t
        self._gamma0 = gamma0

    cpdef update(self, parameters, gradients):
        cdef size_t t = self._t
        cdef float gamma0 = self._gamma0
        cdef float var
        gamma = np.array([gamma0 / (1.0 + gamma0 * (1.0 / var) * t) for var in self._variances], dtype=ptypes.weight)
        self._t += 1
        return parameters + gamma * gradients


cdef class AdaGrad(SGD):

    def __init__(self, squared_gradient, float gamma0, size_t t=0, float epsilon=1e-6):
        self._squared_gradient = squared_gradient
        self._gamma0 = gamma0
        self._t = t
        self._epsilon = epsilon

    cpdef update(self, parameters, gradients):
        self._squared_gradient += np.square(gradients)
        self._t += 1
        return parameters + self._gamma0 * gradients / (np.sqrt(self._squared_gradient) + self._epsilon)


cdef class AdaDelta(SGD):

    def __init__(self, squared_gradient, size_t t=0, float epsilon=1e-6, float rho=0.95):
        self._squared_gradient = squared_gradient
        self._squared_delta = np.zeros(self._squared_gradient.shape[0], dtype=ptypes.weight)
        self._t = t
        self._epsilon = epsilon
        self._rho = rho

    cpdef update(self, parameters, gradients):
        self._squared_gradient = self._rho * self._squared_gradient + (1 - self._rho) * np.square(gradients)
        delta = np.sqrt(self._squared_delta + self._epsilon) / np.sqrt(self._squared_gradient + self._epsilon) * gradients
        self._squared_delta = self._rho * self._squared_delta + (1 - self._rho) * np.square(delta)
        self._t += 1
        return parameters + delta


