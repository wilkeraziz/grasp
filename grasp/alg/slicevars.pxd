from grasp.ptypes cimport weight_t
from grasp.semiring._semiring cimport Semiring
cimport numpy as np
import numpy as np


cdef class Prior:
    """
    A prior on the parameters of the distribution associated with unconstrained slice variables.
    """

    cpdef weight_t[::1] sample(self, s)

    cpdef reset(self, args)


cdef class ConstantPrior(Prior):
    """
    A constant prior is typically used for the scale of the Exponential distribution,
    and for second shape parameter of the Beta distribution.
    """

    cdef weight_t[::1] _const


cdef class BetaPrior(Prior):

    cdef weight_t _a, _b


cdef class SymmetricGamma(Prior):
    """
    A symmetric Gamma prior is typically used for the scale of the Exponential distribution.
    """

    cdef weight_t _scale


cdef class AsymmetricGamma(Prior):
    """
    An asymmetric Gamma prior is typically used for the scale parameter of the Exponential distribution.
    """

    cdef dict _scales


cdef class VectorOfPriors(Prior):
    """
    The Beta distribution has two parameters, their independent priors can be handled as a single object with
    this wrapper.
    """

    cdef list _priors


cdef class Distribution:
    """
    A distribution has to provide 3 simple methods (see below).
    """

    cpdef weight_t sample(self, weight_t[::1] parameters)

    cpdef weight_t pdf(self, weight_t x, weight_t[::1] parameters)

    cpdef weight_t logpdf(self, weight_t x, weight_t[::1] parameters)


cdef class Exponential(Distribution):
    """
    Exponential(scale)
        - the scale is the inverse of the rate.
    """

    pass


cdef class Beta(Distribution):
    """
    Beta(a, b)
    """

    pass


cdef class SliceVariables:

    cpdef bint is_inside(self, key, p)

    cpdef bint is_outside(self, key, theta)

    cpdef bint has_conditions(self, key)

    cpdef bint is_greater(self, weight_t theta, weight_t u)

    cpdef get_assignment(self, key)

    cpdef weight_t pdf(self, key, weight_t theta, bint slice_only=?) except? -999999

    cpdef weight_t logpdf(self, key, weight_t theta, bint slice_only=?) except? -999999

    cpdef weight_t pdf_semiring(self, s, weight_t theta, Semiring semiring, bint slice_only=?) except? -999999

    cpdef reset(self, conditions=?, parameter=?)


cdef class SpanSliceVariables(SliceVariables):
    """
    A general slice variable.

    It is basically a vector of independent slice variables which are lazily assigned.
    We need a distribution for the unconstrained variables (e.g. Exponential, Beta).
    This distribution typically has parameters drawn from a certain prior (e.g. Constant, Gamma).
    """

    cdef object _assignments
    cdef object _conditions
    cdef Distribution _dist
    cdef Prior _prior
    cdef object _constr


cdef class ExpSpanSliceVariables(SpanSliceVariables):

    pass


cdef class GammaSpanSliceVariables(SpanSliceVariables):

    pass



cdef class BetaSpanSliceVariables(SpanSliceVariables):

    pass


cpdef Prior get_prior(prior_type, prior_parameter)


cpdef Distribution get_distribution(dist_type)

