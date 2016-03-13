"""
This implements slice variables to be used in MCMC for CFGs as described in (Blunsom and Cohn, 2010).


TODO:
    - Beta: sample parameters from a Gamma prior ?
        - a ~ G_a
        - b ~ G_b
    - Exponential: sample rate from a Gamma prior ?
        - rate ~ np.random.gamma(1, 1e-4)

:Authors: - Wilker Aziz
"""

from grasp.ptypes cimport weight_t
cimport numpy as np
import numpy as np
import scipy.stats as st
from collections import defaultdict
from types import SimpleNamespace
import grasp.ptypes as ptypes


cdef class Prior:
    """
    A prior on the parameters of the distribution associated with unconstrained slice variables.
    """

    cpdef weight_t[::1] sample(self, s):
        """
        Sample once from the prior. This typically returns an array of size one (but see VectorOfPriors).
        """
        raise NotImplementedError()

    cpdef reset(self, args):
        """
        Some priors are stateful, in which case sometimes one might want to reset its state (see See AssymetricGammaPrior)
        """
        raise NotImplementedError()


cdef class ConstantPrior(Prior):
    """
    A constant prior is typically used for the scale of the Exponential distribution,
    and for second shape parameter of the Beta distribution.
    """

    def __init__(self, weight_t value):
        """
        :param value: the constant value returned by this prior.
        """
        self._const = np.array([value], dtype=ptypes.weight)

    cpdef weight_t[::1] sample(self, args):
        """
        Return an array of size 1 with a constant value.
        """
        return self._const

    cpdef reset(self, args):
        """
        :param args: [constant]
        """
        self._const[0] = args[0]

    def __repr__(self):
        return 'ConstantPrior(%r)' % repr(self._const[0])


cdef class BetaPrior(Prior):

    def __init__(self, weight_t a, weight_t b):
        """
        :param a: Beta's first shape parameter
        :param b: Beta's second shape parameter
        """
        self._a = a
        self._b = b

    cpdef weight_t[::1] sample(self, s):
        """
        Return one draw from Beta(a, b).
        """
        return np.random.beta(self._a, self._b, size=1)

    cpdef reset(self, args):
        """
        :param args: [a, b]
        """
        self._a = args[0]
        self._b = args[1]

    def __repr__(self):
        return 'BetaPrior(%r, %r)' % (self._a, self._b)


cdef class SymmetricGamma(Prior):
    """
    A symmetric Gamma prior is typically used for the scale of the Exponential distribution.
    """

    def __init__(self, weight_t scale):
        """
        :param scale: Gamma's scale parameter
        """
        self._scale = scale

    cpdef weight_t[::1] sample(self, key):
        """
        Return one sample from Gamma(scale, shape=1). Because this is a symmetric Gamma, the key is irrelevant.
        """
        return np.random.gamma(shape=1, scale=self._scale, size=1)

    cpdef reset(self, args):
        """
        :param args: [scale]
        """
        self._scale = args[0]

    def __repr__(self):
        return 'SymmetricGamma(%r)' % repr(self._scale)


cdef class AsymmetricGamma(Prior):
    """
    An asymmetric Gamma prior is typically used for the scale parameter of the Exponential distribution.
    """

    def __init__(self, scales):
        """
        :param scales: dictionary of assymetric scales
        """
        self._scales = dict(scales)

    cpdef weight_t[::1] sample(self, key):
        """
        Return a draw from Gamma(scale=scales[key], shape=1)
        """
        return np.random.gamma(shape=1, scale=self._scales.get(key, 0.0), size=1)

    cpdef reset(self, args):
        """
        :param args: [(key0, scale0), (key1, scale1), ...]
        """
        self._scales = dict(args)

    def __repr__(self):
        return 'AsymmetricGamma(...)'


cdef class VectorOfPriors(Prior):
    """
    The Beta distribution has two parameters, their independent priors can be handled as a single object with
    this wrapper.
    """

    def __init__(self, *priors):
        for prior in priors:
            if isinstance(prior, VectorOfPriors):
                raise ValueError('VectorOfPriors only take simple Prior objects.')
        self._priors = list(priors)

    cpdef weight_t[::1] sample(self, key):
        """
        Return a vector containing one sample from each prior in the vector.
        """
        return np.array([(<Prior>prior).sample(key) for prior in self._priors], dtype=ptypes.weight).flatten()

    cpdef reset(self, args):
        """
        :param args: [args_0, args_1, ...] each args_i is a list of arguments for the respective prior.
        """
        for prior, arg in zip(self._priors, args):
            prior.reset(arg)

    def __repr__(self):
        return 'VectorOfPriors(%s)' % ', '.join(repr(p) for p in self._priors)


cdef class Distribution:
    """
    A distribution has to provide 3 simple methods (see below).
    """

    cpdef weight_t sample(self, weight_t[::1] parameters):
        """x ~ pdf(X; parameters)"""
        raise NotImplementedError()

    cpdef weight_t pdf(self, weight_t x, weight_t[::1] parameters):
        """pdf(x; parameters)."""
        raise NotImplementedError()

    cpdef weight_t logpdf(self, weight_t x, weight_t[::1] parameters):
        """log(pdf(x; parameters))"""
        raise NotImplementedError()


cdef class Exponential(Distribution):
    """
    Exponential(scale)
        - the scale is the inverse of the rate.
    """

    cpdef weight_t sample(self, weight_t[::1] parameters):
        """
        :param parameters: [scale]
        :returns: v ~ Exponential(X; scale)
        """
        return np.random.exponential(scale=parameters[0])

    cpdef weight_t pdf(self, weight_t x, weight_t[::1] parameters):
        """
        :param parameters: [scale]
        :returns: Exponential(x; scale)
        """
        return st.expon.pdf(x, scale=parameters[0])

    cpdef weight_t logpdf(self, weight_t x, weight_t[::1] parameters):
        """
        :param parameters: [scale]
        :returns: log(Exponential(x; scale))
        """
        return st.expon.logpdf(x, scale=parameters[0])


cdef class Beta(Distribution):
    """
    Beta(a, b)
    """

    cpdef weight_t sample(self, weight_t[::1] parameters):
        """
        :param parameters: [a, b]
        :returns: x ~ Beta(X; a, b)
        """
        return np.random.beta(a=parameters[0], b=parameters[1])

    cpdef weight_t pdf(self, weight_t x, weight_t[::1] parameters):
        """
        :param parameters: [a, b]
        :returns: Beta(x; a, b)
        """
        return st.beta.pdf(x, a=parameters[0], b=parameters[1])

    cpdef weight_t logpdf(self, weight_t x, weight_t[::1] parameters):
        """
        :param parameters: [a, b]
        :returns: log(Beta(x; a, b))
        """
        return st.beta.logpdf(x, a=parameters[0], b=parameters[1])


cdef class SliceVariables:

    cpdef bint is_inside(self, key, theta):
        """
        Whether a certain element belongs to the slice.
        :param key: index of slice variable
        :param theta: score of the element.
        :return: is_greater(theta, u[key])
        """
        raise NotImplementedError()

    cpdef bint is_outside(self, s, theta):
        """not is_inside(s, theta)"""
        raise NotImplementedError()

    cpdef bint has_conditions(self, key):
        """Whether there is a condition associated with u[key]"""
        raise NotImplementedError()

    cpdef bint is_greater(self, weight_t theta, weight_t u):
        """
        We define a specific way to compare two real numbers.
        This is basically to deal with some corner cases.
        :param theta: edge's weight
        :param u: random threshold associated with the edge's head
        :return: theta > u
        """
        if theta == u == 0:  # corner case to deal with floating point precision
            return True
        else:
            return theta > u

    cpdef get_assignment(self, key):
        """Return u[key] sampling an assignment for it if necessary."""
        raise NotImplementedError()

    cpdef weight_t pdf(self, key, weight_t theta, bint slice_only=True) except? -999999:
        """Return pdf(u[key]; parameters[key]) and possibly check whether an element whose weight is theta would belong to the slice."""
        raise NotImplementedError()

    cpdef weight_t logpdf(self, key, weight_t theta, bint slice_only=True) except? -999999:
        """Return log(pdf(u[key]; parameters[key])) and possibly check whether an element whose weight is theta would belong to the slice."""
        raise NotImplementedError()

    cpdef weight_t pdf_semiring(self, s, weight_t theta, Semiring semiring, bint slice_only=True) except? -999999:
        """
        Compute pdf(u[key]; parameters[key]) and possibly check whether an element whose weight is theta would belong to the slice.
        Additionally, the return value is converted to a given semiring.
        """
        raise NotImplementedError()

    cpdef reset(self, conditions=None, parameter=None):
        """Reset slice variables assignments and their priors' parameters."""
        raise NotImplementedError()


cdef class SpanSliceVariables(SliceVariables):
    """
    A general slice variable.

    It is basically a vector of independent slice variables which are lazily assigned.
    We need a distribution for the unconstrained variables (e.g. Exponential, Beta).
    This distribution typically has parameters drawn from a certain prior (e.g. Constant, Gamma).
    """

    def __init__(self, conditions, Distribution dist, Prior prior):
        """
        :param conditions:
        :param dist: an instance of Distribution
        :param prior: an instance of Prior
        """
        self._conditions = defaultdict(None, conditions)
        #print('SpanSliceVariable::conditions: %s' % str(self._conditions))
        self._assignments = defaultdict(lambda: SimpleNamespace(u=None, parameter=None))
        self._dist = dist
        self._prior = prior
        # constructor for empty assignments
        self._constr = lambda: SimpleNamespace(u=None, parameter=None)

    cpdef bint is_inside(self, key, theta):
        """Whether the node s is in the slice."""
        return self.is_greater(theta, self[key])

    cpdef bint is_outside(self, key, theta):
        """Whether the node s is off the slice."""
        return not self.is_greater(theta, self[key])

    cpdef bint has_conditions(self, key):
        return key in self._conditions

    cpdef get_assignment(self, key):
        """
        Retrieve the assignment (u_s, \lambda_s) sampling it if necessary.
            \lambda_s is defined by _sample_exp_rate(s)
            u_s is sampled from
                U(0, \theta_r_s) if r_s is a condition
                Exponential(u_s; \lambda_s) otherwise

        :param s: slice variable
        :return: SimpleNamespace(u, exp_rate)
        """
        assignment = self._assignments[key]
        if assignment.u is None:  # we need to sample an assignment of the slice variable
            condition = self._conditions.get(key, None)
            assignment.parameter = self._prior.sample(key)
            if condition is None:  # and there are no conditions
                assignment.u = self._dist.sample(assignment.parameter)
            else:  # we have a condition, thus we sample uniformly
                assignment.u = np.random.uniform(0, condition)
        return assignment

    def __getitem__(self, key):
        """Return u_s."""
        return self.get_assignment(key).u

    cpdef weight_t pdf(self, key, weight_t theta, bint slice_only=True) except? -999999:
        """Return p(r_s|u) where lhs(r) == s and theta = p(rhs(r)|s)."""
        assignment = self.get_assignment(key)
        if self.is_greater(theta, assignment.u):  # theta > u_s
            return 1.0 / self._dist.pdf(assignment.u, assignment.parameter)
        elif slice_only:
            raise ValueError('Variable outside the slice: s=%s theta=%s u=%s param=%s' % (key, theta,
                                                                                          assignment.u,
                                                                                          assignment.parameter))
            return -999999
        else:
            return 0.0

    cpdef weight_t logpdf(self, key, weight_t theta, bint slice_only=True) except? -999999:
        """Return p(r_s|u) where lhs(r) == s and theta = p(rhs(r)|s)."""
        assignment = self.get_assignment(key)
        if self.is_greater(theta, assignment.u):  # theta > u_s
            return - self._dist.logpdf(assignment.u, assignment.parameter)
        elif slice_only:
            raise ValueError('Variable outside the slice: s=%s theta=%s u=%s param=%s' % (key, theta,
                                                                                          assignment.u,
                                                                                          assignment.parameter))
            return -999999
        else:
            return -np.infty

    cpdef weight_t pdf_semiring(self, key, weight_t theta, Semiring semiring, bint slice_only=True) except? -999999:
        """
        :param s: the slice variable
        :param x: value in semiring
        :param semiring: given semiring
        :param slice_only: whether we throw exceptions for values outside the slice
        :return: pdf(u_s) converted to the given semiring
        """
        assignment = self.get_assignment(key)
        if semiring.gt(theta, semiring.from_real(assignment.u)):
            if semiring.LOG:
                return - self._dist.logpdf(assignment.u, assignment.parameter)
            else:
                return semiring.from_real(1.0 / self._dist.pdf(assignment.u, assignment.parameter))
        elif slice_only:
            raise ValueError('Variable outside the slice: s=%s theta=%s u=%s parameter=%s' % (key, semiring.as_real(theta),
                                                                                              assignment.parameter))
            return -999999
        else:
            return semiring.zero

    cpdef reset(self, conditions=None, parameter=None):
        if conditions is not None:
            self._conditions = defaultdict(None, conditions)
        if parameter is not None:
            self._prior.reset(parameter)
        self._assignments = defaultdict(self._constr)
        return 0


cdef class ExpSpanSliceVariables(SpanSliceVariables):

    def __init__(self, conditions, prior):
        super(ExpSpanSliceVariables, self).__init__(conditions, dist=Exponential(), prior=prior)


cdef class BetaSpanSliceVariables(SpanSliceVariables):

    def __init__(self, conditions, prior_a, prior_b):
        super(BetaSpanSliceVariables, self).__init__(conditions,
                                                     dist=Beta(),
                                                     prior=VectorOfPriors(prior_a, prior_b))


cpdef Prior get_prior(prior_type, prior_parameter):
    if prior_type == 'const':
        return ConstantPrior(float(prior_parameter))
    if prior_type == 'gamma' or prior_type == 'sym':
        return SymmetricGamma(float(prior_parameter))
    if prior_type == 'beta':
        params = prior_parameter.split(',')
        if len(params) != 2:
            raise ValueError('A beta prior requires two shape parameters separated by a comma: %s' % prior_parameter)
        return BetaPrior(float(params[0]), float(params[1]))
    raise ValueError('I do not know this prior: %s' % prior_type)


cpdef Distribution get_distribution(dist_type):
    if dist_type == 'beta':
        return Beta()
    if dist_type == 'exponential':
        return Exponential()
    raise ValueError('I do not know this distribution: %s' % dist_type)