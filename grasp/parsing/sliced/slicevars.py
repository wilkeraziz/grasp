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

import numpy as np
import scipy.stats as st
from collections import defaultdict
from types import SimpleNamespace
from grasp.parsing.exact.deduction import SliceVariables as _SliceVariables


class Prior(object):
    """
    A prior on the parameters of the distribution associated with unconstrained slice variables.
    """

    def sample(self, s):
        pass

    def reset(self, parameter):
        pass


class ConstantPrior(Prior):
    """
    A constant prior is typically used for the scale of the Exponential distribution,
    and for second shape parameter of the Beta distribution.
    """

    def __init__(self, const):
        self._const = const

    def sample(self, s):
        return self._const

    def reset(self, const):
        self._const = const

    def __repr__(self):
        return 'ConstantPrior(%r)' % repr(self._const)


class BetaPrior(Prior):

    def __init__(self, a, b):
        self._a = a
        self._b = b

    def sample(self, s):
        return np.random.beta(self._a, self._b)

    def reset(self, parameter):
        self._a = parameter[0]
        self._b = parameter[1]

    def __repr__(self):
        return 'BetaPrior(%r, %r)' % (self._a, self._b)


class SymmetricGamma(Prior):
    """
    A symmetric Gamma prior is typically used for the scale of the Exponential distribution.
    """

    def __init__(self, scale):
        self._scale = scale

    def sample(self, s):
        return np.random.gamma(shape=1, scale=self._scale)

    def reset(self, scale):
        self._scale = scale

    def __repr__(self):
        return 'SymmetricGamma(%r)' % repr(self._scale)


class AsymmetricGamma(Prior):
    """
    An asymmetric Gamma prior is typically used for the scale parameter of the Exponential distribution.
    """

    def __init__(self, scales):
        self._scales = scales

    def sample(self, s):
        return np.random.gamma(shape=1, scale=self._scales[s])

    def reset(self, scales):
        self._scales = scales

    def __repr__(self):
        return 'AsymmetricGamma(...)'


class VectorOfPriors(Prior):
    """
    The Beta distribution has two parameters, their independent priors can be handled as a single object with
    this wrapper.
    """

    def __init__(self, *priors):
        self._priors = priors

    def sample(self, s):
        return [prior.sample(s) for prior in self._priors]

    def reset(self, parameters):
        for prior, parameter in zip(self._priors, parameters):
            prior.reset(parameter)

    def __repr__(self):
        return 'VectorOfPriors(%s)' % ', '.join(repr(p) for p in self._priors)


class Distribution(object):
    """
    A distribution has to provide 3 simple methods (see below).
    """

    @classmethod
    def sample(cls, parameter):
        pass

    @classmethod
    def pdf(cls, x, parameter):
        pass

    @classmethod
    def logpdf(cls, x, parameter):
        pass


class Exponential(Distribution):
    """
    Exponential(scale)
        - the scale is the inverse of the rate.
    """

    @classmethod
    def sample(cls, parameter):
        return np.random.exponential(scale=parameter)

    @classmethod
    def pdf(cls, x, parameter):
        return st.expon.pdf(x, scale=parameter)

    @classmethod
    def logpdf(cls, x, parameter):
        return st.expon.logpdf(x, scale=parameter)


class Beta(Distribution):
    """
    Beta(a, b)
    """

    @classmethod
    def sample(cls, parameter):
        return np.random.beta(a=parameter[0], b=parameter[1])

    @classmethod
    def pdf(cls, x, parameter):
        return st.beta.pdf(x, a=parameter[0], b=parameter[1])

    @classmethod
    def logpdf(cls, x, parameter):
        return st.beta.logpdf(x, a=parameter[0], b=parameter[1])


class SliceVariables(_SliceVariables):
    """
    A general slice variable.

    It is basically a vector of independent slice variables which are lazily assigned.
    We need a distribution for the unconstrained variables (e.g. Exponential, Beta).
    This distribution typically has parameters drawn from a certain prior (e.g. Constant, Gamma).
    """

    def __init__(self, conditions, dist, prior):
        """
        :param conditions:
        :param dist: an instance of Distribution
        :param prior: an instance of Prior
        """
        self._conditions = defaultdict(None, conditions)
        self._assignments = defaultdict(lambda: SimpleNamespace(u=None, parameter=None))
        self._dist = dist
        self._prior = prior


    @staticmethod
    def is_greater(theta, u):
        """
        We define a specific way to compare two real numbers.
        This is basically to deal with some corner cases.
        :param theta: edge's weight
        :param u: random threshold
        :return: theta > u
        """
        if theta == u == 0:  # corner case to deal with floating point precision
            return True
        else:
            return theta > u

    def get_assignment(self, s):
        """
        Retrieve the assignment (u_s, \lambda_s) sampling it if necessary.
            \lambda_s is defined by _sample_exp_rate(s)
            u_s is sampled from
                U(0, \theta_r_s) if r_s is a condition
                Exponential(u_s; \lambda_s) otherwise

        :param s: slice variable
        :return: SimpleNamespace(u, exp_rate)
        """
        assignment = self._assignments[s]
        if assignment.u is None:  # we need to sample an assignment of the slice variable
            condition = self._conditions.get(s, None)
            assignment.parameter = self._prior.sample(s)
            if condition is None:  # and there are no conditions
                assignment.u = self._dist.sample(assignment.parameter)
            else:  # we have a condition, thus we sample uniformly
                assignment.u = np.random.uniform(0, condition)
        return assignment

    def __getitem__(self, s):
        """Return u_s."""
        return self.get_assignment(s).u

    def pdf(self, s, theta, slice_only=True):
        """Return p(r_s|u) where lhs(r) == s and theta = p(rhs(r)|s)."""
        assignment = self.get_assignment(s)
        if SliceVariables.is_greater(theta, assignment.u):  # theta > u_s
            return 1.0 / self._dist.pdf(assignment.u, assignment.parameter)
        elif slice_only:
            raise ValueError('Variable outside the slice: s=%s theta=%s u=%s param=%s' % (s, theta,
                                                                                          assignment.u,
                                                                                          assignment.parameter))
        else:
            return 0.0

    def logpdf(self, s, theta, slice_only=True):
        """Return p(r_s|u) where lhs(r) == s and theta = p(rhs(r)|s)."""
        assignment = self.get_assignment(s)
        if SliceVariables.is_greater(theta, assignment.u):  # theta > u_s
            return - self._dist.logpdf(assignment.u, assignment.parameter)
        elif slice_only:
            raise ValueError('Variable outside the slice: s=%s theta=%s u=%s param=%s' % (s, theta,
                                                                                          assignment.u,
                                                                                          assignment.parameter))
        else:
            return -np.infty

    def pdf_semiring(self, s, x, semiring, slice_only=True):
        """
        :param s: the slice variable
        :param x: value in semiring
        :param semiring: given semiring
        :param slice_only: whether we throw exceptions for values outside the slice
        :return: pdf(u_s) converted to the given semiring
        """
        assignment = self.get_assignment(s)
        if semiring.gt(x, semiring.from_real(assignment.u)):
            if semiring.LOG:
                return - self._dist.logpdf(assignment.u, assignment.parameter)
            else:
                return semiring.from_real(1.0 / self._dist.pdf(assignment.u, assignment.parameter))
        elif slice_only:
            raise ValueError('Variable outside the slice: s=%s theta=%s u=%s parameter=%s' % (s, semiring.as_real(x),
                                                                                              assignment.parameter))
        else:
            return semiring.zero

    def reset(self, conditions=None, parameter=None):
        if conditions is not None:
            self._conditions = defaultdict(None, conditions)
        if parameter is not None:
            self._prior.reset(parameter)
        self._assignments = defaultdict(lambda: SimpleNamespace(u=None, parameter=None))

    def is_inside(self, s, theta):
        """Whether the node s is in the slice."""
        return SliceVariables.is_greater(theta, self[s])

    def is_outside(self, s, theta):
        """Whether the node s is off the slice."""
        return not SliceVariables.is_greater(theta, self[s])

    def has_conditions(self, s):
        return s in self._conditions


class ExpSliceVariables(SliceVariables):

    def __init__(self, conditions, prior):
        super(ExpSliceVariables, self).__init__(conditions, dist=Exponential, prior=prior)


class BetaSliceVariables(SliceVariables):

    def __init__(self, conditions, prior_a, prior_b):
        super(BetaSliceVariables, self).__init__(conditions,
                                                 dist=Beta,
                                                 prior=VectorOfPriors(prior_a, prior_b))


def get_prior(prior_type, prior_parameter):
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


def get_distribution(dist_type):
    if dist_type == 'beta':
        return Beta
    if dist_type == 'exponential':
        return Exponential
    raise ValueError('I do not know this distribution: %s' % dist_type)