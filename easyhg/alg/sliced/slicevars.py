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
from .utils import DEFAULT_FREE_DISTRIBUTION, DEFAULT_FREE_DIST_PARAMETERS
import logging
from types import SimpleNamespace


class _SliceVariables(object):
    """
    Slice variables are indexed by nodes (typically chart cells).
    They are resampled on the basis of a conditioning derivation (which is given) or on the basis of a Beta distribution (whose parameters are given).

    This follows the idea in (Blunsom and Cohn, 2010):

        u_s ~ p(u_s|d) = uniform(0, theta_{r_s}) if r_s in d else beta(u_s; a, b)
        where theta_{r_s} is the parameter associated with a rule in d whose LHS corresponds to the index s.

    """

    @staticmethod
    def get_distribution(distribution, parameters):
        """
        Returns a distribution with a given parameterisation.
        :param distribution: the name of the distribution, one of {beta, exponential, gamma}
        :param parameters:
            beta
                - a, b
            exponential
                - rate
            gamma:
                - shape, scale
        :return: a random number generator, the pdf, and the log-transformed pdf
        """
        get_random = None
        pdf = None
        logpdf = None
        if distribution == 'beta':
            a = parameters.a
            b = parameters.b
            get_random = lambda: np.random.beta(a, b)
            pdf = lambda x: st.beta.pdf(x, a, b)
            logpdf = lambda x: st.beta.logpdf(x, a, b)
        elif distribution == 'exp' or distribution == 'exponential':
            scale = 1.0 / parameters.rate
            get_random = lambda: np.random.exponential(scale)
            pdf = lambda x: st.expon.pdf(x, scale=scale)
            logpdf = lambda x: st.expon.logpdf(x, scale=scale)
        elif distribution == 'gamma':
            shape = parameters.shape
            scale = parameters.scale
            get_random = lambda: np.random.gamma(shape, scale)
            pdf = lambda x: st.gamma.pdf(x, shape, scale=scale)
            logpdf = lambda x: st.gamma.logpdf(x, shape, scale=scale)
        else:
            raise ValueError("I don't know this distribution: %s" % distribution)
        return get_random, pdf, logpdf

    def __init__(self, conditions={},
                 distribution=DEFAULT_FREE_DISTRIBUTION,
                 parameters=DEFAULT_FREE_DIST_PARAMETERS,
                 priors=None):
        """
        :param conditions:
            dict of conditioning parameters, i.e., maps an index s to a parameter theta_{r_s}.
        :param distribution:
            a distribution for the free slice variables
                - it defaults to the Beta distribution as in (Blunsom and Cohn, 2010) which is convenient for PCFGs
                - use 'exponential' or 'gamma' for wCFGs parameterised with log-linear models (e.g. MT)
        :param parameters: a SimpleNamespace
            parameters of the given distribution
                - 'a' and 'b' are the shape parameters of the Beta distribution
                - 'rate' is the parameter of the Exponential distribution
                - 'shape' and 'scale' are the parameters of the Gamma distribution
        """
        self._conditions = defaultdict(None, conditions)
        self._u = defaultdict(None)
        self._get_random = None
        self._pdf = None
        self._logpdf = None
        self._distribution = distribution
        self._parameters = parameters
        self._random, self._pdf, self._logpdf = SliceVariables.get_distribution(distribution, parameters)
        self._priors = priors
        self._rates = defaultdict(None)
        import logging
        logging.info('Mean sample: %s', np.array([self._random() for _ in range(1000)]).mean())

    def __str__(self):
        if self._distribution == 'beta':
            return 'Beta(a={0}, b={1})'.format(self._parameters.a, self._parameters.b)
        elif self._distribution in {'exp', 'exponential'}:
            return 'Exponential(rate={0})'.format(self._parameters.rate)
        elif self._distribution == 'gamma':
            return 'Gamma(shape={0}, scale={1})'.format(self._parameters.shape, self._parameters.scale)

    def pr(self, s, theta, slice_only=True):
        """returns p(r_s|u) where lhs(r) == s and theta = p(rhs(r)|s)"""
        u_s = self[s]
        if self.is_inside(s, theta):  #if u_s < theta:
            return 1.0 / self._pdf(u_s)
        elif slice_only:
            raise ValueError('I received a variable outside the slice: s=%s theta=%s u_s=%s' % (s, theta, u_s))
        else:
            return 0.0

    def logpr(self, s, theta, slice_only=True):
        """returns p(r_s|u) where lhs(r) == s and theta = p(rhs(r)|s)"""
        u_s = self[s]
        if self.is_inside(s, theta):  #if theta > u_s:
            return - self._logpdf(u_s)
        elif slice_only:
            raise ValueError('I received a variable outside the slice: s=%s theta=%s u_s=%s' % (s, theta, u_s))
        else:
            return 0.0

    def reset(self, conditions=None, distribution=None, parameters=None):
        """
        Reset random assignments of the slice variables.
        :param conditions: new conditions (dict) or None
        :param distribution: new distribution (str) or None
        :param parameters: new parameters (dict) or None
        """

        # reset the random assignments
        self._u = defaultdict(None)
        self._rates = defaultdict(None)

        # update conditions if necessary
        if conditions is not None:
            self._conditions = defaultdict(None, conditions)

        update = False

        # change distribution if necessary
        if distribution is not None and distribution != self._distribution:
            self._distribution = distribution
            update = True

        # change parameters if necessary
        if parameters is not None:
            self._parameters = parameters
            update = True
            #changes = vars(parameters).items() - vars(self._parameters).items()
            #if changes:
            #    for k, v in changes:
            #        self._parameters[k] = v
            #    update = True

        # consolidate changes if necessary
        if update:
            self._random, self._pdf, self._logpdf = SliceVariables.get_distribution(self._distribution,
                                                                                    self._parameters)

    def __getitem__(self, s):
        """
        Returns u_s sampling it if necessary.
        :param s: the index of the slice variable.
        :return: a random assignment of u_s.
        """

        u_s = self._u.get(s, None)
        if u_s is None:  # the slice variable has not been sampled yet
            theta_r_s = self._conditions.get(s, None)  # so we check if we have observed real conditions
            if theta_r_s is None:  # there is no r in d for which lhs(r) == s
                if self._priors is None:
                    u_s = self._random()  # in this case we sample u_s from the free distribution
                else:
                    rate = self._rates.get(s, None)
                    if rate is None:
                        rate = np.random.gamma(shape=1.0, scale=self._priors[s])
                        self._rates[s] = rate
                    u_s = np.random.exponential(rate)
                    #logging.info('s=%s theta=%s rate=%s u=%s', s, t, rate, u_s)
            else:  # theta_r_s is the parameter associated with r in d for which lhs(r) == s
                u_s = np.random.uniform(0, theta_r_s)  # in this case we sample uniformly from [0, theta_r_s)
            self._u[s] = u_s
        return u_s

    def is_inside(self, s, theta):
        """Whether the node s is in the slice."""
        #return theta > self[s]
        u = self[s]
        if theta == u == 0:  # corner case to deal with floating point precision
            return True
        else:
            return theta > u

    def is_outside(self, s, theta):
        """Whether the node s is off the slice."""
        return not self.is_inside(s, theta)  #theta < self[s]

    def has_conditions(self, s):
        return s in self._conditions


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


class XSliceVariables(object):
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
        if XSliceVariables.is_greater(theta, assignment.u):  # theta > u_s
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
        if XSliceVariables.is_greater(theta, assignment.u):  # theta > u_s
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
        return XSliceVariables.is_greater(theta, self[s])

    def is_outside(self, s, theta):
        """Whether the node s is off the slice."""
        return not XSliceVariables.is_greater(theta, self[s])

    def has_conditions(self, s):
        return s in self._conditions


class ExpSliceVariables(XSliceVariables):

    def __init__(self, conditions, prior):
        super(ExpSliceVariables, self).__init__(conditions, dist=Exponential, prior=prior)


class BetaSliceVariables(XSliceVariables):

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