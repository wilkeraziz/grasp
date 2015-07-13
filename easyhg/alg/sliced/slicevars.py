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


class SliceVariables(object):
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
                 parameters=DEFAULT_FREE_DIST_PARAMETERS):
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
                u_s = self._random()  # in this case we sample u_s from the free distribution
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