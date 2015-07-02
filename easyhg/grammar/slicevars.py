"""
This implements slice variables to be used in MCMC for CFGs as described in (Blunsom and Cohn, 2010).

:Authors: - Wilker Aziz
"""

import numpy as np
import scipy.stats as st
from collections import defaultdict

class SliceVariables(object):
    """
    Slice variables are indexed by nodes (typically chart cells).
    They are resampled on the basis of a conditioning derivation (which is given) or on the basis of a Beta distribution (whose parameters are given).

    This follows the idea in (Blunsom and Cohn, 2010):

        u_s ~ p(u_s|d) = uniform(0, theta_{r_s}) if r_s in d else beta(u_s; a, b)
        where theta_{r_s} is the parameter associated with a rule in d whose LHS corresponds to the index s.

    """

    def __init__(self, conditions={}, a=1, b=1, heuristic=None, mask=lambda s: s[0]):
        """
        :param conditions: 
            dict of conditioning parameters, i.e., maps an index s to a parameter theta_{r_s}.
        :param a: 
            first shape parameter of the Beta distribution.
        :param b: 
            second shape parameter of the Beta distribution.
        :param preconditions:
            heuristic preconditions that apply to items which are equivalent under a certain mask.
        :param mask:
            a function which takes an item's signature and return a key in preconditions.
        """
        self._conditions = defaultdict(None, conditions)
        self._a = a
        self._b = b
        self._u = defaultdict(None)
        self._heuristic = heuristic
        self._mask = mask

    @property
    def conditions(self):
        return self._conditions

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b


    def pr(self, s, theta, slice_only=True):
        """returns p(r_s|u) where lhs(r) == s and theta = p(rhs(r)|s)"""
        u_s = self[s]
        if u_s < theta:
            return 1.0 / st.beta.pdf(u_s, self._a, self._b)  # (np.power(u_s, self._a - 1) * np.power(1 - u_s, self._b - 1))
        elif slice_only:
            raise ValueError('I received a variable outside the slice: s=%s theta=%s u_s=%s' % (s, theta, u_s))
        else:
            return 0.0

    def logpr(self, s, theta, slice_only=True):
        """returns p(r_s|u) where lhs(r) == s and theta = p(rhs(r)|s)"""
        u_s = self[s]
        if theta > u_s:
            return - st.beta.logpdf(u_s, self._a, self._b)
        elif slice_only:
            raise ValueError('I received a variable outside the slice: s=%s theta=%s u_s=%s' % (s, theta, u_s))
        else:
            return 0.0


    def reset(self, conditions=None, a=None, b=None):
        self._u = defaultdict(None)
        if conditions is not None:
            self._conditions = defaultdict(None, conditions)
            self._heuristic = None  # as soon as we get real conditions, heuristics are discarded for good
        if a is not None:
            self._a = a
        if b is not None:
            self._b = b

    def __getitem__(self, s):
        """
        Returns u_s sampling it if necessary.

        >>> d = {(0, 'X', 1): 0.6, (1, 'X', 2): 0.5, (0, 'X', 2): 0.5, (2, 'X', 3): 0.3, (3, 'X', 4):0.7, (2, 'X', 4):0.9, (0, 'S', 4): 1.0}
        >>> u = SliceVariables(d, 5, 1)
        >>> u[(0, 'X', 2)] < d[(0,'X',2)]
        True
        >>> u[(2, 'X', 4)] < d[(2,'X',4)]
        True
        >>> u[(0, 'S', 4)] < d[(0,'S',4)]
        True
        """

        u_s = self._u.get(s, None)
        if u_s is None:  # the slice variable has not been sampled yet
            theta_r_s = self._conditions.get(s, None)  # so we check if we have observed real conditions
            if theta_r_s is None: # there is no r in d for which lhs(r) == s
                u_s = np.random.beta(self._a, self._b)  # in this case we sample u_s from a beta
            else:  # theta_r_s is the parameter associated with r in d for which lhs(r) == s
                u_s = np.random.uniform(0, theta_r_s)  # in this case we sample uniformly from [0, theta_r_s)
            self._u[s] = u_s
        return u_s

    def __getitem__2(self, s):
        """
        Returns u_s sampling it if necessary.

        >>> d = {(0, 'X', 1): 0.6, (1, 'X', 2): 0.5, (0, 'X', 2): 0.5, (2, 'X', 3): 0.3, (3, 'X', 4):0.7, (2, 'X', 4):0.9, (0, 'S', 4): 1.0}
        >>> u = SliceVariables(d, 5, 1)
        >>> u[(0, 'X', 2)] < d[(0,'X',2)]
        True
        >>> u[(2, 'X', 4)] < d[(2,'X',4)]
        True
        >>> u[(0, 'S', 4)] < d[(0,'S',4)]
        True
        """

        u_s = self._u.get(s, None)
        if u_s is None:  # the slice variable has not been sampled yet
            # u_s will depend on whether or not we are conditioning on some theta_r_s
            theta_r_s = None
            if self._heuristic is None:  # we are not employing heuristics
                theta_r_s = self._conditions.get(s, None)  # so we check if we have observed real conditions 
            else:  # we might sample conditions from some heuristic distribution
                dist = self._heuristic.get(self._mask(s), None)
                if dist is None:  # no known heuristic for this cell
                    theta_r_s = None
                else:
                    theta_r_s = dist.sample()  # draw a condition 
            if theta_r_s is None: # there is no r in d for which lhs(r) == s
                u_s = np.random.beta(self._a, self._b)  # in this case we sample u_s from a beta
            else:  # theta_r_s is the parameter associated with r in d for which lhs(r) == s
                u_s = np.random.uniform(0, theta_r_s)  # in this case we sample uniformly from [0, theta_r_s)
            self._u[s] = u_s
        return u_s

    def is_inside(self, s, theta):
        return theta > self[s]

    def is_outside(self, s, theta):
        return theta <= self[s]


class GeneralisedSliceVariables(object):
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
            a = parameters['a']
            b = parameters['b']
            get_random = lambda: np.random.beta(a, b)
            pdf = lambda x: st.beta.pdf(x, a, b)
            logpdf = lambda x: st.beta.logpdf(x, a, b)
        elif distribution == 'exp' or distribution == 'exponential':
            scale = 1.0 / parameters['rate']
            get_random = lambda: np.random.exponential(scale)
            pdf = lambda x: st.expon.pdf(x, scale=scale)
            logpdf = lambda x: st.expon.logpdf(x, scale=scale)
        elif distribution == 'gamma':
            shape = parameters['shape']
            scale = parameters['scale']
            get_random = lambda: np.random.gamma(shape, scale)
            pdf = lambda x: st.gamma.pdf(x, shape, scale=scale)
            logpdf = lambda x: st.gamma.logpdf(x, shape, scale=scale)
        else:
            raise ValueError("I don't know this distribution: %s" % distribution)
        return get_random, pdf, logpdf

    def __init__(self, conditions={}, distribution='beta', parameters={'a': 0.1, 'b': 1.0, 'rate': 1, 'shape': 1, 'scale': 1.0}):
        """
        :param conditions:
            dict of conditioning parameters, i.e., maps an index s to a parameter theta_{r_s}.
        :param distribution:
            a distribution for the free slice variables
                - it defaults to the Beta distribution as in (Blunsom and Cohn, 2010) which is convenient for PCFGs
                - use 'exponential' or 'gamma' for wCFGs parameterised with log-linear models (e.g. MT)
        :param parameters:
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
        self._random, self._pdf, self._logpdf = GeneralisedSliceVariables.get_distribution(distribution, parameters)

    def __str__(self):
        if self._distribution == 'beta':
            return 'Beta(a={0}, b={1})'.format(self._parameters['a'], self._parameters['b'])
        elif self._distribution in {'exp', 'exponential'}:
            return 'Exponential(rate={0})'.format(self._parameters['rate'])
        elif self._distribution == 'gamma':
            return 'Gamma(shape={0}, scale={1})'.format(self._parameters['shape'], self._parameters['scale'])

    def pr(self, s, theta, slice_only=True):
        """returns p(r_s|u) where lhs(r) == s and theta = p(rhs(r)|s)"""
        u_s = self[s]
        if u_s < theta:
            return 1.0 / self._pdf(u_s)
        elif slice_only:
            raise ValueError('I received a variable outside the slice: s=%s theta=%s u_s=%s' % (s, theta, u_s))
        else:
            return 0.0

    def logpr(self, s, theta, slice_only=True):
        """returns p(r_s|u) where lhs(r) == s and theta = p(rhs(r)|s)"""
        u_s = self[s]
        if theta > u_s:
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
            changes = parameters.items() - self._parameters.items()
            if changes:
                for k, v in changes:
                    self._parameters[k] = v
                update = True

        # consolidate changes if necessary
        if update:
            self._random, self._pdf, self._logpdf = GeneralisedSliceVariables.get_distribution(self._distribution,
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
        return theta > self[s]

    def is_outside(self, s, theta):
        """Whether the node s is off the slice."""
        return theta <= self[s]