"""
This implements slice variables to be used in MCMC for CFGs as described in (Blunsom and Cohn, 2010).

@author wilkeraziz
"""

import numpy as np
from scipy.stats import beta
from collections import defaultdict

class SliceVariables(object):
    """
    Slice variables are indexed by nodes (typically chart cells).
    They are resampled on the basis of a conditioning derivation (which is given) or on the basis of a Beta distribution (whose parameters are given).

    This follows the idea in (Blunsom and Cohn, 2010):

        u_s ~ p(u_s|d) = uniform(0, theta_{r_s}) if r_s in d else beta(u_s; a, b)
        where theta_{r_s} is the parameter associated with a rule in d whose LHS corresponds to the index s.

    """

    def __init__(self, conditions={}, a=1, b=1):
        """
        @param conditions: dict of conditioning parameters, i.e., maps an index s to a parameter theta_{r_s}
        @param a: shape parameter of the Beta distribution
        @param b: shape parameter of the Beta distribution
        """
        self._conditions = defaultdict(None, conditions)
        self._a = a
        self._b = b
        self._u = defaultdict(None)

    @property
    def d(self):
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
            return 1.0 / (np.power(u_s, self.a - 1) * np.power(1 - u_s, self.b - 1)) 
        elif slice_only:
            raise ValueError('I received a variable outside the slice: s=%s theta=%s u_s=%s' % (s, theta, u_s))
        else:
            return 0.0

    def reset(self, conditions=None, a=None, b=None):
        self._u = defaultdict(None)
        if conditions is not None:
            self._conditions = defaultdict(None, conditions)
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
        if u_s is None:
            theta_r_s = self.d.get(s, None)
            if theta_r_s is None: # there is no r in d for which lhs(r) == s
                u_s = np.random.beta(self.a, self.b)  # in this case we sample u_s from a beta
            else:  # theta_r_s is the parameter associated with r in d for which lhs(r) == s
                u_s = np.random.uniform(0, theta_r_s)  # in this case we sample uniformly from [0, theta_r_s)
            self._u[s] = u_s
        return u_s
