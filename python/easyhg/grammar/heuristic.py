"""
This module implements heuristics to reduce control the size of the initial slice.

Heuristics: for each lhs symbol, returns
    - empdist:
        theta_r where r ~ p(rhs|lhs).
    - uniform:
        theta_r where r is sampled uniformly from the support of p(r|rhs).

:Authors: - Wilker Aziz
"""

import numpy as np
from collections import defaultdict

class EmpiricalDistribution(object):
    """
    Heuristically sample a condition (a value of theta) from the empirical distribution associated with the conditional p(rule|lhs).
    """

    def __init__(self, support, alpha=1.0):
        """
        :param support: list of real valued weights.
        :param alpha: value by which we are going to peak/flatten the distribution.
        """
        self._support = np.array(support, float)  # the weights we have observed
        self._support.sort()
        self._prob = self._support ** alpha  # we peak/flatten them
        self._prob /= self._prob.sum(0)  # and renormalise

    def sample(self):
        """Draw a theta from the empirical distribution: theta_r where r ~ p(r|lhs)."""
        return np.random.choice(self._support, p=self._prob)


class UniformDistribution(object):
    """
    Heuristically sample a condition (a value of theta) uniformly from the interval [lower, upper), 
    where lower and upper can be configured in terms of percentiles.
    """

    def __init__(self, support, a=0, b=100):
        """
        :param support: list of real valued weights.
        :param a: lower percentile
        :param b: upper percentile
        """
        support = np.array(support, float)  # the weights we have observed
        support.sort()
        self._lower = np.percentile(support, a)
        self._upper = np.percentile(support, b)

    def sample(self):
        """Draw a theta from the uniform distribution: theta_r where r ~ U(l, u) where u = max p(r|lhs) and l depends on configuration."""
        return np.random.uniform(self._lower, self._upper)


def empdist(cfg, semiring, alpha=1.0):
    """
    Map every lhs symbol to a distribution from which one will heuristically sample an initial condition.
    A condition is a threshold on the parameter theta_r for rules rewriting lhs.
    :param cfg: a CFG-like object.
    :param semiring: requires ``as_real``.
    :param alpha: scaling parameter.
    """
    distributions = defaultdict(None, 
            ((lhs, EmpiricalDistribution([semiring.as_real(r.weight) for r in rules], alpha)) 
                for lhs, rules in cfg.iteritems()))
    return distributions

def uniform(cfg, semiring, a, b):
    """
    Map every lhs symbol to a distribution from which one will heuristically sample an initial condition.
    A condition is a threshold on the parameter theta_r for rules rewriting lhs.
    :param cfg: a CFG-like object.
    :param semiring: requires ``as_real``.
    :param a: lower percentile.
    :param b: upper percentile.
    """
    distributions = defaultdict(None, 
            ((lhs, UniformDistribution([semiring.as_real(r.weight) for r in rules], a, b)) 
                for lhs, rules in cfg.iteritems()))
    return distributions
