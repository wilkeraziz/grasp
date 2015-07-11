"""
:Authors: - Wilker Aziz
"""

from collections import defaultdict
from types import SimpleNamespace

DEFAULT_FREE_DISTRIBUTION = 'beta'

DEFAULT_FREE_DIST_PARAMETERS = SimpleNamespace(a=[0.1, 0.2],
                                               b=[1.0, 1.0],
                                               rate=[1e-4, 1e-5],
                                               shape=[1.0, 1.0],
                                               scale=[1e4, 1e5])


def make_conditions(d, semiring):
    conditions = {r.lhs.label: semiring.as_real(r.weight) for r in d}
    return conditions


def make_batch_conditions(D, semiring):
    if len(D) == 1:
        d = D[0]
        conditions = {r.lhs.label: semiring.as_real(r.weight) for r in d}
    else:
        conditions = defaultdict(set)
        for d in D:
            [conditions[r.lhs.label].add(semiring.as_real(r.weight)) for r in d]
        conditions = {s: min(thetas) for s, thetas in conditions.items()}
    return conditions


def choose_parameters(params, i=0):
    return SimpleNamespace(a=params.a[i],
                           b=params.b[i],
                           rate=params.rate[i],
                           shape=params.shape[i],scale=params.scale[i])


def mcmc_options(samples, lag, burn, batch, resample):
    return SimpleNamespace(samples=samples, lag=lag, burn=burn, batch=batch, resample=resample)


def make_namespace(options):
    return SimpleNamespace(a=options.a,
                           b=options.b,
                           rate=options.rate,
                           shape=options.shape,
                           scale=options.scale)