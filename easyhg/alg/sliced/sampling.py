"""
This module is an interface for parsing as intersection.
One can choose from all available implementations.

:Authors: - Wilker Aziz
"""

import logging
import numpy as np
import itertools
from collections import defaultdict, Counter
from types import SimpleNamespace

from easyhg.grammar.symbol import Nonterminal
from easyhg.grammar.semiring import SumTimes, Count
from easyhg.grammar.cfg import TopSortTable
from easyhg.grammar.result import Result

from easyhg.alg.exact.inference import robust_inside, sample, total_weight
from easyhg.alg.exact import Nederhof
from easyhg.alg.sliced.slicevars import Beta, VectorOfPriors, ConstantPrior, SliceVariables

from .heuristic import attempt_initialisation
from .utils import make_batch_conditions


def slice_sampling(fsa, grammars, glue_grammars,
                   root,
                   N,
                   lag=1,
                   burn=0,
                   batch=1,
                   report_counts=False,
                   goal=Nonterminal('GOAL'),
                   generations=10,
                   free_dist=Beta,
                   free_dist_prior=VectorOfPriors(ConstantPrior(0.1), ConstantPrior(1.0)),
                   semiring=SumTimes):
    # configure slice variables
    u = SliceVariables({}, free_dist, free_dist_prior)
    logging.debug('%s prior=%r', free_dist.__name__, free_dist_prior)

    # attempt a heuristic initialisation
    conditions = None  #attempt_initialisation(fsa, grammars, glue_grammars, options)
    if conditions is not None:  # reconfigure slice variables if possible
        u.reset(conditions)

    report_interval = 10
    checkpoint = report_interval
    history = []
    n_samples = 0
    lag_i = lag

    while n_samples < N + burn:
        # create a bottom-up parser with slice variables
        parser = Nederhof(grammars, fsa,
                          glue_grammars=glue_grammars,
                          semiring=semiring,
                          slice_variables=u)

        # compute a slice (a randomly pruned forest)
        logging.debug('Computing slice...')
        forest = parser.do(root=root, goal=goal)
        if not forest:
            logging.debug('NO PARSE FOUND')
            u.reset()  # reset the slice variables (keeping conditions unchanged if any)
            continue

        logging.debug('Top-sorting...')
        tsort = TopSortTable(forest)

        if report_counts:  # reporting counts
            Ic = robust_inside(forest,
                               tsort,
                               Count,
                               omega=lambda e: Count.convert(e.weight, semiring),
                               infinity=generations)
            logging.info('Done! Forest: %d edges, %d nodes and %d paths' % (len(forest),
                                                                            forest.n_nonterminals(),
                                                                            Ic[tsort.root()]))
        else:
            logging.debug('Done! Forest: %d edges, %d nodes' % (len(forest), forest.n_nonterminals()))

        logging.debug('Computing inside weights...')
        uniformdist = parser.reweight(forest)
        Iv = robust_inside(forest,
                           tsort,
                           semiring,
                           omega=lambda e: uniformdist[e],
                           infinity=generations)

        logging.debug('Sampling...')
        D = list(sample(forest,
                        tsort.root(),
                        semiring,
                        Iv=Iv,
                        N=batch,
                        omega=lambda e: uniformdist[e]))
        assert D, 'The slice should never be empty'

        u.reset(make_batch_conditions(D, semiring))

        history.append(D)

        # collect samples respecting a given lag
        lag_i -= 1
        if lag_i == 0:
            n_samples += len(D)
            lag_i = lag

        if n_samples > checkpoint:
            logging.info('sampling... %d/%d', n_samples, N + burn)
            checkpoint = n_samples + report_interval

    return history


def make_result(batches, lag=1, burn=0, resample=0):

    if lag > 1:
        samples = list(itertools.chain(*batches[lag-1::lag]))
    else:
        samples = list(itertools.chain(*batches))

    # compile results
    if resample > 0:
        # uniformly resample (with replacement) a number of derivations (possibly burning the initial ones)
        count = Counter(samples[i] for i in np.random.randint(burn, len(samples), resample))
    else:
        count = Counter(samples[burn:])

    result = Result()
    for d, n in count.most_common():
        score = total_weight(d, SumTimes)
        result.append(d, n, score)

    return result


def make_result_simple(samples, burn=0, lag=1, resample=0):
    if burn > 0:
        samples = samples[burn:]
    if lag > 1:
        samples = samples[lag-1::lag]

    # compile results
    if resample > 0:
        count = Counter(samples[i] for i in np.random.randint(0, len(samples), resample))
    else:
        count = Counter(samples)

    result = Result()
    for d, n in count.most_common():
        score = total_weight(d, SumTimes)
        result.append(d, n, score)

    return result


def apply_filters(markov_chain, burn=0, lag=1, resample=0):
    if burn > 0:
        markov_chain = markov_chain[burn:]
    if lag > 1:
        markov_chain = markov_chain[lag-1::lag]
    if resample:
        markov_chain = [markov_chain[i] for i in np.random.randint(0, len(markov_chain), resample)]
    return markov_chain


def group_by_projection(samples, get_projection=None):
    if get_projection is None:
        counts = Counter(samples)
        output = []
        for d, n in counts.most_common():
            output.append(SimpleNamespace(projection=d, count=n, derivations=1))
        return output
    else:
        p2d = defaultdict(set)
        counts = Counter()
        for d in samples:
            y = get_projection(d)
            counts[y] += 1
            p2d[y].add(d)
        output = []
        for y, n in counts.most_common():
            output.append(SimpleNamespace(projection=y, count=n, derivations=len(p2d[y])))
        return output