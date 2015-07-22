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

from grasp.recipes import progressbar
from grasp.semiring import SumTimes
from grasp.cfg import Nonterminal, TopSortTable
from grasp.inference import AncestralSampler
from grasp.parsing.exact import Nederhof
from grasp.parsing.sliced.slicevars import Beta, VectorOfPriors, ConstantPrior, SliceVariables

from .heuristic import attempt_initialisation
from .utils import make_batch_conditions


def uninformed_conditions(grammars, glue_grammars, fsa, slicevars, root, goal, batch, generations, semiring):
    """
    Search for an initial set of conditions without any heuristics.

    :param grammars:
    :param glue_grammars:
    :param fsa:
    :param slicevars:
    :param root:
    :param goal:
    :param batch:
    :param generations:
    :param semiring:
    :return:
    """

    while True:
        parser = Nederhof(grammars, fsa,
                          glue_grammars=glue_grammars,
                          semiring=semiring,
                          slice_variables=slicevars)

        # compute a slice (a randomly pruned forest)
        logging.debug('Computing slice...')
        forest = parser.do(root=root, goal=goal)
        if not forest:
            logging.debug('NO PARSE FOUND')
            slicevars.reset()  # reset the slice variables (keeping conditions unchanged if any)
            continue

        tsort = TopSortTable(forest)
        uniformdist = parser.reweight(forest)
        sampler = AncestralSampler(forest,
                                   tsort,
                                   omega=lambda e: uniformdist[e],
                                   generations=generations,
                                   semiring=semiring)
        derivations = list(sampler.sample(batch))
        return make_batch_conditions(derivations, semiring)


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
                   progress=False):
    semiring = SumTimes

    # configure slice variables
    u = SliceVariables({}, free_dist, free_dist_prior)
    logging.debug('%s prior=%r', free_dist.__name__, free_dist_prior)
    # make initial conditions
    # TODO: consider intialisation heuristics such as attempt_initialisation(fsa, grammars, glue_grammars, options)
    logging.info('Looking for initial set of conditions...')
    conditions = uninformed_conditions(grammars, glue_grammars, fsa, u, root, goal, batch, generations, semiring)
    logging.info('Done')
    u.reset(conditions)

    # configure logging
    sizes = [0, 0, 0]  # number of nodes, edges and derivations (for logging purposes)
    if report_counts:
        report_size = lambda: ' nodes={:5d} edges={:5d} |D|={:5d} '.format(*sizes)
    else:
        report_size = lambda: ' nodes={:5d} edges={:5d}'.format(sizes[0], sizes[1])
    if progress:
        bar = progressbar(range(burn + (N * lag)), prefix='Sampling', dynsuffix=report_size)
    else:
        bar = range(burn + (N * lag))

    # sample
    markov_chain = []
    for _ in bar:
        # create a bottom-up parser with slice variables
        parser = Nederhof(grammars, fsa,
                          glue_grammars=glue_grammars,
                          semiring=semiring,
                          slice_variables=u)

        # compute a slice (a randomly pruned forest)
        forest = parser.do(root=root, goal=goal)
        if not forest:
            raise ValueError('A slice can never be emtpy.')

        # sample from the slice
        tsort = TopSortTable(forest)
        residual = parser.reweight(forest)
        sampler = AncestralSampler(forest, tsort,
                                   omega=lambda e: residual[e],
                                   generations=generations,
                                   semiring=semiring)
        derivations = list(sampler.sample(batch))
        # update the slice variables and the state of the Markov chain
        u.reset(make_batch_conditions(derivations, semiring))
        markov_chain.append(derivations)

        # update logging information
        sizes[0], sizes[1] = forest.n_nonterminals(), len(forest)
        if report_counts:  # reporting counts
            sizes[2] = sampler.n_derivations()

    return markov_chain


def apply_filters(markov_chain, burn=0, lag=1, resample=0):
    """
    Filter a Markov chain using a few known tricks to reduce autocorrelation.

    :param markov_chain: the chain of samples (or batches of samples)
    :param burn: discard a number of states from the beginning of the chain
    :param lag: retains states constantly spaced
    :param resample: resamples with replacement a number of states
    :return: a stream of states
    """

    iterable = iter(markov_chain)
    total = len(markov_chain)

    # burn from the left end
    if burn > 0:
        iterable = itertools.islice(iterable, burn, len(markov_chain))
        total -= burn

    # evenly spaced samples
    if lag > 1 and total > 0:
        iterable = itertools.islice(iterable, lag - 1, total, lag)
        total //= lag

    filtered_chain = list(iterable)

    # resampling step
    if resample > 0:
        filtered_chain = [filtered_chain[i] for i in np.random.randint(0, len(filtered_chain), resample)]

    return filtered_chain


def apply_batch_filters(batches, resample=0):
    """
    After applying the usual filters to a Markov chain whose states are batches of samples,
     rather than single samples, we flatten these batches, making a stream of samples.
    In this case, one might want to resample from the batch instead of leaving it whole.

    :param batches: a sequence of batches of samples
    :param resample: resamples with replacement a number of elements from each batch
    :return: a stream of samples
    """
    if resample > 0:
        samples = []
        for batch in batches:
            choices = np.random.randint(0, len(batch), resample)
            samples.extend(batch[i] for i in choices)
    else:
        samples = list(itertools.chain(*batches))
    return samples


def group_by_identity(derivations):
    counts = Counter(derivations)
    output = []
    for d, n in counts.most_common():
        output.append(SimpleNamespace(derivation=d, count=n))
    return output


def group_by_projection(samples, get_projection):
    p2d = defaultdict(set)
    counts = Counter()
    for d in samples:
        y = get_projection(d)
        counts[y] += 1
        p2d[y].add(d)
    output = []
    for y, n in counts.most_common():
        output.append(SimpleNamespace(projection=y, count=n, derivations=p2d[y]))
    return output