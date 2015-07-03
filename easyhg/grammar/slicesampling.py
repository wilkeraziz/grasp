"""
This module is an interface for parsing as intersection.
One can choose from all available implementations.

:Authors: - Wilker Aziz
"""

import logging
import numpy as np
import itertools
from collections import defaultdict, Counter
from .symbol import Nonterminal, make_recursive_symbol
from .semiring import SumTimes, Count
from .slicevars import GeneralisedSliceVariables
from .slicednederhof import Nederhof
from .inference import robust_inside, sample, total_weight
from .cfg import TopSortTable
from .result import Result


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


def make_freedist_parameters(options, i=0):
    return {'a': options.a[i],
            'b': options.b[i],
            'rate': options.rate[i],
            'shape': options.shape[i],
            'scale': options.scale[i]}


def slice_sampling(input, grammars, glue_grammars, options):
    semiring=SumTimes
    # configure slice variables
    u = GeneralisedSliceVariables({}, distribution=options.free_dist, parameters=make_freedist_parameters(options, 0))

    goal = Nonterminal(options.goal)
    checkpoint = 10

    ## TRYING SOMETHING HERE
    #conditions = initialise_itg(input.fsa, grammars, glue_grammars, options)
    #u.reset(conditions, a=options.a[1], b=options.b[1])
    #########################
    lag = options.lag

    first = True

    history = []
    #samples = []

    n_samples = 0

    while n_samples < options.samples + options.burn:
        parser = Nederhof(grammars, input.fsa,
                          glue_grammars=glue_grammars,
                          semiring=semiring,
                          slice_variables=u,
                          make_symbol=make_recursive_symbol)
        logging.debug('Free variables: %s', str(u))
        forest = parser.do(root=Nonterminal(options.start), goal=goal)
        if not forest:
            logging.debug('NO PARSE FOUND')
            u.reset()  # reset the slice variables (keeping conditions unchanged if any)
            continue

        logging.debug('Topsort...')
        tsort = TopSortTable(forest)
        #logging.debug('Top symbol: %s', tsort.root())

        if options.count:
            Ic = robust_inside(forest, tsort, Count, omega=lambda e: Count.one, infinity=options.generations)
            logging.info('Done! Forest: %d edges, %d nodes and %d paths' % (len(forest), forest.n_nonterminals(), Ic[tsort.root()]))
        else:
            logging.debug('Done! Forest: %d edges, %d nodes' % (len(forest), forest.n_nonterminals()))

        logging.debug('Inside...')
        uniformdist = parser.reweight(forest)
        Iv = robust_inside(forest, tsort, semiring, omega=lambda e: uniformdist[e], infinity=options.generations)
        logging.debug('Sampling...')

        D = list(sample(forest, tsort.root(), semiring, Iv=Iv, N=options.batch, omega=lambda e: uniformdist[e]))
        assert D, 'The slice should never be empty'

        if first:  # the first time we find a derivation we change the parameters of the distribution associated with free variables
            u.reset(make_batch_conditions(D, semiring), parameters=make_freedist_parameters(options, 1))
            first = False
        else:
            u.reset(make_batch_conditions(D, semiring))

        history.append(D)
        # collect samples respecting a given lag
        lag -= 1
        if lag == 0:
            n_samples += len(D)
            #samples.extend(D)
            lag = options.lag

        if n_samples > checkpoint:
            logging.info('sampling... %d/%d', n_samples, options.samples + options.burn)
            checkpoint = n_samples + 10

    return history


def make_result(samples_by_iteration, lag=1, burn=0, resample=0):

    if lag > 1:
        samples = list(itertools.chain(*samples_by_iteration[lag-1::lag]))
    else:
        samples = list(itertools.chain(*samples_by_iteration))

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


