"""
:Authors: - Wilker Aziz
"""

import logging
import numpy as np
from collections import defaultdict, Counter
from itertools import chain

from easyhg.grammar.semiring import SumTimes
from easyhg.grammar.cfg import CFG, CFGProduction, Terminal, Nonterminal, TopSortTable
from easyhg.grammar.utils import make_nltk_tree, inlinetree
from easyhg.grammar.symbol import make_span, Span

from easyhg.alg.exact.inference import total_weight
from easyhg.alg.exact import AncestralSampler, EarleyRescoring
from .slicevars import ExpSliceVariables, ConstantPrior, SymmetricGamma, AsymmetricGamma

from easyhg.recipes import progressbar


def make_span_cell(lhs):
    return lhs


def make_span_conditions(d, semiring):
    return {r.lhs: semiring.as_real(r.weight) for r in d}


def make_coarser(derivation):

    def generate_output():
        for r in derivation:
            yield CFGProduction(r.lhs.base if isinstance(r.lhs, Span) else r.lhs,
                                (s.base if isinstance(s, Span) else s for s in r.rhs),
                                r.weight)
    return tuple(generate_output())


def gamma_priors(forest, semiring, percentile=None):
    priors = defaultdict(None)
    if percentile is None:
        for lhs, rules in forest.iteritems():
            thetas = np.array([semiring.as_real(r.weight) for r in rules])
            priors[lhs] = thetas.mean()
            print('{0} min={1} max={2} mean={3} prior={4}'.format(lhs, np.min(thetas), np.max(thetas), np.mean(thetas), priors[lhs]))
    else:
        for lhs, rules in forest.iteritems():
            thetas = np.array([semiring.as_real(r.weight) for r in rules])
            priors[lhs] = np.percentile(thetas, percentile)
            print('{0} min={1} max={2} mean={3} prior={4}'.format(lhs, np.min(thetas), np.max(thetas), np.mean(thetas), priors[lhs]))
    #logging.info('cell=%s incoming=%d sum=%s mean=%s median=%s prior=%s', lhs, len(rules), thetas.sum(), thetas.mean(), np.percentile(thetas, 50), priors[lhs])
    return priors


def prune_and_reweight(forest, tsort, slicevars, semiring, make_cell, dead_terminal=Terminal('<null>')):
    """
    Prune a forest and compute a new weight function for edges.

    :param forest: weighted forest
    :param tsort: forest's top-sort table
    :param slicevars: an instance of SliceVariable
    :param semiring: the appropriate semiring
    :param dead_terminal: a Terminal symbol which represents a dead-end.
    :return: pruned CFG and weight table mapping edges to weights.
    """
    oforest = CFG()
    discovered = set([tsort.root()])
    weight_table = defaultdict(None)
    pruned = 0
    total_cells = 0
    n_uniform = []
    n_exponential = []
    for level in tsort.iterlevels(reverse=True):  # we go top-down level by level
        for lhs in chain(*level):  # flatten the buckets
            total_cells += 1
            if lhs not in discovered:  # symbols not yet discovered have been pruned
                continue
            cell = make_cell(lhs)
            n = 0
            incoming = forest.get(lhs)
            # we sort the incoming edges in order to find out which ones are not in the slice
            for rule in sorted(incoming, key=lambda r: r.weight, reverse=True):  # TODO: use semiring to sort
                u = semiring.from_real(slicevars[cell])
                if semiring.gt(rule.weight, u):  # inside slice
                    oforest.add(rule)
                    weight_table[rule] = slicevars.pdf_semiring(cell, rule.weight, semiring)
                    discovered.update(filter(lambda s: isinstance(s, Nonterminal), rule.rhs))
                    n += 1
                else:  # this edge and the remaining are pruned
                    pruned += 1
                    #logging.info('pruning: %s vs %s', p, slicevars[cell])
                    break
            if slicevars.has_conditions(cell):
                n_uniform.append(float(n)/len(incoming))
            else:
                n_exponential.append(float(n)/len(incoming))
            #logging.info('cell=%s slice=%d/%d conditioning=%s', cell, n, len(incoming), slicevars.has_conditions(cell))
            if n == 0:  # this is a dead-end, instead of pruning bottom-up, we add an edge with 0 weight for simplicity
                dead = CFGProduction(lhs, [dead_terminal], semiring.zero)
                oforest.add(dead)
                weight_table[dead] = semiring.zero
    #logging.info('Pruning: cells=%s/%s in=%s out=%s', pruned, total_cells, np.array(_in).mean(), np.array(_out).mean())
    return oforest, weight_table, np.mean(n_uniform), np.mean(n_exponential)


class SlicedRescoring(object):

    def __init__(self, forest,
                 tsort,
                 stateless,
                 stateful,
                 semiring=SumTimes,
                 generations=10,
                 do_nothing=set(),
                 temperature0=1.0):
        self._forest = forest
        self._tsort = tsort
        self._stateless = stateless
        self._stateful = stateful
        self._do_nothing = do_nothing
        self._semiring = semiring
        self._generations = generations
        if temperature0 == 1.0:
            self._sampler0 = AncestralSampler(self._forest, self._tsort, generations=self._generations)
        else:
            self._sampler0 = AncestralSampler(self._forest, self._tsort, generations=self._generations,
                                              omega=lambda e: e.weight/temperature0)
        logging.info('l-forest: derivations=%d', self._sampler0.n_derivations())

        self._make_conditions = make_span_conditions
        self._make_cell = make_span_cell

        #self._priors = gamma_priors(self._forest, self._semiring)

    def sample_d0(self):
        """Draw an initial derivation from the locally weighted forest."""
        return next(self._sampler0.sample(1))

    def _rescore_derivation(self, d):
        semiring = self._semiring
        w = semiring.one
        # stateless scorer goes edge by edge
        # TODO: stateless.total_score(derivation)
        for edge in filter(lambda e: e.lhs not in self._do_nothing, d):
            w = semiring.times(w, self._stateless.score(edge))
        w = semiring.times(w, self._stateful.score_derivation(d))
        return w

    def _make_slice_varialbes(self, conditions, prior_type, prior_parameter):
        if prior_type == 'const':
            prior = ConstantPrior(const=float(prior_parameter))
            logging.debug('Constant scale: %s', prior_parameter)
        elif prior_type == 'sym':
            prior = SymmetricGamma(scale=float(prior_parameter))
            logging.debug('Symmetric Gamma: %s', prior_parameter)
        elif prior_type == 'asym':
            if prior_parameter == 'mean':
                prior = AsymmetricGamma(scales=gamma_priors(self._forest, self._semiring))
            else:
                try:
                    percentile = float(prior_parameter)
                except ValueError:
                    raise ValueError("The parameter of the asymmetric "
                                     "Gamma must be the keyword 'mean' or a number between 0-100: %s" % percentile)
                if not (0 <= percentile <= 100):
                    raise ValueError("A percentile is a real number between 0 and 100: %s" % percentile)
                prior = AsymmetricGamma(scales=gamma_priors(self._forest, self._semiring, percentile=percentile))
            logging.debug('Asymmetric Gamma: %s', prior_parameter)

        return ExpSliceVariables(conditions=conditions, prior=prior)

    def _importance(self, l_slice, g, batch_size):
        semiring = self._semiring
        # sample from g(d) over the truncated support of l(d)
        sampler = AncestralSampler(l_slice,
                                   TopSortTable(l_slice),
                                   omega=lambda e: g[e],
                                   generations=self._generations)

        # sample a number of derivations and group them by identity
        counts = Counter(sampler.sample(batch_size))
        # compute the empirical (importance) distribution r(d) \propto f(d|u)/g(d|u)
        support = [None] * len(counts)
        numerators = np.zeros(len(counts))
        for i, (d, n) in enumerate(counts.items()):
            support[i] = d
            numerators[i] = semiring.times(self._rescore_derivation(d), semiring.from_real(n))
        log_prob = numerators - np.logaddexp.reduce(numerators)
        prob = np.exp(log_prob)

        # sample a derivation
        i = np.random.choice(len(log_prob), p=prob)
        d_i = support[i]
        # make conditions based on the slice variables and l(d)
        c_i = self._make_conditions(d_i, semiring)

        return d_i, c_i, sampler.n_derivations()

    def _ancestral(self, l_slice, g, batch_size):
        semiring = self._semiring
        # compute f(d|u) \propto \pi(d) g(d) exactly
        rescorer = EarleyRescoring(l_slice,
                                   semiring=semiring,
                                   omega=lambda e: g[e],
                                   stateful=self._stateful,
                                   stateless=self._stateless,
                                   do_nothing=self._do_nothing)
        f2l_edges = defaultdict(None)
        f_slice = rescorer.do(root=self._tsort.root(), edge_mapping=f2l_edges)

        # sample a derivation
        sampler = AncestralSampler(f_slice,
                                   TopSortTable(f_slice),
                                   generations=self._generations)

        d_i = next(sampler.sample(1))  # a derivation from f(d|u)

        # make new conditions and reset slice variables
        # 1) the cells must be unwrapped (remember that EarleyRescoring wrap them into spans
        # 2) the weights must come from l, thus we use the edge map
        c_i = {self._make_cell(e.lhs.base): semiring.as_real(f2l_edges[e].weight) for e in d_i[1:]}

        return tuple(f2l_edges[e] for e in d_i[1:]), c_i, sampler.n_derivations()

    def sample(self, args):

        semiring = self._semiring
        sampler = self._ancestral if args.within == 'ancestral' else self._importance
        d0 = self.sample_d0()
        logging.info('Initial sample: prob=%s tree=%s', semiring.as_real(total_weight(d0, semiring) - self._sampler0.Z),
                     inlinetree(make_nltk_tree(d0)))
        # get slice variables
        slicevars = self._make_slice_varialbes(self._make_conditions(d0, semiring), args.prior[0], args.prior[1])

        history = []
        slices_sizes = [self._sampler0.n_derivations()]
        # mean ratio of sliced edges constrained by u either uniformly or exponentially distributed
        muni, mexp = 1.0, 1.0
        report_size = lambda: ' |D|={:5d} uni={:5f} exp={:5f}'.format(slices_sizes[-1], muni, mexp)

        for _ in progressbar(range(args.burn + (args.samples * args.lag)), prefix='Sampling', dynsuffix=report_size):

            # get a truncated forest weighted by l(d)
            # and a weight table corresponding to g(d)
            l_slice, g, muni, mexp = prune_and_reweight(self._forest, self._tsort, slicevars, semiring, self._make_cell)

            d_i, c_i, n_i = sampler(l_slice, g, args.batch)

            # update the history
            history.append(d_i)
            # reset slice variables
            slicevars.reset(c_i)

            # reports
            slices_sizes.append(n_i)
            #logging.info('Sample: tree=%s', inlinetree(make_nltk_tree(d_i)))
            if n_i == 0:
                raise ValueError('I found an empty slice!')

        logging.info('Slice size: mean=%s std=%s', np.mean(slices_sizes[1:]), np.std(slices_sizes[1:]))
        return history