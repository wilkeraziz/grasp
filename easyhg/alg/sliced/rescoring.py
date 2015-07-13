"""
:Authors: - Wilker Aziz
"""

import logging
import numpy as np
import sys
from collections import defaultdict, Counter
from itertools import chain

from easyhg.grammar.semiring import SumTimes
from easyhg.grammar.cfg import CFG, CFGProduction, Terminal, Nonterminal, TopSortTable
from easyhg.grammar.utils import make_nltk_tree, inlinetree
from easyhg.grammar.symbol import make_span, Span

from easyhg.alg.exact.inference import total_weight
from easyhg.alg.exact import AncestralSampler, EarleyRescoring

from .utils import choose_parameters
from .slicevars import SliceVariables

from easyhg.alg import progressbar


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
    for level in tsort.iterlevels(reverse=True):  # we go top-down level by level
        for lhs in chain(*level):  # flatten the buckets
            if lhs not in discovered:  # symbols not yet discovered have been pruned
                continue
            cell = make_cell(lhs)
            n = 0
            incoming = forest.get(lhs)
            # we sort the incoming edges in order to find out which ones are not in the slice
            for rule in sorted(incoming, key=lambda r: r.weight, reverse=True):  # TODO: use semiring to sort
                p = semiring.as_real(rule.weight)

                if slicevars.is_inside(cell, p):
                    oforest.add(rule)
                    weight_table[rule] = slicevars.logpr(cell, p)
                    discovered.update(filter(lambda s: isinstance(s, Nonterminal), rule.rhs))
                    n += 1
                else:  # this edge and the remaining are pruned
                    break
            #logging.info('cell=%s slice=%d/%d conditioning=%s', cell, n, len(incoming), slicevars.has_conditions(cell))
            if n == 0:  # this is a dead-end, instead of pruning bottom-up, we add an edge with 0 weight for simplicity
                dead = CFGProduction(lhs, [dead_terminal], semiring.zero)
                oforest.add(dead)
                weight_table[dead] = semiring.zero
    return oforest, weight_table


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

    def importance(self, args):
        """
        Slices are sampled via importance sampling.

            u_k ~ f(u_k|d) = I(u_k < l_k(d)) if u_k \in d or alpha(u_k)
            f(d|u) \propto \pi(d) g(d|u)
            g(d|u) = \prod_{u_k \in d} I(u_k < l_k(d))/alpha(u_k)
            (d_i ~ g(d|u)) for i = 1..B
            f(d|u) \approx r(d|u) \propto f(d|u)/g(d|u) \propto \pi(d)
            d ~ r(d|u)



        :param args:
        :return:
        """

        semiring = self._semiring
        d0 = self.sample_d0()

        logging.info('Initial sample: prob=%s tree=%s', semiring.as_real(total_weight(d0, semiring) - self._sampler0.Z),
                     inlinetree(make_nltk_tree(d0)))

        slicevars = SliceVariables(conditions=self._make_conditions(d0, semiring),
                                   distribution=args.free_dist,
                                   parameters=choose_parameters(args, 0))

        history = []

        slices_sizes = [self._sampler0.n_derivations()]
        report_size = lambda: ' |D|={0}'.format(slices_sizes[-1])
        for _ in progressbar(range(args.burn + (args.samples * args.lag)), prefix='Sampling', dynsuffix=report_size):

            # get a truncated forest weighted by l(d)
            # and a weight table corresponding to g(d)
            l_slice, g = prune_and_reweight(self._forest, self._tsort, slicevars, semiring, self._make_cell)

            # sample from g(d) over the truncated support of l(d)
            sampler = AncestralSampler(l_slice,
                                       TopSortTable(l_slice),
                                       omega=lambda e: g[e],
                                       generations=self._generations)

            slices_sizes.append(sampler.n_derivations())
            #logging.debug('Slice: derivations=%d', sampler.n_derivations())

            # sample a number of derivations and group them by identity
            counts = Counter(sampler.sample(args.batch))
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
            slicevars.reset(self._make_conditions(d_i, semiring))

            # update the history
            history.append(d_i)

            # report stuff
            #logging.info('Sample: prob=%s tree=%s', prob[i], inlinetree(make_nltk_tree(d_i)))


        return history

    def ancestral(self, args):
        """
        Slices are sampled exactly, i.e., slices are rescored exactly.

            f(d) \propto \pi(d) g(d)

            u_k ~ f(u_k|d) = I(u_k < l_k(d)) if u_k \in d or alpha(u_k)
            g(d|u) = \prod_{u_k \in d} I(u_k < l_k(d))/alpha(u_k)
            <d_i ~ f(d|u) \propto \pi(d) g(d|u)> for i = 1..B
            d ~ f(d|u)

        :param args:
        :return:
        """

        semiring = self._semiring
        history = []

        d0 = self.sample_d0()
        logging.info('Initial sample: prob=%s tree=%s', semiring.as_real(total_weight(d0, semiring) - self._sampler0.Z),
                     inlinetree(make_nltk_tree(d0)))

        slicevars = SliceVariables(conditions=self._make_conditions(d0, semiring),
                                   distribution=args.free_dist,
                                   parameters=choose_parameters(args, 0))
        root = make_span(Nonterminal(args.goal), None, None)

        slices_sizes = [self._sampler0.n_derivations()]
        report_size = lambda: ' |D|={0}'.format(slices_sizes[-1])

        for _ in progressbar(range(args.burn + (args.samples * args.lag)), prefix='Sampling', dynsuffix=report_size):

            # get a truncate forest weighted by l(d)
            # and a weight table corresponding to g(d)
            l_slice, g = prune_and_reweight(self._forest, self._tsort, slicevars, semiring, self._make_cell)

            # compute f(d|u) \propto \pi(d) g(d) exactly
            rescorer = EarleyRescoring(l_slice,
                                       semiring=semiring,
                                       omega=lambda e: g[e],
                                       stateful=self._stateful,
                                       stateless=self._stateless,
                                       do_nothing=self._do_nothing)
            f2l_edges = defaultdict(None)
            f_slice = rescorer.do(root=root, edge_mapping=f2l_edges)

            # sample a derivation
            sampler = AncestralSampler(f_slice,
                                       TopSortTable(f_slice),
                                       generations=self._generations)

            #logging.debug('Slice: derivations=%d', sampler.n_derivations())
            slices_sizes.append(sampler.n_derivations())
            d_i = next(sampler.sample(1))  # a derivation from f(d|u)

            # make new conditions and reset slice variables
            # 1) the cells must be unwrapped (remember that EarleyRescoring wrap them into spans
            # 2) the weights must come from l, thus we use the edge map
            slicevars.reset({self._make_cell(e.lhs.base): semiring.as_real(f2l_edges[e].weight) for e in d_i[1:]})

            # update the sample
            history.append(tuple(f2l_edges[e] for e in d_i[1:]))

            # report stuff
            #logging.info('Sample: prob=%s tree=%s', semiring.as_real(total_weight(d_i) - sampler.Z),
            #             inlinetree(make_nltk_tree(make_coarser(d_i[1:]))))

        return history