"""
:Authors: - Wilker Aziz
"""

import logging
import numpy as np
from collections import defaultdict, Counter
from itertools import chain

from easyhg.grammar.semiring import SumTimes
from easyhg.grammar.cfg import CFG, CFGProduction, Terminal, Nonterminal, TopSortTable
from easyhg.grammar.projection import get_leaves
from easyhg.grammar.utils import make_nltk_tree, inlinetree
from easyhg.grammar.symbol import make_span, Span

from easyhg.alg.exact import AncestralSampler, EarleyRescoring

from .utils import make_conditions, make_freedist_parameters
from .slicevars import SliceVariables


def make_coarser(derivation, skip=0):

    def generate_output():
        for r in derivation[skip:]:
            yield CFGProduction(r.lhs.base if isinstance(r.lhs, Span) else r.lhs,
                                (s.base if isinstance(s, Span) else s for s in r.rhs),
                                r.weight)
    return tuple(generate_output())


def prune_and_reweight(forest, tsort, slicevars, semiring, dead_terminal=Terminal('<null>')):
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
            cell = lhs.underlying
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
            #logging.info('cell=(%s, %s, %s) slice=%d/%d conditioning=%s', cell[0], cell[1], cell[2], n, len(incoming), slicevars.has_conditions(cell))
            if n == 0:  # this is a dead-end, instead of pruning bottom-up, we add an edge with 0 weight for simplicity
                dead = CFGProduction(lhs, [dead_terminal], semiring.zero)
                oforest.add(dead)
                weight_table[dead] = semiring.zero
    return oforest, weight_table


class SlicedRescoring(object):

    def __init__(self, forest, tsort, scorer,
                 semiring=SumTimes,
                 generations=10):
        self._forest = forest
        self._tsort = tsort
        self._scorer = scorer
        self._semiring = semiring
        self._generations = generations
        self._sampler0 = AncestralSampler(self._forest, self._tsort, generations=self._generations)

    def sample_d0(self):
        """Draw an initial derivation from the locally weighted forest."""
        return next(self._sampler0.sample(1))

    def sample(self, args):
        """
        Slices are sampled via importance sampling.

        :param args:
        :return:
        """

        semiring = self._semiring
        samples = []
        d = self.sample_d0()
        slicevars = SliceVariables(distribution=args.free_dist,
                                   parameters=make_freedist_parameters(args, 0))

        while len(samples) < args.samples:
            conditions = make_conditions(d, semiring)
            slicevars.reset(conditions)

            forest, weight_table = prune_and_reweight(self._forest, self._tsort, slicevars, semiring)

            sampler = AncestralSampler(forest,
                                       TopSortTable(forest),
                                       omega=lambda e: weight_table[e],
                                       generations=self._generations)

            logging.info('Slice: derivations=%d', sampler.n_derivations())

            # sample a number of derivations
            batch = list(sampler.sample(args.batch))
            # compute their projections
            projections = [get_leaves(d) for d in batch]
            # maps projections to scores (saves calls to the LM as multiple derivations will project onto the same string)
            proj2counts = Counter(projections)
            proj2score = defaultdict(None, ((proj, self._scorer.total_score(proj)) for proj in proj2counts.keys()))
            # compute global score
            numerators = np.array([proj2score[proj] for proj in projections])
            # normalisation
            log_prob = numerators - np.logaddexp.reduce(numerators)
            # sample a derivation
            i = np.random.choice(len(log_prob), p=np.exp(log_prob))
            # sampled derivation
            d_i = batch[i]
            proj_i = projections[i]

            # total probability
            total_prob = 0
            for j in range(len(batch)):
                if projections[j] == proj_i:
                    total_prob += np.exp(log_prob[j])

            logging.info('Sample: prob=%s score=%d count=%d totalprob=%s proj=%s', np.exp(log_prob[i]),
                         numerators[i],
                         proj2counts[proj_i],
                         total_prob,
                         ' '.join(w.surface for w in proj_i))
            logging.info('Tree: %s', inlinetree(make_nltk_tree(d_i)))
            print(' '.join(w.surface for w in proj_i))

            # update the sample
            samples.append(d_i)
            d = d_i

        return samples

    def sample2(self, args):
        """
        Slices are sampled exactly, i.e., slices are rescored exactly.

        :param args:
        :return:
        """

        semiring = self._semiring
        samples = []
        conditions = make_conditions(self.sample_d0(), semiring)
        slicevars = SliceVariables(conditions,
                                   distribution=args.free_dist,
                                   parameters=make_freedist_parameters(args, 0))
        root = make_span(Nonterminal(args.goal), None, None)

        logging.info('Forest: derivations=%d', self._sampler0.n_derivations())

        while len(samples) < args.samples:
            forest, weight_table = prune_and_reweight(self._forest, self._tsort, slicevars, semiring)
            logging.info('Slice: terminals=%d nonterminals=%d edges=%d', forest.n_terminals(), forest.n_nonterminals(), len(forest))

            old_weights = defaultdict(None)
            rescorer = EarleyRescoring(forest, self._scorer, semiring)
            rescored = rescorer.do(root=root, original_weights=old_weights)

            sampler = AncestralSampler(rescored,
                                       TopSortTable(rescored),
                                       generations=self._generations)

            logging.info('Slice: derivations=%d', sampler.n_derivations())
            d = next(sampler.sample(1))
            y = get_leaves(d)
            logging.info('Tree: %s', inlinetree(make_nltk_tree(make_coarser(d, 1))))
            logging.info('Sample: %s', ' '.join(w.surface for w in y))
            print(' '.join(w.surface for w in y))
            # make new conditions based on local weights
            conditions = {r.lhs.base.underlying: semiring.as_real(old_weights[r]) for r in d[1:]}
            # TODO: use Span in SliceVariables

            # reset conditions
            slicevars.reset(conditions)

            # update the sample
            samples.append(d)

        return samples