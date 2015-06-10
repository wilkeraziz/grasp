"""
This module is an interface for parsing as intersection.
One can choose from all available implementations.

:Authors: - Wilker Aziz
"""

import logging
from collections import defaultdict, Counter
from itertools import chain

from .symbol import Nonterminal, make_recursive_symbol
from .semiring import SumTimes, Count
from .cfg import topsort_cfg
from .slicevars import SliceVariables
from .slicednederhof import Nederhof
from .inference import inside, sample
from .utils import make_nltk_tree, inlinetree
from . import heuristic
from .reader import load_grammar
from .sentence import make_sentence


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


def make_heuristic(args, cfg, semiring):
    if not args.heuristic:
        return None
    if args.heuristic == 'empdist':
        return heuristic.empdist(cfg, semiring, args.heuristic_empdist_alpha)
    elif args.heuristic == 'uniform':
        return heuristic.uniform(cfg, semiring, args.heuristic_uniform_params[0], args.heuristic_uniform_params[1])
    else:
        raise ValueError('Unknown heuristic')


def slice_sampling(cfg, sentence, semiring, args):
    # get a heuristic
    heuristic = make_heuristic(args, cfg, semiring)
    # configure slice variables
    u = SliceVariables({}, a=args.beta_a[0], b=args.beta_b[0], heuristic=heuristic)
    samples = []
    goal = Nonterminal(args.goal)
    checkpoint = 100
    while len(samples) < args.samples + args.burn:
        parser = Nederhof(cfg, sentence.fsa, semiring=semiring, slice_variables=u, make_symbol=make_recursive_symbol)
        logging.debug('Parsing...')
        forest = parser.do(root=Nonterminal(args.start), goal=goal)
        if not forest:
            logging.debug('NO PARSE FOUND')
            u.reset()  # reset the slice variables (keeping conditions unchanged if any)
            continue
        topsorted = list(chain(*topsort_cfg(forest)))

        if args.count:
            Ic = inside(forest, topsorted, Count, omega=lambda e: Count.one)
            logging.debug('Done! Forest: %d edges, %d nodes and %d paths' % (len(forest), forest.n_nonterminals(), Ic[topsorted[-1]]))
        else:
            logging.debug('Done! Forest: %d edges, %d nodes' % (len(forest), forest.n_nonterminals()))

        uniformdist = parser.reweight(forest)
        Iv = inside(forest, topsorted, semiring, omega=lambda e: uniformdist[e])
        logging.debug('Sampling...')
        D = list(sample(forest, topsorted[-1], semiring, Iv=Iv, N=args.batch, omega=lambda e: uniformdist[e]))
        assert D, 'The slice should never be empty'
        u.reset(make_batch_conditions(D, semiring), a=args.beta_a[1], b=args.beta_b[1])
        samples.extend(D)
        if len(samples) > checkpoint:
            logging.info('sampling... %d/%d', len(samples), args.samples + args.burn)
            checkpoint = len(samples) + 100

    count = Counter(samples[args.burn:])
    for d, n in reversed(count.most_common()):
        t = make_nltk_tree(d)
        print('# count=%d prob=%f\n%s' % (n, float(n)/args.samples, inlinetree(t)))
    print()


def main(args):

    semiring = SumTimes

    logging.info('Loading grammar...')
    cfg = load_grammar(args.grammar, args.grammarfmt, args.log)
    logging.info('Done: rules=%d', len(cfg))

    for input_str in args.input:
        # get an input automaton
        sentence, extra_rules = make_sentence(input_str, semiring, cfg.lexicon, args.unkmodel, args.default_symbol)
        cfg.update(extra_rules)
        slice_sampling(cfg, sentence, semiring, args)

