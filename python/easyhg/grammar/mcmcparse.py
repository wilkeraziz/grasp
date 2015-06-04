"""
This module is an interface for parsing as intersection.
One can choose from all available implementations.

@author wilkeraziz
"""

import logging
import itertools
import argparse
import sys
import numpy as np
from collections import defaultdict
from fsa import make_linear_fsa
from ply_cfg import read_grammar
from symbol import Nonterminal
from rule import CFGProduction
from semiring import Prob, SumTimes, Count
from cfg import CFG, topsort_cfg
from slicesampling import SliceVariables
from slicednederhof import Nederhof
from inference import inside, sample, normalised_edge_inside
from itertools import chain
from utils import make_nltk_tree, inlinetree
from collections import Counter
from symbol import make_recursive_symbol
import heuristic

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
        conditions = {s: min(thetas) for s, thetas in conditions.iteritems()}
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


def main(args):
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

    semiring = SumTimes
    logging.info('Loading grammar...')
    if args.log:
        cfg = read_grammar(args.grammar, transform=semiring.from_real)
    else:
        cfg = read_grammar(args.grammar)
    logging.info('Done: rules=%d', len(cfg))

    for input_str in args.input:
        logging.info('Input: %s', input_str.strip())
        fsa = make_linear_fsa(input_str, semiring)

        if args.pass_through:
            for word in fsa.itersymbols():
                if not cfg.is_terminal(word):
                    cfg.add(CFGProduction(Nonterminal(args.default_symbol), [word], semiring.one))
                    logging.debug('Passthrough rule for %s', word)

        heuristic = make_heuristic(args, cfg, semiring)

        u = SliceVariables({}, a=args.beta_a[0], b=args.beta_b[0], heuristic=heuristic)
        samples = []
        goal = Nonterminal(args.goal)
        checkpoint = 100
        while len(samples) < args.samples + args.burn:
            parser = Nederhof(cfg, fsa, semiring=semiring, slice_variables=u, make_symbol=make_recursive_symbol)
            logging.debug('Parsing...')
            forest = parser.do(root=Nonterminal(args.start), goal=goal)
            if not forest:
                logging.debug('NO PARSE FOUND')
                u.reset()  # reset the slice variables (keeping conditions unchanged if any)
                continue
            topsorted = list(chain(*topsort_cfg(forest)))
            Ic = inside(forest, topsorted, Count, omega=lambda e: Count.one)
            logging.debug('Done! Forest: %d edges, %d nodes and %d paths' % (len(forest), forest.n_nonterminals(), Ic[topsorted[-1]]))
            uniformdist = parser.reweight(forest)
            Iv = inside(forest, topsorted, semiring, omega=lambda e: uniformdist[e])
            logging.debug('Sampling...')
            D = list(sample(forest, topsorted[-1], semiring, Iv=Iv, N=args.batch, omega=lambda e: uniformdist[e]))
            assert D, 'The slice should never be empty'
            u.reset(make_batch_conditions(D, semiring), a=args.beta_a[1], b=args.beta_b[1])
            samples.extend(D)
            if len(samples) > checkpoint:
                logging.info('sampling... %d/%d', len(samples), args.samples + args.burn)
                checkpoint =  len(samples) + 100
        
        count = Counter(samples[args.burn:])
        for d, n in reversed(count.most_common()):
            t = make_nltk_tree(d)
            print '# count=%d prob=%f\n%s' % (n, float(n)/args.samples, inlinetree(t))
        print



def argparser():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(prog='parse')

    parser.description = 'MCMC parsing'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('grammar',
            type=argparse.FileType('r'), 
            help='CFG rules')
    parser.add_argument('input', nargs='?', 
            type=argparse.FileType('r'), default=sys.stdin,
            help='input corpus (one sentence per line)')
    parser.add_argument('output', nargs='?', 
            type=argparse.FileType('w'), default=sys.stdout,
            help='directs output to a file')
    parser.add_argument('--log',
            action='store_true', 
            help='apply the log transform to the grammar (by default we assume this has already been done)')
    parser.add_argument('--intersection', 
            type=str, default='nederhof', choices=['nederhof'],
            help='default goal symbol (root after intersection)')
    parser.add_argument('--sampler', 
            type=str, default='slice', choices=['slice'],
            help='default goal symbol (root after intersection)')
    parser.add_argument('--pass-through', 
            action='store_true',
            help='add pass through rules for unknown terminals')
    parser.add_argument('--default-symbol', 
            type=str, default='X',
            help='default nonterminal (use for pass through rules)')
    parser.add_argument('--start', 
            type=str, default='S',
            help='default start symbol')
    parser.add_argument('--goal', 
            type=str, default='GOAL',
            help='default goal symbol (root after intersection)')
    parser.add_argument('--samples', 
            type=int, default=1000,
            help='number of samples (effectively we sample a bit more, see --burn)')
    parser.add_argument('--burn', 
            type=int, default=0,
            help='number of initial samples to be discarded (burn-in time) - but also consider --restart')
    parser.add_argument('--batch', 
            type=int, default=1,
            help='number of samples per slice')
    parser.add_argument('--beta-a', 
            type=float, nargs=2, default=[0.5, 0.5],
            help="Beta's shape parameter before and after we have something to condition on")
    parser.add_argument('--beta-b', 
            type=float, nargs=2, default=[0.5, 0.5],
            help="Beta's shape parameter before and after we have something to condition on")
    parser.add_argument('--heuristic', 
            type=str, choices=['empdist', 'uniform'], 
            help='pick a heuristic for the first slice')
    parser.add_argument('--heuristic-empdist-alpha', 
            type=float, default=1.0, 
            help='the heuristic "empdist" can peak/flatten the distribution using this parameter')
    parser.add_argument('--heuristic-uniform-params', 
            type=float, nargs=2, default=[0, 100], 
            help='the lower and upper percentiles for heuristic "uniform"')
    parser.add_argument('--verbose', '-v',
            action='store_true',
            help='increase the verbosity level')

    return parser

if __name__ == '__main__':
    main(argparser().parse_args())
