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
from utils import make_nltk_tree
from collections import Counter
from symbol import make_recursive_symbol


def main(args):
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    cfg = read_grammar(args.grammar)
    semiring = SumTimes

    for input_str in args.input:
        fsa = make_linear_fsa(input_str, semiring)

        if args.pass_through:
            for word in fsa.itersymbols():
                if not cfg.is_terminal(word):
                    cfg.add(CFGProduction(Nonterminal(args.default_symbol), [word], semiring.one))
                    logging.debug('Passthrough rule for %s', word)

        u = SliceVariables({}, a=args.beta_a, b=args.beta_b)
        samples = []
        goal = Nonterminal(args.goal)

        while len(samples) < args.samples + args.burn:
            parser = Nederhof(cfg, fsa, semiring=semiring, slice_variables=u, make_symbol=make_recursive_symbol)
            forest = parser.do(root=Nonterminal(args.start), goal=goal)
            if not forest:
                logging.debug('NO PARSE FOUND')
                u.reset()
                continue
            topsorted = list(chain(*topsort_cfg(forest)))
            uniformdist = parser.reweight(forest)
            #Ic = inside(forest, topsorted, Count, omega=lambda e: 1)
            #logging.debug('FOREST: %d edges, %d nodes and %d paths' % (len(forest), len(forest.nonterminals), Ic[topsorted[-1]]))
            Iv = inside(forest, topsorted, semiring, omega=lambda e: uniformdist[e])
            batch = list(sample(forest, goal, semiring, Iv=Iv, N=args.batch, omega=lambda e: uniformdist[e]))
            if len(batch) > 1:  # resampling step
                d = batch[np.random.randint(0, len(batch))]
            else:
                d = batch[0]
            if np.random.uniform(0, 1) > args.restart:
                conditions = {r.lhs.label: semiring.as_real(r.weight) for r in d}
            else:
                conditions = {}
                logging.info('Random restart')
            u.reset(conditions)
            samples.append(d)
            if len(samples) % 100 == 0:
                logging.info('sampling... %d/%d', len(samples), args.samples + args.burn)
        
        count = Counter(samples[args.burn:])
        for d, n in reversed(count.most_common()):
            t = make_nltk_tree(d)
            print '%dx %s' % (n, t)
            print
            #t.draw()



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
    parser.add_argument('--restart', 
            type=float, default=0.01,
            help='chance of random restart (note that burn-in always burns the first samples, regardless of random restarts)')
    parser.add_argument('--batch', 
            type=int, default=1,
            help='number of samples per slice (if bigger than 1, we will resample 1 candidate uniformly from the batch)')
    parser.add_argument('--beta-a', 
            type=float, default=0.2,
            help="Beta's shape parameter")
    parser.add_argument('--beta-b', 
            type=float, default=1.0,
            help="Beta's shape parameter")
    parser.add_argument('--verbose', '-v',
            action='store_true',
            help='increase the verbosity level')

    return parser

if __name__ == '__main__':
    main(argparser().parse_args())
