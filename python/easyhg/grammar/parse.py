"""
This module is an interface for parsing as intersection.
One can choose from all available implementations.

@author wilkeraziz
"""

import logging
from itertools import chain
import argparse
import sys
from fsa import make_linear_fsa
from ply_cfg import read_grammar
from symbol import Nonterminal
from rule import CFGProduction
from cfg import CFG, topsort_cfg
from earley import Earley
from cky import CKY
from nederhof import Nederhof
from utils import make_nltk_tree
from semiring import Prob, SumTimes, MaxTimes, Count
from inference import inside, sample, optimise
from collections import Counter


def main(args):
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    semiring = SumTimes
    cfg = read_grammar(args.grammar, transform=semiring.from_real)

    for input_str in args.input:
        fsa = make_linear_fsa(input_str, semiring)

        if args.pass_through:
            for word in fsa.itersymbols():
                if not cfg.is_terminal(word):
                    cfg.add(CFGProduction(Nonterminal(args.default_symbol), [word], semiring.one))
                    logging.debug('Passthrough rule for %s', word)

        if args.algorithm == 'earley':
            parser = Earley(cfg, fsa, semiring=semiring)
        elif args.algorithm == 'nederhof':
            parser = Nederhof(cfg, fsa, semiring=semiring)
        else: 
            parser = CKY(cfg, fsa, semiring=semiring)

        forest = parser.do(root=Nonterminal(args.start), goal=Nonterminal(args.goal))
        if not forest:
            logging.error('NO PARSE FOUND')
            continue
        topsorted = list(chain(*topsort_cfg(forest)))
        Ic = inside(forest, topsorted, Count, omega=lambda e: 1)
        print '# FOREST: edges=%d nodes=%d paths=%d' % (len(forest), len(forest.nonterminals), Ic[topsorted[-1]])
        if args.forest:
            for r in forest.iterrules_topdown():
                print r
            print
        if args.samples > 0:
            print '# SAMPLE: size=%d' % args.samples
            Iv = inside(forest, topsorted, semiring)
            count = Counter(sample(forest, topsorted[-1], semiring, Iv=Iv, N=args.samples))
            for d, n in reversed(count.most_common()):
                t = make_nltk_tree(d)
                print '%d (%f) %s' % (n, float(n)/args.samples, t)
                print
        if args.nbest > 0:  # TODO: enumerate n-best for n > 1 (Huang and Chiang, 2005)
            print '# NBEST: size=%d' % args.nbest
            Iv = inside(forest, topsorted, MaxTimes)
            d = optimise(forest, topsorted[-1], MaxTimes, Iv=Iv)
            t = make_nltk_tree(d)
            print '%d (%f) %s' % (1, Iv[topsorted[-1]], t)
            print


def argparser():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(prog='parse')

    parser.description = 'Parsing as intersection'
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
    parser.add_argument('--algorithm', 
            type=str, default='earley', choices=['earley', 'cky', 'nederhof'],
            help='default goal symbol (root after intersection)')
    parser.add_argument('--forest', 
            action='store_true',
            help='dump forest (chart)')
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
            type=int, default=0,
            help='number of samples')
    parser.add_argument('--nbest', 
            type=int, default=0,
            help='number of top scoring solutions')
    parser.add_argument('--verbose', '-v',
            action='store_true',
            help='increase the verbosity level')

    return parser

if __name__ == '__main__':
    main(argparser().parse_args())
