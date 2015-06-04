"""
This module is an interface for parsing as intersection.
One can choose from all available implementations.

@author wilkeraziz
"""

import logging
from itertools import chain
import argparse
import sys

from sentence import make_sentence
from symbol import Nonterminal, make_flat_symbol, make_recursive_symbol
from cfg import CFG, topsort_cfg
from earley import Earley
from cky import CKY
from nederhof import Nederhof
from utils import make_nltk_tree, inlinetree
from semiring import Prob, SumTimes, MaxTimes, Count
from inference import inside, sample, optimise, total_weight
from kbest import KBest
from collections import Counter
import projection
from reader import load_grammar


def get_parser(cfg, fsa, semiring, make_symbol, algorithm):
    if algorithm == 'earley':
        parser = Earley(cfg, fsa, semiring=semiring, make_symbol=make_symbol)
    elif algorithm == 'nederhof':
        parser = Nederhof(cfg, fsa, semiring=semiring, make_symbol=make_symbol)
    elif algorithm == 'cky':
        parser = CKY(cfg, fsa, semiring=semiring, make_symbol=make_symbol)
    else:
        raise NotImplementedError("I don't know this intersection algorithm: %s" % algorithm)
    return parser



def main(args):
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

    semiring = SumTimes

    logging.info('Loading grammar...')
    cfg = load_grammar(args.grammar, args.grammarfmt, args.log)
    logging.info('Done: rules=%d', len(cfg))

    for input_str in args.input:
        # get an input automaton
        sentence, extra_rules = make_sentence(input_str, semiring, cfg.lexicon, args.unkmodel, args.default_symbol)
        cfg.update(extra_rules)
        # get a parser
        parser = get_parser(cfg, sentence.fsa, semiring, make_flat_symbol, args.intersection)
        # make a forest
        forest = parser.do(root=Nonterminal(args.start), goal=Nonterminal(args.goal))

        if not forest:
            logging.error('NO PARSE FOUND')
            continue

        logging.info('Done! Top-sorting...')
        topsorted = list(chain(*topsort_cfg(forest)))
        logging.info('Done! Couting...')
        Ic = inside(forest, topsorted, Count, omega=lambda e: 1)
        logging.info('Done! Forest: edges=%d nodes=%d paths=%d', len(forest), forest.n_nonterminals(), Ic[topsorted[-1]])

        print '# FOREST: edges=%d nodes=%d paths=%d' % (len(forest), forest.n_nonterminals(), Ic[topsorted[-1]])
        if args.forest:
            print forest
            print
        if args.samples > 0:
            logging.info('Inside...')
            Iv = inside(forest, topsorted, semiring)
            logging.info('Done! Sampling...')
            count = Counter(sample(forest, topsorted[-1], semiring, Iv=Iv, N=args.samples))
            print '# SAMPLE: size=%d' % args.samples
            for d, n in reversed(count.most_common()):
                t = make_nltk_tree(d)
                p = total_weight(d, SumTimes, Iv[topsorted[-1]])
                print '# n={0} emp={1} exact={2}\n{3}'.format(n, float(n)/args.samples, semiring.as_real(p), inlinetree(t))
            print
        if args.kbest == 1:
            logging.info('Inside...')
            Iv = inside(forest, topsorted, MaxTimes)
            logging.info('Done! Viterbi...')
            d = optimise(forest, topsorted[-1], MaxTimes, Iv=Iv)
            t = make_nltk_tree(d)
            print '# VITERBI'
            print '# k={0} score={1}\n{2}'.format(1, Iv[topsorted[-1]], inlinetree(t))
            print
        if args.kbest > 1:
            logging.info('K-best...')
            kbest = KBest(forest, topsorted[-1], args.kbest, MaxTimes, traversal=projection.string, uniqueness=False).do()
            print '# K-BEST: size=%d' % args.kbest
            for k, d in enumerate(kbest.iterderivations()):
                t = make_nltk_tree(d)
                print '# k={0} score={1}\n{2}'.format(k + 1, total_weight(d, MaxTimes), inlinetree(t))
            print
        logging.info('Finished!')


def argparser():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(prog='parse')

    parser.description = 'Parsing as intersection'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('grammar',
            type=str,
            help='grammar file (or prefix to grammar files)')
    parser.add_argument('input', nargs='?', 
            type=argparse.FileType('r'), default=sys.stdin,
            help='input corpus (one sentence per line)')
    parser.add_argument('output', nargs='?', 
            type=argparse.FileType('w'), default=sys.stdout,
            help='directs output to a file')
    parser.add_argument('--log',
            action='store_true', 
            help='apply the log transform to the grammar (by default we assume this has already been done)')
    parser.add_argument('--grammarfmt',
            type=str, default='bar',
            choices=['bar', 'discodop'],
            help="grammar format ('bar' is the native cdec-inspired format; if 'discodop' the grammar path is interpreted as a prefix)")
    parser.add_argument('--intersection', 
            type=str, default='nederhof', choices=['nederhof', 'cky', 'earley'],
            help='default goal symbol (root after intersection)')
    parser.add_argument('--forest', 
            action='store_true',
            help='dump forest (chart)')
    parser.add_argument('--unkmodel',
            type=str, default=None,
            choices=['passthrough', 'stfdbase', 'stfd4', 'stfd6'],
            help="unknown word model")
    parser.add_argument('--default-symbol', 
            type=str, default='X',
            help='default nonterminal (use for pass-through rules)')
    parser.add_argument('--start', 
            type=str, default='S',
            help='default start symbol')
    parser.add_argument('--goal', 
            type=str, default='GOAL',
            help='default goal symbol (root after intersection)')
    parser.add_argument('--samples', 
            type=int, default=0,
            help='number of samples')
    parser.add_argument('--kbest', 
            type=int, default=0,
            help='number of top scoring solutions')
    parser.add_argument('--report-top',
            action='store_true',
            help='report the top symbol(s) in the grammar and quit')
    parser.add_argument('--verbose', '-v',
            action='store_true',
            help='increase the verbosity level')

    return parser

if __name__ == '__main__':
    main(argparser().parse_args())
