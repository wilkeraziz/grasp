"""
@author wilkeraziz
"""

import logging
import itertools
import argparse
import sys
from fsa import make_linear_fsa
from ply_cfg import read_grammar
from symbol import Nonterminal
from rule import CFGProduction
from cfg import CFG
from semiring import Prob
from earley import Earley
from cky import CKY
from nederhof import Nederhof

def main(args):
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    cfg = read_grammar(args.grammar)

    for input_str in args.input:
        fsa = make_linear_fsa(input_str, Prob)

        if args.pass_through:
            for word in fsa.itersymbols():
                if not cfg.is_terminal(word):
                    cfg.add(CFGProduction(Nonterminal(args.default_symbol), [word], Prob.one))
                    logging.debug('Passthrough rule for %s', word)

        if args.algorithm == 'earley':
            parser = Earley(cfg, fsa, semiring=Prob)
        elif args.algorithm == 'nederhof':
            parser = Nederhof(cfg, fsa, semiring=Prob)
        else: 
            parser = CKY(cfg, fsa, semiring=Prob)

        forest = parser.do(root=Nonterminal(args.start), goal=Nonterminal(args.goal))
        if not forest:
            logging.error('NO PARSE FOUND')
            continue
        if args.forest:
            for r in forest.iterrules_topdown():
                print r
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
    parser.add_argument('--verbose', '-v',
            action='store_true',
            help='increase the verbosity level')

    return parser

if __name__ == '__main__':
    main(argparser().parse_args())
