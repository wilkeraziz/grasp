"""
@author wilkeraziz
"""

import logging
import itertools
import argparse
import sys
from fsa import make_linear_fsa
from ply_cfg import read_grammar
from cfg import FrozenCFG
from earley import Earley

def main(args):
    wcfg = read_grammar(args.grammar)
    print 'GRAMMAR'
    print wcfg

    for input_str in args.input:
        wfsa = make_linear_fsa(input_str)
        print 'FSA'
        print wfsa
        parser = Earley(wcfg, wfsa)
        status, R = parser.do()
        if not status:
            print 'NO PARSE FOUND'
            continue
        forest = FrozenCFG(R)
        print 'FOREST'
        print forest



def argparser():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(prog='parse')

    parser.description = 'Earley parser'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('grammar',
            type=argparse.FileType('r'), 
            help='CFG rules')
    parser.add_argument('input', nargs='?', 
            type=argparse.FileType('r'), default=sys.stdin,
            help='input corpus (one sentence per line)')
    #parser.add_argument('output', nargs='?', 
    #        type=argparse.FileType('w'), default=sys.stdout,
    #        help='parse trees')
    parser.add_argument('--verbose', '-v',
            action='store_true',
            help='increase the verbosity level')

    return parser

if __name__ == '__main__':
    main(argparser().parse_args())
