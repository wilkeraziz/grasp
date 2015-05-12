"""
@author wilkeraziz
"""

import logging
import itertools
import argparse
import sys
from fsa import make_linear_fsa
from ply_scfg import read_grammar
from scfg import SCFG
from cfg import FrozenCFG
from earley import Earley
from semiring import Prob

def main(args):
    scfg = read_grammar(args.grammar)
    print 'SCFG'
    print scfg

    for input_str in args.input:
        wfsa = make_linear_fsa(input_str, semiring=Prob)
        print 'FSA'
        print wfsa
        parser = Earley(scfg.f_projection(semiring=Prob, marginalise=False), wfsa, semiring=Prob, scfg=scfg)
        status, R = parser.do()
        if not status:
            print 'NO PARSE FOUND'
            continue
        forest = FrozenCFG(R)
        print 'FOREST'
        print forest



def argparser():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(prog='decode')

    parser.description = 'Decode using Earley intersection'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('grammar',
            type=argparse.FileType('r'), 
            help='SCFG rules')
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
