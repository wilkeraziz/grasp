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
from cfg import CFG
from symbol import Terminal, Nonterminal
from rule import SCFGProduction
from earley import Earley
from semiring import Prob
from symbol import make_flat_symbol, make_recursive_symbol

def main(args):
    scfg = read_grammar(args.grammar, cdec_adapt=args.cdec)
    print 'SCFG'
    print scfg

    for input_str in args.input:
        semiring = Prob

        # input sentence
        wfsa = make_linear_fsa(input_str, semiring=semiring)

        # add unknown rules
        for unk in wfsa.vocabulary - scfg.sigma:
            scfg.add(SCFGProduction(Nonterminal('X'), [unk], [unk], [1], semiring.one))  # TODO: generalise X

        # add glue rules
        if args.glue:
            scfg.add(SCFGProduction(Nonterminal('S'), 
                [Nonterminal('X')], 
                [Nonterminal('1')], 
                [1], 
                semiring.one))  # TODO: generalise S and X
            scfg.add(SCFGProduction(Nonterminal('S'), 
                [Nonterminal('X'), Nonterminal('X')], 
                [Nonterminal('1'), Nonterminal('2')], 
                [1, 2], 
                semiring.one))  # TODO: generalise S and X

        print 'F-CFG'
        f_cfg = scfg.f_projection(semiring=Prob, marginalise=False)
        print f_cfg
        
        print 'FSA'
        print wfsa
        
        # parsing
        parser = Earley(f_cfg, wfsa, semiring=semiring, scfg=scfg, make_symbol=make_flat_symbol)
        status, R = parser.do()

        if not status:
            print 'NO PARSE FOUND'
            continue
        forest = SCFG(R)
        print 'FOREST'
        print forest
        print 'E-CFG'
        print forest.e_projection(semiring=Prob, marginalise=True)

        # TODO: generalise feature representation and transform (to include dot product)
        # write a version of Earley taylored to single sentence input



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
    parser.add_argument('--glue',
            action='store_true',
            help='add glue grammars')
    parser.add_argument('--cdec',
            action='store_true',
            help='read cdec-formatted grammars')
    parser.add_argument('--verbose', '-v',
            action='store_true',
            help='increase the verbosity level')

    return parser

if __name__ == '__main__':
    main(argparser().parse_args())
