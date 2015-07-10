"""
:Authors: - Wilker Aziz
"""

import argparse
import sys


def argparser():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(prog='decoder')

    parser.description = 'MT decoding by sampling'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('input', nargs='?',
            type=argparse.FileType('r'), default=sys.stdin,
            help='input corpus (one sentence per line)')
    parser.add_argument('workspace',
                        type=str,
                        help='where everything happens')
    parser.add_argument("--grammars",
                        type=str,
                        help="where to find grammars (grammar files are expected to be named grammar.$i.sgm, "
                             "with $i 0-based)")
    parser.add_argument("--config", '-c',
                        type=str,
                        help="path to config file")


    cmd_grammar(parser.add_argument_group('Grammar'))
    cmd_model(parser.add_argument_group('Log-linear model'))
    cmd_parser(parser.add_argument_group('Parser'))
    cmd_info(parser.add_argument_group('Info'))
    cmd_viterbi(parser.add_argument_group('Viterbi'))
    cmd_sampling(parser.add_argument_group('Sampling'))
    cmd_slice(parser.add_argument_group('Slice sampling'))

    return parser


def cmd_grammar(group):
    group.add_argument('--start', '-S',
            type=str, default='S',
            metavar='LABEL',
            help='default start symbol')
    group.add_argument('--extra-grammar',
                       action='append', default=[], metavar='PATH',
                       help="path to an additional grammar (multiple allowed)")
    group.add_argument('--glue-grammar',
                       action='append', default=[], metavar='PATH',
                       help="glue rules are only applied to initial states (multiple allowed)")
    group.add_argument('--pass-through',
            action='store_true',
            help="add pass-through rules for every input word (and an indicator feature for unknown words)")
    group.add_argument('--default-symbol', '-X',
            type=str, default='X', metavar='LABEL',
            help='default nonterminal (used for pass-through rules and automatic glue rules)')
    group.add_argument('--hiero-glue', '-H',
            action='store_true',
            help="add hiero's glue rules")


def cmd_model(group):
    group.add_argument('--weights',
                       type=str,
                       metavar='FILE',
                       help='weight vector')
    group.add_argument('--wp',
                       action='store_true',
                       help='include a word penalty feature')
    group.add_argument('--lm', nargs=2,
                       help='rescore forest with a language model (order, path).')


def cmd_parser(group):
    group.add_argument('--goal',
            type=str, default='GOAL', metavar='LABEL',
            help='default goal symbol (root after parsing/intersection)')
    group.add_argument('--framework',
            type=str, default='exact', choices=['exact', 'slice'],
            metavar='FRAMEWORK',
            help="inference framework: 'exact', 'slice' sampling")
    group.add_argument('--generations',
            type=int, default=20, metavar='N',
            help='number of generations in loopy inside computations')


def cmd_info(group):
    group.add_argument('--count',
            action='store_true',
            help='report the number of derivations in the forest')
    group.add_argument('--forest',
                       action='store_true',
                       help='dump unpruned forest as a grammar')
    group.add_argument('--verbose', '-v',
            action='store_true',
            help='increase the verbosity level')
    group.add_argument('--profile',
            type=str, metavar='PSTATS',
            help='use cProfile and save a pstats report')


def cmd_viterbi(group):
    group.add_argument('--viterbi',
        action='store_true',
        help='best derivation (via max-times semiring)')
    group.add_argument('--kbest',
        type=int, default=0, metavar='K',
        help='top scoring derivations (Huang and Chiang, 2005)')


def cmd_sampling(group):
    group.add_argument('--samples',
            type=int, default=0, metavar='N',
            help='number of samples')
    group.add_argument('--resample',
            action='store_true',
            help='enable a resampling step')


def cmd_slice(group):
    group.add_argument('--burn',
            type=int, default=0, metavar='N',
            help='number of initial samples to be discarded (burn-in time) - but also consider --restart')
    group.add_argument('--batch',
            type=int, default=1, metavar='K',
            help='number of samples per slice')
    group.add_argument('--history',
                       action='store_true',
                       help='dumps history files with all samples in the order they were collected (no burn-in, no lag, no resampling)')
    group.add_argument('--free-dist',
                       type=str, default='exponential', metavar='DIST', choices=['beta', 'exponential', 'gamma'],
                       help='the distribution of the free variables (those with no condition), one of {beta, exponential, gamma}.')
    group.add_argument('--a',
                       type=float, nargs=2, default=[0.1, 0.3], metavar='BEFORE AFTER',
                       help="Beta's first shape parameter before and after we have something to condition on")
    group.add_argument('--b',
                       type=float, nargs=2, default=[1.0, 1.0], metavar='BEFORE AFTER',
                       help="Beta's second shape parameter before and after we have something to condition on")
    group.add_argument('--rate',
                       type=float, nargs=2, default=[0.00001, 0.00001], metavar='BEFORE AFTER',
                       help="rate parameter of the exponential distribution: smaller for thinner slices (scale=1.0/rate)")
    group.add_argument('--shape',
                       type=float, nargs=2, default=[1.0, 1.0], metavar='BEFORE AFTER',
                       help="shape parameter of the gamma distribution")
    group.add_argument('--scale',
                       type=float, nargs=2, default=[1.0, 1.0], metavar='BEFORE AFTER',
                       help="scale parameter of the gamma distribution")


if __name__ == '__main__':
    argparser().parse_args()