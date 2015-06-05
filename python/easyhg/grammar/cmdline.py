"""
:Authors: - Wilker Aziz
"""

import argparse
import sys


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

    cmd_grammar(parser.add_argument_group('Grammar'))
    cmd_parser(parser.add_argument_group('Parser'))
    cmd_tagging(parser.add_argument_group('POS tagger'))
    cmd_info(parser.add_argument_group('Info'))
    cmd_viterbi(parser.add_argument_group('Viterbi'))
    cmd_ancestral(parser.add_argument_group('Sampling'))
    cmd_slice(parser.add_argument_group('Slice sampling'))

    return parser


def cmd_parser(group):
    group.add_argument('--intersection',
            type=str, default='nederhof', choices=['nederhof', 'cky', 'earley'],
            help='default goal symbol (root after intersection)')
    group.add_argument('--goal',
            type=str, default='GOAL',
            help='default goal symbol (root after intersection)')


def cmd_tagging(group):
    group.add_argument('--unkmodel',
            type=str, default=None,
            choices=['passthrough', 'stfdbase', 'stfd4', 'stfd6'],
            help="unknown word model")
    group.add_argument('--default-symbol',
            type=str, default='X',
            help='default nonterminal (use for pass-through rules)')

def cmd_grammar(group):
    group.add_argument('--start',
            type=str, default='S',
            help='default start symbol')
    group.add_argument('--log',
            action='store_true',
            help='apply the log transform to the grammar (by default we assume this has already been done)')
    group.add_argument('--grammarfmt',
            type=str, default='bar',
            choices=['bar', 'discodop'],
            help="grammar format ('bar' is the native cdec-inspired format; if 'discodop' the grammar path is interpreted as a prefix)")


def cmd_info(group):
    group.add_argument('--forest',
            action='store_true',
            help='dump forest as a grammar')
    group.add_argument('--report-top',
            action='store_true',
            help='report the top symbol(s) of the grammar and quit')
    group.add_argument('--count',
            action='store_true',
            help='report the number of derivations in the forest')

    group.add_argument('--verbose', '-v',
            action='store_true',
            help='increase the verbosity level')


def cmd_viterbi(group):
    group.add_argument('--kbest',
        type=int, default=0,
        help='number of top scoring solutions')


def cmd_ancestral(group):
    group.add_argument('--samples',
            type=int, default=0,
            help='number of samples')
    group.add_argument('--sampler',
            type=str, default='ancestral', choices=['ancestral', 'slice'],
            help='sampling algorithm)')

def cmd_slice(group):
    group.add_argument('--burn',
            type=int, default=0,
            help='number of initial samples to be discarded (burn-in time) - but also consider --restart')
    group.add_argument('--batch',
            type=int, default=1,
            help='number of samples per slice')
    group.add_argument('--beta-a',
            type=float, nargs=2, default=[0.1, 0.3],
            help="Beta's shape parameter before and after we have something to condition on")
    group.add_argument('--beta-b',
            type=float, nargs=2, default=[1.0, 1.0],
            help="Beta's shape parameter before and after we have something to condition on")
    group.add_argument('--heuristic',
            type=str, choices=['empdist', 'uniform'],
            help='pick a heuristic for the first slice')
    group.add_argument('--heuristic-empdist-alpha',
            type=float, default=1.0,
            help='the heuristic "empdist" can peak/flatten the distribution using this parameter')
    group.add_argument('--heuristic-uniform-params',
            type=float, nargs=2, default=[0, 100],
            help='the lower and upper percentiles for heuristic "uniform"')


if __name__ == '__main__':
    argparser().parse_args()