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
    parser.add_argument('workspace',
                        type=str,
                        help='where everything happens')
    parser.add_argument('--experiment',
                        type=str,
                        help='folder within the workspace where results are stored'
                             'by default we use a timestamp and a random suffix')

    cmd_grammar(parser.add_argument_group('Grammar'))
    cmd_parser(parser.add_argument_group('Parser'))
    cmd_tagger(parser.add_argument_group('POS tagger'))
    cmd_info(parser.add_argument_group('Info'))
    cmd_viterbi(parser.add_argument_group('Viterbi'))
    cmd_sampling(parser.add_argument_group('Sampling'))
    cmd_slice(parser.add_argument_group('Slice sampling'))

    return parser


def cmd_grammar(group):
    group.add_argument('--start',
                       type=str, default='S',
                       metavar='LABEL',
                       help='default start symbol')
    group.add_argument('--log',
                       action='store_true',
                       help='apply the log transform to the grammar (by default we assume this has already been done)')
    group.add_argument('--grammarfmt',
                       type=str, default='bar', metavar='FMT',
                       choices=['bar', 'discodop', 'cdec'],
                       help="grammar format: bar, discodop, cdec")
    group.add_argument('--extra-grammar',
                       action='append', default=[], metavar='PATH',
                       help="path to an additional grammar (multiple allowed)")
    group.add_argument('--glue-grammar',
                       action='append', default=[], metavar='PATH',
                       help="glue rules are only applied to initial states (multiple allowed)")


def cmd_parser(group):
    group.add_argument('--intersection',
                       type=str, default='nederhof', choices=['nederhof', 'earley'],
                       metavar='ALG',
                       help='intersection algorithms: nederhof, earley')
    group.add_argument('--goal',
                       type=str, default='GOAL', metavar='LABEL',
                       help='default goal symbol (root after intersection)')
    group.add_argument('--framework',
                       type=str, default='exact', choices=['exact', 'slice', 'gibbs'],
                       metavar='FRAMEWORK',
                       help="inference framework: 'exact', 'slice' sampling, 'gibbs' sampling")
    group.add_argument('--generations',
                       type=int, default=10,
                       metavar='MAX',
                       help="maximum number of generations in approximating supremum values (for cyclic forests)")


def cmd_tagger(group):
    group.add_argument('--unkmodel',
                       type=str, default=None, metavar='MODEL',
                       choices=['passthrough', 'stfdbase', 'stfd4', 'stfd6'],
                       help="unknown word model: passthrough, stfdbase, stfd4, stfd6")
    group.add_argument('--unklhs',
                       type=str, default='X', metavar='LABEL',
                       help='default nonterminal (use for pass-through rules)')


def cmd_info(group):
    group.add_argument('--report-top',
                       action='store_true',
                       help='topsort the grammar, report the top symbol(s), and quit')
    group.add_argument('--report-tsort',
                       action='store_true',
                       help='topsort the grammar, report the partial ordering of symbols, and quit')
    group.add_argument('--report-cycles',
                       action='store_true',
                       help='topsort the grammar, report cycles, and quit')
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


def cmd_slice(group):
    group.add_argument('--lag',
                       type=int, default=1, metavar='I',
                       help='lag between samples')
    group.add_argument('--burn',
                       type=int, default=0, metavar='N',
                       help='number of initial samples to be discarded (applies after lag)')
    group.add_argument('--resample',
                       type=int, default=0, metavar='N',
                       help='uniformly sample with replacement N derivations from the sampled derivations (use 0 or less to disable)')
    group.add_argument('--batch',
                       type=int, default=1, metavar='K',
                       help='number of samples per slice (typically leads to larger slices)')
    group.add_argument('--heuristic',
                       type=str, choices=['coarse', 'itg'],
                       help='initialisation strategy')
    group.add_argument('--history',
                       action='store_true',
                       help='dumps history files with all samples in the order they were collected (no burn-in, no lag, no resampling)')
    group.add_argument('--free-dist',
                       type=str, default='beta', metavar='DIST', choices=['beta', 'exponential', 'gamma'],
                       help='the distribution of the free variables (those with no condition), one of {beta, exponential, gamma}.')
    group.add_argument('--a',
                       type=float, nargs=2, default=[0.1, 0.3], metavar='BEFORE AFTER',
                       help="Beta's first shape parameter before and after we have something to condition on")
    group.add_argument('--b',
                       type=float, nargs=2, default=[1.0, 1.0], metavar='BEFORE AFTER',
                       help="Beta's second shape parameter before and after we have something to condition on")
    group.add_argument('--rate',
                       type=float, nargs=2, default=[1.0, 1.0], metavar='BEFORE AFTER',
                       help="rate parameter of the exponential distribution (scale=1.0/rate)")
    group.add_argument('--shape',
                       type=float, nargs=2, default=[1.0, 1.0], metavar='BEFORE AFTER',
                       help="shape parameter of the gamma distribution")
    group.add_argument('--scale',
                       type=float, nargs=2, default=[1.0, 1.0], metavar='BEFORE AFTER',
                       help="scale parameter of the gamma distribution")


if __name__ == '__main__':
    argparser().parse_args()