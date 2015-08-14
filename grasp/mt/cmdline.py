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
    parser.add_argument('--experiment',
                        type=str,
                        help='folder within the workspace where results are stored'
                             'by default we use a timestamp and a random suffix')
    parser.add_argument("--grammars",
                        type=str,
                        help="where to find grammars (grammar files are expected to be named grammar.$i.sgm, "
                             "with $i 0-based)")
    #parser.add_argument("--config", '-c',
    #                    type=str,
    #                    help="path to config file")
    parser.add_argument('--cpus',
                        type=int, default=1,
                        help='number of cpus avaiable')
    parser.add_argument('--shuffle',
                        action='store_true',
                        help='shuffle the input sentences before decoding')


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
    group.add_argument('--rt',
                       action='store_true',
                       help='include rule table features')
    group.add_argument('--wp', nargs=2,
                       help='include a word penalty feature (name, penalty)')
    group.add_argument('--ap', nargs=2,
                       help='include an arity penalty feature (name, penalty)')
    group.add_argument('--slm', nargs=3,
                       help='score n-grams within rules with a stateless LM (name, order, path).')
    group.add_argument('--lm', nargs=3,
                       help='rescore forest with a language model (name, order, path).')


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
    group.add_argument('--profile',
            type=str, metavar='PSTATS',
            help='use cProfile and save a pstats report')
    group.add_argument('--report',
                      action='store_true',
                      help='Human readable report of performance (much simpler than proper profiling).')
    group.add_argument('--verbose', '-v',
            action='count', default=0,
            help='increase the verbosity level')
    group.add_argument('--progress',
            action='store_true',
            help='display a progress bar (within slice sampling only)')


def cmd_viterbi(group):
    group.add_argument('--viterbi',
        action='store_true',
        help='best derivation (via max-times semiring)')
    group.add_argument('--kbest',
        type=int, default=0, metavar='K',
        help='top scoring derivations (Huang and Chiang, 2005)')


def cmd_sampling(group):
    group.add_argument('--temperature',
                       type=float, default=1.0,
                       help='peak (0 < t < 1.0) or flatten (t > 1.0) the distribution')
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
    group.add_argument('--batch',
            type=int, default=1, metavar='K',
            help='number of samples per slice')
    group.add_argument('--chains',
            type=int, default=1, metavar='K',
            help='number of random restarts')
    group.add_argument('--within',
                       type=str, default='ancestral', choices=['ancestral', 'importance', 'uniform'],
                       help='how to sample within the slice')
    group.add_argument('--save-chain',
                       action='store_true',
                       help='Save the actual Markov chain on disk.')
    group.add_argument('--temperature0',
                       type=float, default=1.0,
                       help='flattens the distribution from where we obtain the initial derivation')
    group.add_argument('--prior', nargs=2,
                       default=['asym', 'mean'],
                       help="We have a slice variable for each node in the forest. "
                            "Some of them are constrained (those are sampled uniformly), "
                            "some of them are not (those are sampled from an exponential distribution). "
                            "An exponential distribution has a scale parameter which is inversely proportional "
                            "to the size of the slice. Think of the scale as a mean threshold. "
                            "You can choose a constant or a prior distribution for the scale: "
                            "'const', 'sym' (symmetric Gamma) and 'asym' (asymmetric Gamma). "
                            "Each option takes one argument. "
                            "The constant distribution takes a real number (>0). "
                            "The symmetric Gamma takes a single scale parameter (>0). "
                            "The asymmetric Gamma takes either the keyword 'mean' or "
                            "a percentile expressed as a real value between 0-100. "
                            "These are computed based on the distribution over incoming edges.")


if __name__ == '__main__':
    argparser().parse_args()