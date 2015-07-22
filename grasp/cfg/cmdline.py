"""
:Authors: - Wilker Aziz
"""

import argparse


def argparser():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(prog='grasp.cfg.parser',
                                     usage='python -m %(prog)s [options]',
                                     description="Grasp's CFG parser",
                                     epilog="")

    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('grammar',
                        type=str,
                        help='grammar file (or prefix to grammar files)')
    parser.add_argument('workspace',
                        type=str,
                        help='where everything happens')
    parser.add_argument('--experiment',
                        type=str,
                        help='folder within the workspace where results are stored'
                             'by default we use a timestamp and a random suffix')
    parser.add_argument('--cpus',
                        type=int, default=1,
                        help='number of cpus available (-1 for all)')

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
                       action='count', default=0,
                       help='increase the verbosity level')
    group.add_argument('--progress',
                       action='store_true',
                       help='display a progress bar (within slice sampling only)')
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
    group.add_argument('--batch',
                       type=int, default=1, metavar='K',
                       help='number of samples per slice (typically leads to larger slices)')
    group.add_argument('--resample',
                       type=int, default=0, metavar='N',
                       help='if strictly positve, uniformly resamples with replacement N derivations from each batch')
    group.add_argument('--heuristic',
                       type=str, choices=['coarse', 'itg'],
                       help='initialisation strategy')
    group.add_argument('--save-chain',
                       action='store_true',
                       help='Save the complete Markov chain')
    group.add_argument('--free-dist',
                       type=str, default='beta', metavar='DIST', choices=['beta', 'exponential'],
                       help="We have a slice variable for each node in the forest. "
                            "Some of them are constrained (those are sampled uniformly), "
                            "some of them are not (those are sampled from a Beta or an Exponential distribution). "
                            "Options: 'beta', 'exponential'.")

    group.add_argument('--prior-a', nargs=2,
                       default=['const', 0.1],
                       help="A Beta distribution has two shape parameters (a, b) and mean a/(a+b). "
                            "The mean is inversely proportional to the size of the slice. "
                            "Think of it as an expected threshold. "
                            "You can choose a constant or a prior for the parameter a: "
                            "'const', 'sym' (symmetric Gamma), 'beta' (Beta). Each option takes one argument. "
                            "The constant takes a real number (>0). "
                            "The symmetric Gamma takes a single scale parameter (>0). "
                            "The Beta takes two shape parameters separated by a comma (x,y>0).")
    group.add_argument('--prior-b', nargs=2,
                       default=['const', 1.0],
                       help="You can choose a constant or a prior distribution for the parameter b: "
                            "'const', 'sym' (symmetric Gamma), 'beta' (Beta). Each option takes one argument. "
                            "The constant takes a real number (>0). "
                            "The symmetric Gamma takes a single scale parameter (>0). "
                            "The Beta takes two shape parameters separated by a comma (x,y>0).")
    group.add_argument('--prior-scale', nargs=2,
                       default=['const', 0.1],
                       help="The Exponential distribution has one scale parameter which is also its mean. "
                            "This scale/mean is inversely proportional to the size of the slice. "
                            "Think of it as an expected threshold. "
                            "You can choose a constant or a prior for the scale: 'const', 'sym' (symmetric Gamma). "
                            "The constant is a real number (>0). "
                            "The symmetric Gamma takes a single scale parameter (>0). ")


if __name__ == '__main__':
    argparser().parse_args()