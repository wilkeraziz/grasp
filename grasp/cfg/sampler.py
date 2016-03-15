"""
Sample trees from a PCFG.

:Authors: - Wilker Aziz
"""

import argparse
import logging
import os
from grasp.recipes import make_unique_directory
from grasp.inference.ancestral import AncestralSampler, derivation_weight, LocalSampler
from grasp.parsing.sliced.sampling import group_by_projection, group_by_identity
from grasp.cfg.projection import DerivationYield
from grasp.io.results import save_mc_derivations, save_mc_yields
from grasp.cfg import TopSortTable
import numpy as np
from grasp.recipes import progressbar


def load_grammar(path, log_transform, scfg=False):
    if log_transform:
        transform = np.log
    else:
        transform = float

    if scfg:
        import grasp.cfg.ply_scfg as ply_scfg
        grammar = ply_scfg.read_grammar(path, transform=transform)
    else:
        import grasp.cfg.ply_cfg as ply_cfg
        grammar = ply_cfg.read_grammar(path, transform=transform)

    return grammar


def argparser():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(prog='grasp.cfg.sampler',
                                     usage='python -m %(prog)s [options]',
                                     description="Grasp's PCFG sampler",
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

    cmd_grammar(parser.add_argument_group('Grammar'))
    cmd_parser(parser.add_argument_group('Parser'))
    cmd_info(parser.add_argument_group('Info'))
    cmd_sampling(parser.add_argument_group('Sampling'))
    return parser


def cmd_grammar(group):

    group.add_argument('--start',
                       type=str, default='S',
                       metavar='LABEL',
                       help='default start symbol')
    group.add_argument('--log',
                       action='store_true',
                       help='apply the log transform to the grammar (by default we assume this has already been done)')
    group.add_argument('--scfg',
                       action='store_true',
                       help="treats the input grammar as an SCFG instead of a CFG")


def cmd_parser(group):
    group.add_argument('--generations',
                       type=int, default=-1,
                       metavar='MAX',
                       help="maximum number of generations in approximating supremum values (for cyclic forests)")


def cmd_info(group):
    group.add_argument('--progress', '-p',
                       action='store_true',
                       help='display a progress bar')
    group.add_argument('--verbose', '-v',
                       action='count', default=0,
                       help='increase the verbosity level')


def cmd_sampling(group):
    group.add_argument('--samples',
                       type=int, default=0, metavar='N',
                       help='number of samples')
    group.add_argument('--temperature', '-T',
                       type=float, default=1.0,
                       help="peak/flatten the distribution")
    group.add_argument('--local', '-L',
                       action='store_true',
                       help='sample locally bypassing the computation of inside weights')


def configure():
    """
    Parse command line arguments, configures the main logger.
    :returns: command line arguments
    """

    args = argparser().parse_args()

    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    return args


def make_dirs(args, exist_ok=True):
    """
    Make output directories and saves the command line arguments for documentation purpose.

    :param args: command line arguments
    :return: main output directory within workspace (prefix is a timestamp and suffix is a unique random string)
    """

    # create the workspace if missing
    if not os.path.exists(args.workspace):
        os.makedirs(args.workspace)

    # create a unique experiment area or reuse a given one
    if not args.experiment:
        outdir = make_unique_directory(args.workspace)
    else:
        outdir = '{0}/{1}'.format(args.workspace, args.experiment)

    # create output directories for the several inference algorithms
    if args.samples > 0:
        os.makedirs('{0}/ancestral'.format(outdir), exist_ok=exist_ok)
        os.makedirs('{0}/ancestral/derivations'.format(outdir), exist_ok=exist_ok)
        os.makedirs('{0}/ancestral/yields'.format(outdir), exist_ok=exist_ok)
    if args.local:
        os.makedirs('{0}/local'.format(outdir), exist_ok=exist_ok)
        os.makedirs('{0}/local/trees'.format(outdir), exist_ok=exist_ok)
        os.makedirs('{0}/local/yields'.format(outdir), exist_ok=exist_ok)

    # write the command line arguments to an ini file
    args_ini = '{0}/args.ini'.format(outdir)

    with open(args_ini, 'w') as fo:
        for k, v in sorted(vars(args).items()):
            print('{0}={1}'.format(k,repr(v)),file=fo)

    return outdir, args_ini


def monolingual(args, outdir):
    """
    Configures the parser by parsing command line arguments and calling the core code.
    It might also profile the run if the user chose to do so.
    """
    # Load main grammars
    logging.info('Loading main grammar...')
    cfg = load_grammar(args.grammar, log_transform=args.log)

    logging.info('Main grammar: terminals=%d nonterminals=%d productions=%d', cfg.n_terminals(),
                 cfg.n_nonterminals(),
                 len(cfg))

    logging.info('Top-sorting grammar')
    tsort = TopSortTable(cfg)
    if tsort.n_top_symbols() != 1:
        raise ValueError('The input grammar has more than one start symbol, or the start symbol participates in a cycle')

    if not args.local:
        logging.info('Sampling')
        sampler = AncestralSampler(cfg, tsort, omega=lambda r: r.weight/args.temperature, generations=args.generations)
        if args.progress:
            bar = progressbar(sampler.sample(args.samples), count=args.samples, prefix='Sampling')
            samples = list(bar)
        else:
            samples = list(sampler.sample(args.samples))

        # group samples by derivation and yield
        derivations = group_by_identity(samples)
        sentences = group_by_projection(samples, get_projection=DerivationYield.string)

        # save the empirical distribution over derivations
        save_mc_derivations('{0}/ancestral/derivations/{1}.gz'.format(outdir, 0),
                            derivations,
                            inside=sampler.Z,
                            omega_d=derivation_weight)
        # save the empirical distribution over strings
        save_mc_yields('{0}/ancestral/yields/{1}.gz'.format(outdir, 0),
                       sentences)
    else:
        sampler = LocalSampler(cfg, tsort, omega=lambda r: r.weight/args.temperature)
        if args.progress:
            bar = progressbar(sampler.sample(args.samples), count=args.samples, prefix='Sampling')
            samples = list(bar)
        else:
            samples = list(sampler.sample(args.samples))
        # group samples by derivation and yield
        trees = group_by_projection(samples, get_projection=DerivationYield.derivation)
        sentences = group_by_projection(samples, get_projection=DerivationYield.string)

        # save the empirical distribution over derivations
        save_mc_yields('{0}/local/trees/{1}.gz'.format(outdir, 0), trees)
        # save the empirical distribution over strings
        save_mc_yields('{0}/local/yields/{1}.gz'.format(outdir, 0), sentences)

    print('Check output files in: %s' % outdir)


def main():
    args = configure()

    # Prepare output directories
    outdir, _ = make_dirs(args)
    logging.info('Writing files to: %s', outdir)
    print('Writing files to: %s' % outdir)

    if args.scfg:
        pass
    else:
        monolingual(args, outdir)

if __name__ == '__main__':
    main()
