"""
:Authors: - Wilker Aziz
"""
import logging
import argparse
import sys

"""

:Authors: - Wilker Aziz
"""

from os.path import splitext
import subprocess as sp
import shlex
import argparse
import logging
import sys
import itertools
import os
import numpy as np
import traceback
from multiprocessing import Pool
from functools import partial
from collections import deque



from grasp.loss.fast_bleu import DecodingBLEU
from grasp.loss.fast_bleu import doc_bleu, stream_doc_bleu

import grasp.ptypes as ptypes

from grasp.recipes import smart_ropen, smart_wopen, make_unique_directory, pickle_it, unpickle_it, traceit

from grasp.scoring.scorer import TableLookupScorer, StatelessScorer, StatefulScorer
from grasp.scoring.util import make_models
from grasp.scoring.util import read_weights

from grasp.mt.cdec_format import load_grammar
from grasp.mt.util import GoalRuleMaker
from grasp.mt.util import save_forest, save_ffs, load_ffs, make_dead_srule, make_batches, number_of_batches
from grasp.mt.segment import SegmentMetaData
from grasp.mt.input import make_pass_grammar

import grasp.semiring as semiring
from grasp.semiring.operator import FixedLHS, FixedRHS

from grasp.formal.scfgop import output_projection
from grasp.formal.fsa import make_dfa, make_dfa_set, make_dfa_set2
from grasp.formal.scfgop import make_hypergraph_from_input_view, output_projection
from grasp.formal.scfgop import lookup_components, stateless_components
from grasp.formal.topsort import AcyclicTopSortTable
from grasp.formal.traversal import bracketed_string, yield_string
from grasp.formal.wfunc import TableLookupFunction, ConstantFunction, derivation_weight

from grasp.cfg.model import DummyConstant
from grasp.cfg.symbol import Nonterminal
from grasp.cfg.symbol import Terminal
from grasp.cfg.srule import OutputView

from grasp.alg.deduction import NederhofParser, EarleyParser, EarleyRescorer
from grasp.alg.inference import viterbi_derivation, AncestralSampler
from grasp.alg.value import acyclic_value_recursion, acyclic_reversed_value_recursion, compute_edge_expectation
from grasp.alg.rescoring import weight_edges
from grasp.alg.rescoring import SlicedRescoring
from grasp.alg.rescoring import stateless_rescoring
from grasp.alg.chain import apply_filters, group_by_identity, group_by_projection
from grasp.alg.expectation import expected_components

from grasp.scoring.frepr import FComponents

from grasp.io.results import save_mcmc_yields, save_mcmc_derivations, save_markov_chain

from random import shuffle
from numpy import linalg as LA
from scipy.optimize import minimize
from time import time, strftime
from types import SimpleNamespace

import grasp.mt.pipeline2 as pipeline


def npvec2str(nparray, fnames=None, separator=' '):
    """converts an array of feature values into a string (fnames can be provided)"""
    if fnames is None:
        return separator.join(repr(fvalue) for fvalue in nparray)
    else:
        return separator.join('{0}={1}'.format(fname, repr(fvalue)) for fname, fvalue in zip(fnames, nparray))


def cmd_optimisation(parser):
    # Optimisation
    parser.add_argument("--maxiter", '-M', type=int, default=10,
                        help="Maximum number of iterations")
    parser.add_argument('--mode', type=str, default='10',
                        help="use 'all' for all data, use 'online' for online updates, "
                             "use 0-100 to specify batch size in percentage")
    parser.add_argument('--shuffle',
                        action='store_true',
                        help='shuffle training instances')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='scales the initial model')
    parser.add_argument('--proxy-init', type=str, default='uniform',
                        help="use 'uniform' for uniform weights, 'random' for random weights, or choose a default weight")
    parser.add_argument('--target-init', type=str, default='uniform',
                        help="use 'uniform' for uniform weights, 'random' for random weights, or choose a default weight")
    parser.add_argument("--resume", type=int, default=0,
                        help="Resume from a certain iteration (requires the config file of the preceding run)")
    parser.add_argument('--merge', type=int, default=0,
                        help="how many iterations should we consider in estimating Z(x) (use 0 or less for all)")
    parser.add_argument("--sgd", type=int, nargs=2, default=[10, 10],
                        help="Number of iterations and function evaluations for target optimisation")
    parser.add_argument("--tol", type=float, nargs=2, default=[1e-9, 1e-9],
                        help="f-tol and g-tol in target optimisation")
    parser.add_argument("--L2", type=float, default=0.0,
                        help="Weight of L2 regulariser in target optimisation")


def cmd_logging(parser):
    parser.add_argument('--save-d',
                        action='store_true', default=0,
                        help='store sampled derivations (after MCMC filters apply)')
    parser.add_argument('--save-y',
                        action='store_true', default=0,
                        help='store sampled translations (after MCMC filters apply)')
    parser.add_argument('--verbose', '-v',
                        action='count', default=0,
                        help='increase the verbosity level')


def cmd_loss(group):
    group.add_argument('--bleu-order',
                       type=int, default=4,
                       metavar='N',
                       help="longest n-gram feature for sentence-level IBM-BLEU")
    group.add_argument('--bleu-smoothing',
                       type=float, default=1.0,
                       metavar='F',
                       help="add-p smoothing for sentence-level IBM-BLEU")


def cmd_parser(group):
    group.add_argument('--goal',
                       type=str, default='GOAL', metavar='LABEL',
                       help='default goal symbol (root after parsing/intersection)')
    group.add_argument('--framework',
                       type=str, default='exact', choices=['exact', 'slice'],
                       metavar='FRAMEWORK',
                       help="inference framework: 'exact', 'slice' sampling")


def cmd_grammar(group):
    group.add_argument('--start', '-S',
                       type=str, default='S',
                       metavar='LABEL',
                       help='default start symbol')
    group.add_argument("--dev-grammars", type=str,
                       help="grammars for the dev set")
    group.add_argument("--devtest-grammars", type=str,
                       help="grammars for the devtest set")
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


def cmd_sampler(group):
    group.add_argument('--samples',
                       type=int, default=100,
                       metavar='N',
                       help="number of samples from proxy")

def get_argparser():
    parser = argparse.ArgumentParser(description='Training by MLE',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #parser.add_argument('config', type=str, help="configuration file")
    parser.add_argument("workspace",
                        type=str, default=None,
                        help="where samples can be found and where decisions are placed")
    parser.add_argument("proxy", type=str,
                        help="proxy model description")
    parser.add_argument("target", type=str,
                        help="target model description")
    parser.add_argument("dev", type=str,
                        help="development set")
    parser.add_argument('--experiment',
                        type=str,
                        help='folder within the workspace where results are stored'
                             'by default we use a timestamp and a random suffix')
    parser.add_argument("--proxy-weights", '-Q', type=str,
                        help="proxy weights")
    parser.add_argument("--target-weights", '-P', type=str,
                        help="target weights")
    parser.add_argument("--proxy-temperature", '-Tq', type=float, default=1.0,
                        help="scales the model (the bigger the more uniform)")
    parser.add_argument("--target-temperature", '-Tp', type=float, default=1.0,
                        help="scales the model (the bigger the more uniform)")
    parser.add_argument("--jobs", type=int, default=2, help="number of processes")
    parser.add_argument('--dev-alias', type=str, default='dev',
            help='Change the alias of the dev set')
    parser.add_argument("--devtest", type=str,
                        help="devtest set")
    parser.add_argument('--devtest-alias', type=str, default='devtest',
            help='Change the alias of the devtest set')
    parser.add_argument('--redo', action='store_true',
                        help='overwrite already computed files (by default we do not repeat computation)')
    cmd_parser(parser.add_argument_group('Parser'))
    cmd_grammar(parser.add_argument_group('Grammar'))
    cmd_optimisation(parser.add_argument_group('Parameter optimisation by SGD'))
    cmd_loss(parser.add_argument_group('Loss'))
    cmd_sampler(parser.add_argument_group('Importance sampler'))
    cmd_logging(parser.add_argument_group('Logging'))
    # General

    return parser


def make_dirs(args, exist_ok=True):
    """
    Make output directories and saves the command line arguments for documentation purpose.

    :param args: command line arguments
    :return: main output directory within workspace (prefix is a timestamp and suffix is a unique random string)
    """

    # create the workspace if missing
    logging.info('Workspace: %s', args.workspace)
    if not os.path.exists(args.workspace):
        os.makedirs(args.workspace, exist_ok=exist_ok)

    # create a unique experiment area or reuse a given one
    if not args.experiment:
        outdir = make_unique_directory(args.workspace)
    else:
        outdir = '{0}/{1}'.format(args.workspace, args.experiment)
        os.makedirs(outdir, exist_ok=exist_ok)
    logging.info('Writing files to: %s', outdir)

    devdir = '{0}/{1}'.format(outdir, args.dev_alias)
    os.makedirs(devdir, exist_ok=exist_ok)

    if args.devtest:
        devtestdir = '{0}/{1}'.format(outdir, args.devtest_alias)
        os.makedirs(devtestdir, exist_ok=exist_ok)

    dynamicdir = '{0}/iterations'.format(outdir)
    os.makedirs(dynamicdir, exist_ok=exist_ok)

    return outdir, devdir


@traceit
def pass0_to_pass2(seg, options, workingdir, model, redo, log):
    saving = {'pass2.forest': '{0}/{1}.q-forest'.format(workingdir, seg.id),
              'pass2.components': '{0}/{1}.q-components'.format(workingdir, seg.id)}

    if pipeline.all_steps_complete(saving, redo):
        return True

    forest, components = pipeline.pass0_to_pass2(seg,
                                                 options,
                                                 model.lookup,
                                                 model.stateless,
                                                 model.stateful,
                                                 saving=saving, redo=redo, log=log)

    return forest.n_nodes() > 0


def make_pass0_to_pass2_options(args):
    options = SimpleNamespace()
    options.extra_grammars = args.extra_grammar
    options.glue_grammars = args.glue_grammar
    options.pass_through = args.pass_through
    options.default_symbol = args.default_symbol
    options.goal = args.goal
    options.start = args.start
    return options


def parse_training(args, staticdir, model, segments):
    logging.info('Parsing %d training instances using %d workers', len(segments), args.jobs)

    with Pool(args.jobs) as workers:
        feedback = workers.map(partial(pass0_to_pass2,
                                       options=make_pass0_to_pass2_options(args),
                                       workingdir=staticdir,
                                       model=model,
                                       redo=args.redo,
                                       log=logging.info),
                               segments)
    return tuple([seg for seg, status in zip(segments, feedback) if status])


def make_impsamp_options(args):
    options = make_pass0_to_pass2_options(args)
    options.samples = args.samples
    options.bleu_order = args.bleu_order
    options.bleu_smoothing = args.bleu_smoothing
    return options


@traceit
def importance_sample(seg, options, staticdir, workingdir, proxy, target, redo, log):

    saving = {'is.samples': '{0}/samples/{1}.is'.format(workingdir, seg.id),
              'pass2.forest': '{0}/{1}.q-forest'.format(staticdir, seg.id),
              'pass2.components': '{0}/{1}.q-components'.format(staticdir, seg.id)}

    # TODO:
    #   1. normalise q(d) \propto g(d) exactly?
    #   2. use sample frequency for q(d)?
    #   3. use unnormalised g(d)

    samples = pipeline.importance_sample(seg,
                                         options,
                                         proxy,
                                         target,
                                         saving=saving, redo=redo, log=log)

    # support
    Y = [None] * len(samples)
    # posterior
    Q = np.zeros(len(samples), dtype=ptypes.weight)
    P = np.zeros(len(samples), dtype=ptypes.weight)
    # compute posterior
    for i, sample in enumerate(samples):
        Y[i] = sample.y.split()
        D = sample.D
        qy = 0.0
        py = 0.0
        #py = semiring.inside.zero
        for d in D:
            f = target.score(d.p_comps)
            g = proxy.score(d.q_comps)  # TODO: consider normalising g exactly
            w = semiring.inside.divide(f, g)
            qy += float(d.count) / len(samples)
            py += d.count * semiring.inside.as_real(w)
            #py = semiring.inside.plus(semiring.inside.times(semiring.inside.from_real(d.count), w), py)
        #P[i] = semiring.inside.as_real(py)
        Q[i] = qy
        P[i] = py
    P /= P.sum()
    # compute consensus loss
    bleu = DecodingBLEU(Y, P, max_order=options.bleu_order, smoothing=options.bleu_smoothing)
    L = [bleu.loss(y) for y in Y]
    ranking = sorted(range(len(Y)), key=lambda i: (L[i], -P[i]))

    with smart_wopen('{0}/samples/{1}.ranking.gz'.format(workingdir, seg.id)) as fo:
        print('# L ||| p(y) ||| q(y) ||| y', file=fo)
        for i in ranking:
            print('{0} ||| {1} ||| {2} ||| {3}'.format(L[i], P[i], Q[i], samples[i].y), file=fo)

    return samples[i].y, P[i], L[i]


def sample_and_decode(args, staticdir, workingdir, proxy, target, segments):
    logging.info('Decoding %d segments using %d workers', len(segments), args.jobs)
    os.makedirs('{0}/samples'.format(workingdir), exist_ok=True)
    with Pool(args.jobs) as workers:
        decisions = workers.map(partial(importance_sample,
                                        options=make_impsamp_options(args),
                                        staticdir=staticdir,
                                        workingdir=workingdir,
                                        proxy=proxy,
                                        target=target,
                                        redo=args.redo,
                                        log=logging.info),
                                segments)
    return decisions


def mteval(args, workspace, iteration, proxy, target, segments, alias):
    decisions = sample_and_decode(args,
                                  '{0}/{1}'.format(workspace, alias),
                                  '{0}/iterations/{1}/{2}'.format(workspace, iteration, alias),
                                  proxy, target, segments)
    evaldir = '{0}/iterations/{1}/{2}'.format(workspace, iteration, alias)
    os.makedirs(evaldir, exist_ok=True)
    with smart_wopen('{0}/hyps'.format(evaldir)) as fo:
        for y, p, l in decisions:
            print(y, file=fo)
    bleu, pn, bp = stream_doc_bleu(smart_ropen('{0}/hyps'.format(evaldir)),
                                   smart_ropen('{0}/{1}/refs'.format(workspace, alias)),
                                   max_order=args.bleu_order,
                                   smoothing=args.bleu_smoothing)
    logging.info('BLEU %s: %.4f', alias, bleu)
    return bleu


def sanity_checks(args):
    failed = False
    if not os.path.exists(args.dev):
        logging.error('Training set not found: %s', args.dev)
        failed = True
    if args.devtest and not os.path.exists(args.devtest):
        logging.error('Validation set not found: %s', args.devtest)
        failed = True
    if not os.path.exists(args.proxy):
        logging.error('Proxy model description not found: %s', args.proxy)
        failed = True
    if not os.path.exists(args.target):
        logging.error('Target model description not found: %s', args.target)
        failed = True
    if args.proxy_weights and not os.path.exists(args.proxy_weights):
        logging.error('Proxy model weights not found: %s', args.proxy_weights)
        failed = True
    if args.target_weights and not os.path.exists(args.target_weights):
        logging.error('Target model weights not found: %s', args.target_weights)
        failed = True
    return not failed


def core(args):

    workspace, devdir = make_dirs(args)

    if not sanity_checks(args):
        raise FileNotFoundError('One or more files could not be found')

    proxy = pipeline.load_model(args.proxy, args.proxy_weights, args.proxy_init, args.proxy_temperature)
    logging.info('Proxy:\n%s', proxy)
    target = pipeline.load_model(args.target, args.target_weights, args.target_init, args.target_temperature)
    logging.info('Target:\n%s', target)

    # 2. Parse data
    dev = pipeline.read_segments_from_file(args.dev, args.dev_grammars)
    dev = parse_training(args, devdir, proxy, dev)
    logging.info(' %d training instances', len(dev))

    # store references for evaluation purposes
    pipeline.save_references('{0}/{1}/refs'.format(workspace, args.dev_alias), dev)

    # Validation set
    if args.devtest is None:
        args.devtest = args.dev
        args.devtest_alias = args.dev_alias
        args.devtest_grammars = args.dev_grammars
        devtest = dev
    else:
        devtest = pipeline.read_segments_from_file(args.devtest, args.devtest_grammars)
        devtest = parse_training(args, '{0}/{1}'.format(workspace, args.devtest_alias), proxy, devtest)
        logging.info(' %d validation instances', len(devtest))
        pipeline.save_references('{0}/{1}/refs'.format(workspace, args.devtest_alias), devtest)


    # evaluate the initial model
    mteval(args, workspace, 0, proxy, target, devtest, args.devtest_alias)
    ##print('{0} ||| init ||| {1}={2} ||| {3}'.format(0, args.devtest_alias, bleu, npvec2str(model.weights().densify(), fnames)))

    # 3. Optimise
    #dimensionality = len(fnames)



def main():
    args = get_argparser().parse_args()

    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    core(args)


if __name__ == '__main__':
    main()
