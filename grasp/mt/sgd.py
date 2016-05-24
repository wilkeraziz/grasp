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
    parser.add_argument('--batch-size', '-B', type=int, default='10',
                        help="Size of mini-batches")
    #parser.add_argument('--shuffle',
    #                    action='store_true',
    #                    help='shuffle training instances')
    parser.add_argument('--init', type=str, default='uniform',
                        help="use 'uniform' for uniform weights, 'random' for random weights, or choose a default weight")
    parser.add_argument("--resume", type=int, default=0,
                        help="Resume from a certain iteration (requires the config file of the preceding run)")
    parser.add_argument('--merge', type=int, default=0,
                        help="how many iterations should we consider in estimating Z(x) (use 0 or less for all)")
#    parser.add_argument("--sgd", type=int, nargs=2, default=[10, 10],
#                        help="Number of iterations and function evaluations for target optimisation")
#    parser.add_argument("--tol", type=float, nargs=2, default=[1e-9, 1e-9],
#                        help="f-tol and g-tol in target optimisation")
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
    group.add_argument('--max-span',
                       type=int, default=-1, metavar='N',
                       help='A Hiero-style constraint: size of the longest input path under an X nonterminal (a negative value implies no constraint)')


def cmd_grammar(group):
    group.add_argument('--start', '-S',
                       type=str, default='S',
                       metavar='LABEL',
                       help='default start symbol')
    group.add_argument("--training-grammars", type=str,
                       help="grammars for the training set")
    group.add_argument("--validation-grammars", type=str,
                       help="grammars for the validation set")
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
    parser.add_argument("model", type=str,
                        help="model specification")
    parser.add_argument("training", type=str,
                        help="training set")
    #parser.add_argument('--joint', '-J',
    #                    type=str,
    #                    help='factorisation of joint inference')
    #parser.add_argument('--conditional', '-C',
    #                    type=str,
    #                    help='factorisation of conditional inference')
    parser.add_argument('--experiment',
                        type=str,
                        help='folder within the workspace where results are stored'
                             'by default we use a timestamp and a random suffix')
    parser.add_argument("--weights", '-W', type=str,
                        help="model parameters")
    parser.add_argument("--jobs", type=int, default=2, help="number of processes")
    parser.add_argument('--training-alias', type=str, default='training',
            help='Change the alias of the training set')
    parser.add_argument("--validation", type=str,
                        help="Validation set")
    parser.add_argument('--validation-alias', type=str, default='validation',
            help='Change the alias of the validation set')
    parser.add_argument('--redo', action='store_true',
                        help='overwrite already computed files (by default we do not repeat computation)')
    cmd_parser(parser.add_argument_group('Parser'))
    cmd_grammar(parser.add_argument_group('Grammar'))
    cmd_optimisation(parser.add_argument_group('Parameter optimisation by SGD'))
    cmd_loss(parser.add_argument_group('Loss'))
    cmd_sampler(parser.add_argument_group('Slice sampler'))
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

    training_dir = '{0}/{1}'.format(outdir, args.training_alias)
    os.makedirs(training_dir, exist_ok=exist_ok)

    if args.validation:
        validation_dir = '{0}/{1}'.format(outdir, args.validation_alias)
        os.makedirs(validation_dir, exist_ok=exist_ok)

    dynamicdir = '{0}/iterations'.format(outdir)
    os.makedirs(dynamicdir, exist_ok=exist_ok)

    return outdir


def sanity_checks(args):
    failed = False
    if not os.path.exists(args.training):
        logging.error('Training set not found: %s', args.training)
        failed = True
    if args.validation and not os.path.exists(args.validation):
        logging.error('Validation set not found: %s', args.validation)
        failed = True
    if not os.path.exists(args.model):
        logging.error('Model specification not found: %s', args.proxy)
        failed = True
    if args.weights and not os.path.exists(args.weights):
        logging.error('Model parameters not found: %s', args.proxy_weights)
        failed = True
    #if args.joint and not os.path.exists(args.joint):
    #    logging.error('Joint model factorisation not found: %s', args.joint)
    #    failed = True
    #if args.conditional and not os.path.exists(args.conditional):
    #    logging.error('Conditional model factorisation not found: %s', args.conditional)
    #    failed = True
    return not failed


def make_options_for_biparse(args):
    """
    Transform command line arguments into options for pipeline.biparse.
    :param args:
    :return: a SimpleNamespace object with the options
    """
    options = SimpleNamespace()
    options.extra_grammars = args.extra_grammar
    options.glue_grammars = args.glue_grammar
    options.pass_through = args.pass_through
    options.default_symbol = args.default_symbol
    options.goal = args.goal
    options.start = args.start
    options.max_span = args.max_span
    return options


@traceit
def _biparse(seg, *args, **kwargs):
    """
    This method is here to wrap pipeline.biparse and catch errors.
    This is useful for Pool.
    Also, it returns a simple bool to indicate whether the segment is parsable.
    We do not want to keep large objects in memory.

    :param seg:
    :param args:
    :param kwargs:
    :return: parsable or not
    """
    result = pipeline.biparse(seg, *args, **kwargs)
    return bool(result.conditional.forest)


def biparse(args, workingdir, model, segments):
    """
    This method bi-parses a whole data set and saves forests and components to a working directory.

    :param args: command line arguments
    :param workingdir: where to save everything
    :param model: a Model
    :param segments: a data set
    :return: parsable segments
    """
    logging.info('Parsing %d training instances using %d workers', len(segments), args.jobs)
    options = make_options_for_biparse(args)
    with Pool(args.jobs) as workers:
        feedback = workers.map(partial(_biparse,
                                       options=options,
                                       lookup=model.lookup,
                                       stateless=model.stateless,
                                       joint_stateful=model.dummy,
                                       conditional_stateful=model.dummy,
                                       workingdir=workingdir,
                                       redo=args.redo,
                                       log=logging.info),
                               segments)
    return [seg for seg, status in zip(segments, feedback) if status]


def preprocess_dataset(args, path, grammars, workingdir, model):
    """
    Apply basic pre-processing steps to a data set.
        i.e. load, save references, biparse, save forests and components.

    :param args:
    :param path:
    :param grammars:
    :param workingdir:
    :param model:
    """
    unconstrained_data = pipeline.read_segments_from_file(path, grammars)
    logging.info('Loaded %d segments', len(unconstrained_data))
    # store references for evaluation purposes
    pipeline.save_references('{0}/refs'.format(workingdir), unconstrained_data)
    # biparse
    data = biparse(args, workingdir, model, unconstrained_data)
    return unconstrained_data, data


@traceit
def _sample(seg, args, staticdir, supportdir, workspace, model):
    files = ['{0}/{1}.D.ffs.all'.format(supportdir, seg.id),
             '{0}/{1}.hyp.ffs.all'.format(workspace, seg.id)]

    if all(os.path.exists(path) for path in files) and not args.redo:
        logging.info('Reusing samples for segment %d', seg.id)
        return

    # 1. Load pickled objects for the conditional distribution
    logging.debug('[%d] Loading target forest', seg.id)
    conditional_forest = unpickle_it('{0}/{1}.conditional.forest'.format(staticdir, seg.id))

    logging.debug('[%d] Loading local components', seg.id)
    conditional_components = unpickle_it('{0}/{1}.conditional.components'.format(staticdir, seg.id))


def core(args):

    workspace = make_dirs(args)
    training_dir = '{0}/{1}'.format(workspace, args.training_alias)
    validation_dir = '{0}/{1}'.format(workspace, args.validation_alias)

    if not sanity_checks(args):
        raise FileNotFoundError('One or more files could not be found')

    model = pipeline.load_model(args.model, args.weights, args.init)
    logging.info('Model:\n%s', model)

    # 1. Load/Parse
    # 1a. Training data
    _training, training = preprocess_dataset(args, args.training, args.training_grammars, training_dir, model)
    # 1b. Validation data (if applicable)
    if args.validation:
        _validation, validation = preprocess_dataset(args, args.validation, args.validation_grammars, validation_dir, model)
    else:
        _validation, validation = [], []
    logging.info('This model can generate %d out of %d training instances', len(training),
                 len(_training))
    logging.info('This model can generate %d out of %d validation instances', len(validation),
                 len(_validation))

    # 2. SGD
    fnames = model.fnames()
    dimensionality = len(fnames)  # TODO: Sparse SGD

    for epoch in range(1, args.maxiter + 1):

        # where we store everything related to this iteration
        epoch_dir = '{0}/epoch{1}'.format(workspace, epoch)
        os.makedirs(epoch_dir, exist_ok=True)

        # first we get a random permutation of the training data
        shuffle(training)

        avg = np.zeros(dimensionality, dtype=ptypes.weight)
        weights = np.zeros(dimensionality, dtype=ptypes.weight)

        # then we operate in mini-batches
        for b, first in enumerate(range(0, len(training), args.batch_size), 1):
            # gather segments in this batch
            batch = [training[ith] for ith in range(first, min(first + args.batch_size, len(training)))]
            print([seg.id for seg in batch])
            batch_dir = '{0}/batch{0}'.format(epoch, b)
            os.makedirs(batch_dir, exist_ok=True)

            # sample with the current parameters
            #sample(args, devdir, supportdir, batchdir, model, iteration, b, batch)

            # optimise parameters

            #weights = sgd_optimise(args, devdir, batchdir, model, iteration, b, batch, merging, gamma=1,
            #                       N=training_size)

            print('{0} ||| batch{1} |||  ||| {2}'.format(epoch, b, npvec2str(weights, fnames)))
            # recursive average (Polyak and Juditsky, 1992)
            t = float(epoch - 1)
            avg = t / (t + 1) * avg + 1.0 / (t + 1) * weights
            print('{0} ||| avg{1} |||  ||| {2}'.format(epoch, b, npvec2str(avg, fnames)))

        model = make_models(dict(zip(model.fnames(), avg)), model.extractors())





def main():
    args = get_argparser().parse_args()

    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    core(args)


if __name__ == '__main__':
    main()
