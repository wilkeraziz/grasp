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
from grasp.loss.fast_bleu import doc_bleu

import grasp.ptypes as ptypes

from grasp.recipes import smart_ropen, smart_wopen, make_unique_directory, pickle_it, unpickle_it, traceit, timeit

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
from grasp.scoring.model import ModelView
from grasp.scoring.parameter import SymmetricGuassianPrior, AsymmetricGuassianPrior

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
    #parser.add_argument("--L2", type=float, default=0.0,
    #                    help="Weight of L2 regulariser in target optimisation")
    parser.add_argument("--learning-rate", type=float, default=1.0,
                        help="Learning rate")
    parser.add_argument("--gaussian-mean", type=str, default=None,
                        help="Set the mean of the Gaussian prior over parameters to something other than 0 (format: a weight file)")
    parser.add_argument("--gaussian-variance", type=float, default=1.0,
                        help="Variance of the Gaussian prior over parameters")

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
    group.add_argument('--max-length',
                       type=int, default=-1, metavar='N',
                       help='if positive, impose a maximum sentence length')


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

def cmd_slice(group):
    group.add_argument('--chains',
                       type=int, default=1, metavar='K',
                       help='number of random restarts')
    group.add_argument('--samples', type=int, nargs=2, default=[100, 100],
                       metavar='N N',
                       help='number of samples for training (MLE) and testing (consensus)')
    group.add_argument('--lag',
                       type=int, default=1, metavar='I',
                       help='lag between samples')
    group.add_argument('--burn',
                       type=int, default=10, metavar='N',
                       help='number of initial samples to be discarded (applies after lag)')
    group.add_argument('--within',
                       type=str, default='importance', choices=['exact', 'importance', 'uniform', 'cimportance'],
                       help='how to sample within the slice')
    group.add_argument('--slice-size',
                       type=int, default=100, metavar='K',
                       help='number of samples from slice (for importance and uniform sampling)')
    group.add_argument('--initial',
                       type=str, default='uniform', choices=['uniform', 'local'],
                       help='how to sample the initial state of the Markov chain')
    group.add_argument('--temperature0',
                       type=float, default=1.0,
                       help='flattens the distribution from where we obtain the initial derivation (for local initialisation only)')
    group.add_argument('--gamma-shape', type=float, default=0,
                       help="Unconstrained slice variables are sampled from a Gamma(shape, scale)."
                            "By default, the shape is the number of local components.")
    group.add_argument('--gamma-scale', nargs=2, default=['const', '1'],
                       help="Unconstrained slice variables are sampled from a Gamma(shape, scale)."
                            "The scale can be either a constant, or it can be itself sampled from a Gamma(1, scale')."
                            "For the former use for example 'const 1', for the latter use 'gamma 1'")

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
    parser.add_argument("--factorisation", '-F', type=str,
                        help="change the default factorisation of the model")
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
    cmd_slice(parser.add_argument_group('Slice sampler'))
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
        logging.error('Model specification not found: %s', args.model)
        failed = True
    if args.weights and not os.path.exists(args.weights):
        logging.error('Model parameters not found: %s', args.proxy_weights)
        failed = True

    if float(args.gamma_scale[1]) <= 0:
        raise ValueError('The scale parameter of a Gamma distribution must be strictly positive')

    #if args.joint and not os.path.exists(args.joint):
    #    logging.error('Joint model factorisation not found: %s', args.joint)
    #    failed = True
    #if args.conditional and not os.path.exists(args.conditional):
    #    logging.error('Conditional model factorisation not found: %s', args.conditional)
    #    failed = True
    return not failed

class BiparserOptions:

    def __init__(self, args):
        self.extra_grammars = args.extra_grammar
        self.glue_grammars = args.glue_grammar
        self.pass_through = args.pass_through
        self.default_symbol = args.default_symbol
        self.goal = args.goal
        self.start = args.start
        self.max_span = args.max_span


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


def biparse(args, workingdir: str, joint_model: ModelView, conditional_model: ModelView, segments) -> list:
    """
    This method bi-parses a whole data set and saves forests and components to a working directory.

    :param args: command line arguments
    :param workingdir: where to save everything
    :param joint_model: a factorised view of the joint model
    :param conditional_model: a factorised view of the conditional model
    :param segments: a data set
    :return: parsable segments
    """
    logging.info('Parsing %d training instances using %d workers', len(segments), args.jobs)
    options = BiparserOptions(args)
    with Pool(args.jobs) as workers:
        feedback = workers.map(partial(_biparse,
                                       options=options,
                                       joint_model=joint_model,
                                       conditional_model=conditional_model,
                                       workingdir=workingdir,
                                       redo=args.redo,
                                       log=logging.info),
                               segments)
    return [seg for seg, status in zip(segments, feedback) if status]


def preprocess_dataset(args, path, grammars, workingdir, joint_model, conditional_model):
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
    # filter for length
    if args.max_length > 0:
        length_constrained = [seg for seg in unconstrained_data if len(seg.src_tokens()) <= args.max_length]
    else:
        length_constrained = unconstrained_data
    # biparse
    biparsable = biparse(args, workingdir, joint_model, conditional_model, length_constrained)
    return unconstrained_data, length_constrained, biparsable


def sample_derivations(seg, args, staticdir, forest_path, components_path,
                      model: ModelView, n_top, sample_size):
    # 1. Load pickled objects for the conditional distribution
    logging.debug('[%d] Loading target forest', seg.id)
    forest = unpickle_it(forest_path)
    tsort = AcyclicTopSortTable(forest)  # TODO: save/load it?

    logging.debug('[%d] Loading local components', seg.id)
    local_components = unpickle_it(components_path)

    logging.debug('[%d] Computing l(d)', seg.id)
    lfunc = TableLookupFunction(np.array([model.local_model().score(fcomp)
                                          for fcomp in local_components], dtype=ptypes.weight))

    if not model.nonlocal_model():  # with local models only we can do ancestral sampling (or even exact computations)
        sampler = AncestralSampler(forest, tsort, lfunc)
        raw = sampler.sample(sample_size)

        # TODO: create code to produce SampleReturn objects and reconstruct components
        from grasp.alg.rescoring import SampleReturn
        iidsamples = [None] * len(raw)
        for i, edges in enumerate(raw):
            score = lfunc.reduce(semiring.inside.times, edges)
            sample = SampleReturn(edges, score, FComponents([]))
            iidsamples[i] = sample

    else:  # with nonlocal models we need slice sampling

        # 3. Sample from f(d) = n(d) * l(d)
        logging.debug('[%d] Sampling from f(d) = n(d) * l(d)', seg.id)

        goal_maker = GoalRuleMaker(args.goal, args.start, n=n_top)  # 2 or 3

        slice_sampler = SlicedRescoring(forest,
                                        lfunc,
                                        tsort,
                                        TableLookupScorer(model.nonlocal_model().lookup),
                                        StatelessScorer(model.nonlocal_model().stateless),
                                        StatefulScorer(model.nonlocal_model().stateful),
                                        semiring.inside,
                                        goal_maker.get_oview(),
                                        OutputView(make_dead_srule()))

        if args.gamma_shape > 0:
            gamma_shape = args.gamma_shape
        else:
            gamma_shape = len(model.local_model())  # number of local components
        gamma_scale_type = args.gamma_scale[0]
        gamma_scale_parameter = float(args.gamma_scale[1])


        t0 = time()
        # here samples are represented as sequences of edge ids
        d0, markov_chain = slice_sampler.sample(n_samples=sample_size,
                                                batch_size=args.slice_size,
                                                within=args.within,
                                                initial=args.initial,
                                                gamma_shape=gamma_shape,
                                                gamma_scale_type=gamma_scale_type,
                                                gamma_scale_parameter=gamma_scale_parameter,
                                                burn=args.burn,
                                                lag=args.lag,
                                                temperature0=args.temperature0)
        logging.info('[%d] Slice sampling took %s seconds', seg.id, time() - t0)

        iidsamples = apply_filters(markov_chain,
                                burn=args.burn,
                                lag=args.lag)
        shuffle(iidsamples)  # make it look like MC samples

    n_samples = len(iidsamples)

    # 4. Complete feature vectors and compute expectation
    f_vectors = []
    expected_fvec = model.constant(semiring.prob.zero)
    d_groups = group_by_identity(iidsamples)
    for d_group in d_groups:
        derivation = d_group.key
        # print(derivation)
        # reconstruct components
        # TODO: create code to reconstruct components
        local_vec = model.local_model().constant(semiring.inside.one)
        for e in derivation.edges:
            # print('e_%d' % e, coarse_components[e])
            local_vec = local_vec.hadamard(local_components[e], semiring.inside.times)
        # print('D', comps)

        # complete components
        # note that here we are updating derivation.components!
        derivation.components = model.merge(local_vec, derivation.components)
        # print('D', derivation.components)
        # incorporate sample frequency
        f_vectors.append(derivation.components.power(float(d_group.count) / n_samples, semiring.inside))
        expected_fvec = expected_fvec.hadamard(f_vectors[-1], semiring.prob.plus)

    return forest, iidsamples, d_groups, f_vectors, expected_fvec


def expected_features(seg, args, staticdir, forest_path, components_path,
                      model: ModelView, n_top):
    _, _, _, _, expected = sample_derivations(seg, args, staticdir, forest_path, components_path,
                                              model, n_top, sample_size=args.samples[0])
    return expected


@traceit
def _gradient(seg: SegmentMetaData, args, staticdir: str,
              joint_model: ModelView, conditional_model: ModelView) -> np.array:
    """
    Compute the gradient vector:
        Expectation(feature vector; conditional distribution) - Expectation(feature vector; joint distribution)

    :param seg:
    :param args:
    :param staticdir:
    :param joint_model:
    :param conditional_model:
    :return: np.array (gradient vector)
    """

    conditional_vec = expected_features(seg, args, staticdir,
                                        '{0}/{1}.conditional.forest'.format(staticdir, seg.id),
                                        '{0}/{1}.conditional.components'.format(staticdir, seg.id),
                                        conditional_model, 3)

    conditional_vec = np.array(list(conditional_vec.densify()), dtype=ptypes.weight)

    joint_vec = expected_features(seg, args, staticdir,
                                  '{0}/{1}.joint.forest'.format(staticdir, seg.id),
                                  '{0}/{1}.joint.components'.format(staticdir, seg.id),
                                  joint_model, 2)
    joint_vec = np.array(list(joint_vec.densify()), dtype=ptypes.weight)
    gradient_vector = conditional_vec - joint_vec
    return gradient_vector


def gradient(segments, args, staticdir: str,
             joint_model: ModelView, conditional_model: ModelView) -> np.array:
    logging.info('Computing gradient based on a batch of %d instances using %d workers', len(segments), args.jobs)
    with Pool(args.jobs) as workers:
        vectors = workers.map(partial(_gradient,
                                      args=args,
                                      staticdir=staticdir,
                                      joint_model=joint_model,
                                      conditional_model=conditional_model),
                              segments)
    return np.array(vectors)


@traceit
def _decode(seg, args, joint_model: ModelView, staticdir, outdir, sample_size):
    """
    """

    # load forest and local components

    forest, iidsamples, d_groups, f_vectors, expected_fvec = sample_derivations(seg, args, outdir,
                                 '{0}/{1}.joint.forest'.format(staticdir, seg.id),
                                 '{0}/{1}.joint.components'.format(staticdir, seg.id),
                                 joint_model,
                                 2, sample_size=sample_size)

    # decision rule
    from grasp.mt.pipeline import consensus
    decisions = consensus(seg, forest, iidsamples)
    return decisions


def decode_and_eval(segments: 'list[SegmentMetaData]', args, joint_model: ModelView, staticdir, outdir):
    logging.info('Consensus decoding (and evaluation of) %d segments using %d workers', len(segments), args.jobs)
    os.makedirs(outdir, exist_ok=True)
    with Pool(args.jobs) as workers:
        all_decisions = workers.map(partial(_decode,
                                        args=args,
                                        joint_model=joint_model,
                                        staticdir=staticdir,
                                        outdir=outdir,
                                        sample_size=args.samples[1]),
                                segments)

    one_best = []
    with smart_wopen('{0}/translations'.format(outdir)) as fout:
        with smart_wopen('{0}/translations.log.gz'.format(outdir)) as flog:
            for seg, decisions in zip(segments, all_decisions):
                # first decision, third field (translation string)
                print(decisions[0][2], file=fout)
                one_best.append(decisions[0][2])
                print(seg.to_sgm(), file=flog)
                print('#loss\tposterior\tdecision', file=flog)
                for loss, posterior, decision in decisions:
                    print('{0}\t{1}\t{2}'.format(loss, posterior, decision), file=flog)
                print(file=flog)

    bleu, pn, bp = doc_bleu([decision.split() for decision in one_best],
                            [[ref.split() for ref in seg.refs] for seg in segments],
                            max_order=args.bleu_order,
                            smoothing=args.bleu_smoothing)

    with open('{0}/bleu'.format(outdir), 'w') as fbleu:
        print('{0}\t{1}\t{2}'.format(bleu, ' '.join(str(x) for x in pn), bp), file=fbleu)

    return bleu


def core(args):

    workspace = make_dirs(args)
    training_dir = '{0}/{1}'.format(workspace, args.training_alias)
    validation_dir = '{0}/{1}'.format(workspace, args.validation_alias)

    if not sanity_checks(args):
        raise FileNotFoundError('One or more files could not be found')

    # We load the complete model
    model = pipeline.load_model(args.model, args.weights, args.init)
    # Then we get a joint factorisation of the model and a conditional factorisation of the model
    # each factorisation defines a local model and a nonlocal model
    # we assume that exact inference is tractable wrt local models
    # and for nonlocal models we resort to slice sampling
    joint_model, conditional_model = pipeline.get_factorised_models(model, args.factorisation)

    logging.info('Model:\n%s', model)
    logging.info('Joint view:\n%s', joint_model)
    logging.info('Conditional view:\n%s', conditional_model)

    if args.gaussian_mean:
        gaussian_prior = AsymmetricGuassianPrior(mean=pipeline.read_weights(args.gaussian_mean), var=args.gaussian_variance)
    else:
        gaussian_prior = SymmetricGuassianPrior(mean=0.0, var=args.gaussian_variance)


    # make factorisations
    # e.g. joint model contains lookup and stateless
    # e.g. conditional model contains lookup, stateless

    # 1. Load/Parse
    # 1a. Training data (biparsable sentences)
    _training, _, training = preprocess_dataset(args, args.training, args.training_grammars, training_dir,
                                             joint_model, conditional_model)
    # 1b. Validation data (if applicable)
    if args.validation:  # we do not need to biparse the validation set
        _validation, validation, _ = preprocess_dataset(args, args.validation, args.validation_grammars, validation_dir,
                                                     joint_model, conditional_model=None)
    else:
        _validation, validation = [], []
    logging.info('This model can generate %d out of %d training instances', len(training),
                 len(_training))
    logging.info('We selected %d out of %d validation instances', len(validation),
                 len(_validation))

    # 2. SGD
    fnames = model.fnames()
    #dimensionality = len(fnames)  # TODO: Sparse SGD
    weights = np.array(list(model.weights().densify()), dtype=ptypes.weight)
    prior_mean = gaussian_prior.mean_vector(fnames)
    logging.info('Parameters: %s', npvec2str(weights, fnames))
    logging.info('Prior: variance=%s mean=(%s)', gaussian_prior.var(), npvec2str(prior_mean, fnames))
    dimensionality = len(weights)
    avg = np.zeros(dimensionality, dtype=ptypes.weight)
    learning_rate = args.learning_rate

    t = 0
    n_batches = np.ceil(len(training) / float(args.batch_size))

    # save initial weights
    epoch_dir = '{0}/epoch{1}'.format(workspace, 0)
    os.makedirs(epoch_dir, exist_ok=True)
    # save weights
    with smart_wopen('{0}/weights.txt'.format(epoch_dir)) as fw:
        for fname, fvalue in zip(fnames, avg):
            print('{0} {1}'.format(fname, repr(fvalue)), file=fw)

    # evaluate with initial weights
    if validation:
        logging.info('Evaluating validation set')
        mteval_dir = '{0}/mteval-{1}'.format(epoch_dir, args.validation_alias)
        os.makedirs(epoch_dir, exist_ok=True)
        bleu = decode_and_eval(validation, args, joint_model, validation_dir, mteval_dir)
        logging.info('Epoch %d - Validation BLEU %s', 0, bleu)

    # udpate parameters
    for epoch in range(1, args.maxiter + 1):

        # where we store everything related to this iteration
        epoch_dir = '{0}/epoch{1}'.format(workspace, epoch)
        os.makedirs(epoch_dir, exist_ok=True)

        # first we get a random permutation of the training data
        shuffle(training)

        # then we operate in mini-batches
        for b, first in enumerate(range(0, len(training), args.batch_size), 1):
            # gather segments in this batch
            batch = [training[ith] for ith in range(first, min(first + args.batch_size, len(training)))]
            logging.info('Epoch %d/%d - Batch %d/%d', epoch, args.maxiter, b, n_batches)
            #print([seg.id for seg in batch])
            #batch_dir = '{0}/batch{0}'.format(epoch, b)
            #os.makedirs(batch_dir, exist_ok=True)

            gradient_vectors = gradient(batch, args, training_dir, joint_model, conditional_model)
            gradient_vector = gradient_vectors.mean(0)  # normalise for batch size
            #print(npvec2str(gradient_vector, fnames))
            # incorporate regulariser
            regulariser = (weights - prior_mean) / gaussian_prior.var()
            batch_coeff = float(len(batch)) / len(training)  # batch importance
            gradient_vector -= regulariser * batch_coeff
            # maximisation update w = w + learning_rate * w
            weights += learning_rate * gradient_vector

            print('{0} ||| batch{1} ||| L2={2} ||| {3} ||| '.format(epoch, b, np.linalg.norm(weights - prior_mean, 2),
                                                                    npvec2str(weights, fnames)))
            # recursive average (Polyak and Juditsky, 1992)
            avg = t / (t + 1) * avg + 1.0 / (t + 1) * weights
            t += 1
            print('{0} ||| avg{1} ||| L2={2} ||| {3}'.format(epoch, t, np.linalg.norm(avg - prior_mean, 2),
                                                             npvec2str(avg, fnames)))


            # update models
            model = make_models(dict(zip(model.fnames(), avg)), model.extractors())
            joint_model, conditional_model = pipeline.get_factorised_models(model, args.factorisation)

        # save weights
        with smart_wopen('{0}/weights.txt'.format(epoch_dir)) as fw:
            for fname, fvalue in zip(fnames, avg):
                print('{0} {1}'.format(fname, repr(fvalue)), file=fw)

        # decode validation set
        if validation:
            logging.info('Evaluating validation set')
            mteval_dir = '{0}/mteval-{1}'.format(epoch_dir, args.validation_alias)
            os.makedirs(epoch_dir, exist_ok=True)
            bleu = decode_and_eval(validation, args, joint_model, validation_dir, mteval_dir)
            logging.info('Epoch %d - Validation BLEU %s', epoch, bleu)


def main():
    args = get_argparser().parse_args()

    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    core(args)


if __name__ == '__main__':
    main()
