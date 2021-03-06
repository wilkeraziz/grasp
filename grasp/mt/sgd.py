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
from collections import deque, defaultdict
import shutil
import glob



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

from grasp.alg.rescoring import stateless_rescoring
from grasp.alg.chain import apply_filters, group_by_identity, group_by_projection
from grasp.alg.expectation import expected_components

from grasp.scoring.frepr import FComponents
from grasp.scoring.model import ModelView
from grasp.scoring.parameter import SymmetricGuassianPrior, AsymmetricGuassianPrior
from grasp.scoring.fvecfunc import derivation_fvec
from grasp.scoring.parameter import GaussianPrior
from grasp.scoring.util import save_model, save_factorisation, compare_models, compare_factorisations

from grasp.io.results import save_mcmc_yields, save_mcmc_derivations, save_markov_chain

from random import shuffle
from numpy import linalg as LA
from scipy.optimize import minimize
from time import time, strftime
from types import SimpleNamespace

import grasp.mt.pipeline2 as pipeline
from grasp.optimisation.sgd import SGD
from grasp.optimisation.sgd import FlatLearningRateSGD
from grasp.optimisation.sgd import DecayingLearningRateSGD
from grasp.optimisation.sgd import AdaGrad
from grasp.optimisation.sgd import AdaDelta

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
    parser.add_argument("--local-gaussian-mean", type=str, default=None,
                        help="Set the mean of the Gaussian prior over parameters of the local model "
                             "to something other than 0 (format: a weight file)")
    parser.add_argument("--local-gaussian-variance", type=float, default=float('inf'),
                        help="Variance of the Gaussian prior over parameters of the local model")
    parser.add_argument("--nonlocal-gaussian-mean", type=str, default=None,
                        help="Set the mean of the Gaussian prior over parameters of the nonlocal model "
                             "to something other than 0 (format: a weight file)")
    parser.add_argument("--nonlocal-gaussian-variance", type=float, default=float('inf'),
                        help="Variance of the Gaussian prior over parameters of the nonlocal model")
    parser.add_argument("--max-temperature", type=float, default=0.0,
                        help="Annealing / Entropy regularisation")
    parser.add_argument("--min-temperature", type=float, default=0.0,
                        help="Annealing / Entropy regularisation")
    parser.add_argument("--cooling-factor", type=float, default=1.0,
                        help="Cooling factor (see cooling-schedule)")
    parser.add_argument("--cooling-schedule", type=int, default=1,
                        help="If positive, we cool the temperature in epochs multiple of this number")
    parser.add_argument("--cooling-regime", type=int, default=0,
                        help="If positive, specifies the number of parameter updates between cooling actions."
                             "By default we cool at the end of an epoch.")
    parser.add_argument("--assess-epoch0", '-0', action='store_true',
                        help="Assess the initial model")
    parser.add_argument("--eval-schedule", type=int, default=0,
                        help="If positive, specifies the number of parameter updates between assessments."
                             "Note that we always evaluate at the end of an epoch.")
    parser.add_argument("--optimiser", default='adagrad', choices=['flat', 'decaying', 'adagrad', 'adadelta'],
                        type=str,
                        help="Choose the update strategy.")
    parser.add_argument("--ada-epsilon", type=float, default=1e-6,
                        help="Epsilon parameter for AdaGrad and AdaDelta")
    parser.add_argument("--ada-rho", type=float, default=0.95,
                        help="Momentum-like parameter for AdaDelta")
    parser.add_argument("--resume", type=int, default=0,
                        help="Pick up from the end of a certain iteration (avoid it for now!!!)")

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
    group.add_argument('--impsamp-estimates', action='store_true',
                       help='Use the importance sampling estimate of the expected feature vectors')
    group.add_argument('--slice-size',
                       type=int, default=100, metavar='K',
                       help='number of samples from slice (for importance and uniform sampling)')
    group.add_argument('--initial',
                       type=str, default='uniform', choices=['uniform', 'local'],
                       help='how to sample the initial state of the Markov chain')
    group.add_argument('--temperature0',
                       type=float, default=1.0,
                       help='flattens the distribution from where we obtain the initial derivation (for local initialisation only)')
    group.add_argument('--count', action='store_true',
                       help='Count derivations in which slice')
    group.add_argument('--normalised-svars', action='store_true',
                       help="By default slice variables are Gamma distributed over the positive real line."
                            "Alternatively, we can use Beta distributed variables over [0,1]")
    group.add_argument('--shape', nargs=2, default=['const', '1'],
                       help="Unconstrained slice variables are sampled from Gamma(shape > 0, scale > 0) "
                            "or Beta(shape > 0, scale > 0). "
                            "The shape can be either a constant, or it can be itself sampled from Gamma(1, scale > 0)."
                            "For example, use 'const 1' for the former, and 'gamma 1' for the latter.")
    group.add_argument('--scale', nargs=2, default=['const', '1'],
                       help="Unconstrained slice variables are sampled from Gamma(shape > 0, scale > 0) "
                            "or Beta(shape > 0, scale > 0)."
                            "The scale can be either a constant, or it can be itself sampled from Gamma(1, scale > 0)."
                            "For example, use 'const 1' the former, and 'gamma 1' for the latter.")

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
    parser.add_argument('--preparse', action='store_true',
                        help='Pre-compute parse forests and local components for training data.'
                             'We always pre-compute them for validation data.')
    parser.add_argument('--assess-training', action='store_true',
                        help='Assess external loss in training set after each epoch')
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

    if float(args.scale[1]) <= 0:
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


def preprocess_dataset(args, path, grammars, workingdir, joint_model, conditional_model, preparse):
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
    # filter for length
    if args.max_length > 0:
        length_constrained = [seg for seg in unconstrained_data if len(seg.src_tokens()) <= args.max_length]
    else:
        length_constrained = unconstrained_data
    # store references for evaluation purposes
    pipeline.save_references('{0}/refs'.format(workingdir), unconstrained_data)
    # biparse
    if preparse:
        biparsable = biparse(args, workingdir, joint_model, conditional_model, length_constrained)
    else:
        biparsable = None
    return unconstrained_data, length_constrained, biparsable


def sample_derivations(seg, args, staticdir,
                       forest, local_components,
                      model: ModelView, n_top, sample_size):

    # 1. Load pickled objects for the conditional distribution
    #logging.debug('[%d] Loading target forest', seg.id)
    #forest = unpickle_it(forest_path)
    tsort = AcyclicTopSortTable(forest)  # TODO: save/load it?

    #logging.debug('[%d] Loading local components', seg.id)
    #local_components = unpickle_it(components_path)

    logging.debug('[%d] Computing l(d)', seg.id)
    lfunc = TableLookupFunction(np.array([model.local_model().score(fcomp)
                                          for fcomp in local_components], dtype=ptypes.weight))

    from grasp.alg.rescoring2 import SlicedRescoring
    from grasp.alg.rescoring2 import LocalDistribution
    from grasp.scoring.fvecfunc import TableLookupFVecFunction
    from grasp.alg.rescoring2 import SampleReturn

    local_fvecfunc = TableLookupFVecFunction(local_components,
                                             model.local_model().constant(semiring.inside.one),
                                             semiring.inside)

    if not model.nonlocal_model():  # with local models only we can do ancestral sampling (or even exact computations)
        logging.debug('[%d] Ancestral sampling from f(d) = l(d)', seg.id)
        sampler = AncestralSampler(forest, tsort, lfunc)
        raw = sampler.sample(sample_size)

        iidsamples = [None] * len(raw)
        for i, edges in enumerate(raw):
            sample = SampleReturn(edges, lfunc.reduce(semiring.inside.times, edges), local_fvecfunc.reduce(edges))
            iidsamples[i] = sample

        mean = None

    else:  # with nonlocal models we need slice sampling

        # 3. Sample from f(d) = n(d) * l(d)
        logging.debug('[%d] Slice sampling from f(d) = n(d) * l(d)', seg.id)

        goal_maker = GoalRuleMaker(args.goal, args.start, n=n_top)  # 2 or 3
        slice_sampler = SlicedRescoring(model,
                                        LocalDistribution(forest, tsort, lfunc, local_fvecfunc),
                                        semiring.inside,
                                        goal_maker.get_oview(),
                                        OutputView(make_dead_srule()),
                                        log_slice_size=args.count)

        shape_type = args.shape[0]
        shape_parameter = float(args.shape[1])
        scale_type = args.scale[0]
        scale_parameter = float(args.scale[1])

        t0 = time()
        # here samples are represented as sequences of edge ids
        d0, markov_chain, mean_chain = slice_sampler.sample(
            n_samples=sample_size, burn=args.burn, lag=args.lag,
            batch_size=args.slice_size, within=args.within,
            initial=args.initial, temperature0=args.temperature0,
            normalised_svars=args.normalised_svars,
            shape_type=shape_type, shape_parameter=shape_parameter,
            scale_type=scale_type, scale_parameter=scale_parameter)
        logging.info('[%d] Slice sampling took %s seconds', seg.id, time() - t0)

        iidsamples = apply_filters(markov_chain, burn=args.burn, lag=args.lag)
        shuffle(iidsamples)  # make it look like MC samples

        mean = model.constant(semiring.prob.zero)
        for fvec in mean_chain:
            mean = mean.hadamard(fvec, semiring.prob.plus)
        mean = mean.prod(1.0/len(mean_chain))

    n_samples = len(iidsamples)

    # 4. Complete feature vectors and compute expectation
    f_vectors = []
    expected_fvec = model.constant(semiring.prob.zero)
    d_groups = group_by_identity(iidsamples)
    posterior = np.zeros(len(d_groups))
    H = 0.0

    for i, d_group in enumerate(d_groups):
        derivation = d_group.key
        ####local_vec = derivation_fvec(model.local_model(), semiring.inside, local_components, derivation.edges)
        # complete components
        # note that here we are updating derivation.components!
        ####derivation.components = model.merge(local_vec, derivation.components)
        # print('D', derivation.components)
        # incorporate sample frequency
        prob = float(d_group.count) / n_samples
        f_vectors.append(derivation.components.power(prob, semiring.inside))
        expected_fvec = expected_fvec.hadamard(f_vectors[-1], semiring.prob.plus)
        H += prob * np.log(prob)
        posterior[i] = prob

    result = SimpleNamespace()
    result.forest = forest
    result.iidsamples = iidsamples
    result.d_groups = d_groups
    if args.impsamp_estimates and mean is not None:
        result.expected_fvec = mean
    else:
        result.expected_fvec = expected_fvec
    result.f_vectors = f_vectors
    result.posterior = posterior
    result.entropy = -H

    return result


#def expected_features(seg, args, staticdir, forest_path, components_path,
#                      model: ModelView, n_top):
#    result = sample_derivations(seg, args, staticdir, forest_path, components_path,
#                                              model, n_top, sample_size=args.samples[0])
#    return result.expected_fvec


def negative_entropy_derivative(expected_fvec: FComponents, d_groups, posterior, entropy, model: ModelView):

    # Expected[log p(d|x) * fvec(x, d)] +
    #  H(p) * Expected(fvec(x,d))
    # where H(p) = - Expected(log p(d|x))

    derivative = model.constant(semiring.prob.zero)
    for d_group, prob in zip(d_groups, posterior):
        derivation = d_group.key
        derivative = derivative.hadamard(derivation.components.power(prob * np.log(prob), semiring.inside), semiring.prob.plus)
    derivative = derivative.hadamard(expected_fvec.power(entropy, semiring.inside), semiring.prob.plus)
    return derivative


def get_forest_and_local_components(args, seg, joint_model: ModelView, conditional_model: ModelView, staticdir):

    if args.preparse:
        result = SimpleNamespace()
        result.joint = SimpleNamespace()
        result.conditional = SimpleNamespace()
        result.conditional.forest = unpickle_it('{0}/{1}.conditional.forest'.format(staticdir, seg.id))
        result.conditional.components = unpickle_it('{0}/{1}.conditional.components'.format(staticdir, seg.id))
        result.joint.forest = unpickle_it('{0}/{1}.joint.forest'.format(staticdir, seg.id))
        result.joint.components = unpickle_it('{0}/{1}.joint.components'.format(staticdir, seg.id))
    else:
        result = pipeline.biparse(seg, BiparserOptions(args), joint_model, conditional_model)

    return result


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

    data_structures = get_forest_and_local_components(args, seg, joint_model, conditional_model, staticdir)

    if not data_structures.conditional.forest or not data_structures.joint.forest:  # not parsable
        return False, None, None

    t0 = time()
    conditional = sample_derivations(seg, args, staticdir,
                                     data_structures.conditional.forest,
                                     data_structures.conditional.components,
                                     conditional_model, 3,
                                     args.samples[0])
    logging.info('[%d] Conditional gradient took %s seconds', seg.id, time() - t0)

    conditional_vec = np.array(list(conditional.expected_fvec.densify()), dtype=ptypes.weight)

    t0 = time()
    joint = sample_derivations(seg, args, staticdir,
                               data_structures.joint.forest,
                               data_structures.joint.components,
                               joint_model, 2, args.samples[0])
    joint_vec = np.array(list(joint.expected_fvec.densify()), dtype=ptypes.weight)
    logging.info('[%d] Joint gradient took %s seconds', seg.id, time() - t0)

    gradient_vector = conditional_vec - joint_vec

    negH_derivative = negative_entropy_derivative(joint.expected_fvec, joint.d_groups, joint.posterior, joint.entropy, joint_model)
    entropy_vec = np.array(list(negH_derivative.densify()), dtype=ptypes.weight)

    print('[{0}] ||| conditional ||| {1}'.format(seg.id, npvec2str(conditional_vec,
                                                                   conditional_model.fnames())))
    print('[{0}] ||| joint ||| {1}'.format(seg.id, npvec2str(joint_vec,
                                                             joint_model.fnames())))
    print('[{0}] ||| entropy ||| {1}'.format(seg.id, npvec2str(entropy_vec,
                                                               joint_model.fnames())))

    return True, gradient_vector, entropy_vec


def gradient(segments, args, staticdir: str,
             joint_model: ModelView, conditional_model: ModelView) -> np.array:
    logging.info('Computing gradient based on a batch of %d instances using %d workers', len(segments), args.jobs)
    with Pool(args.jobs) as workers:
        derivatives = workers.map(partial(_gradient,
                                      args=args,
                                      staticdir=staticdir,
                                      joint_model=joint_model,
                                      conditional_model=conditional_model),
                              segments)
    L = np.array([likelihood for status, likelihood, entropy in derivatives if status])
    H = np.array([entropy for status, likelihood, entropy in derivatives if status])

    return L, H


@traceit
def _decode(seg, args, joint_model: ModelView, staticdir, outdir, sample_size):
    """
    """

    # load forest and local components
    forest = unpickle_it('{0}/{1}.joint.forest'.format(staticdir, seg.id))
    components = unpickle_it('{0}/{1}.joint.components'.format(staticdir, seg.id))

    result = sample_derivations(seg, args, outdir,
                                forest, components,
                                joint_model,
                                2, sample_size=sample_size)

    # decision rule
    from grasp.mt.pipeline import consensus
    decisions = consensus(seg, result.forest, result.iidsamples)
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


def get_prior_mean_and_variance(model: ModelView, fnames: list,
                                local_prior: GaussianPrior, nonlocal_prior: GaussianPrior):
    mean = []
    var = []
    for name in fnames:
        if model.is_local(name):
            mean.append(local_prior.mean(name))
            var.append(local_prior.var())
        else:
            mean.append(nonlocal_prior.mean(name))
            var.append(nonlocal_prior.var())
    return np.array(mean, dtype=ptypes.weight), np.array(var, dtype=ptypes.weight)


def get_optimiser(parameters, mean, variance, gamma0, epsilon, rho, t=0, optimiser_type='flat') -> SGD:
    if optimiser_type == 'flat':
        return FlatLearningRateSGD(gamma0, t)
    elif optimiser_type == 'decaying':
        return DecayingLearningRateSGD(variance, gamma0, t)
    elif optimiser_type == 'adagrad':
        return AdaGrad(np.zeros(parameters.shape[0], dtype=ptypes.weight),
                       gamma0, t, epsilon=epsilon)
    elif optimiser_type == 'adadelta':
        return AdaDelta(np.zeros(parameters.shape[0], dtype=ptypes.weight),
                        t, epsilon=epsilon, rho=rho)
    else:
        raise ValueError('I do not know this optimiser: %s' % optimiser_type)


def run_assessments(args, epoch, update_number,
                    fnames, avg, joint_model, training, validation,
                    epoch_dir, training_dir, validation_dir):

    # save weights
    with smart_wopen('{0}/weights-{1}.txt'.format(epoch_dir, update_number)) as fw:
        for fname, fvalue in zip(fnames, avg):
            print('{0} {1}'.format(fname, repr(fvalue)), file=fw)

    training_bleu, validation_bleu = None, None

    # decode training set
    if args.assess_training:
        logging.info('Evaluating training set')
        mteval_dir = '{0}/mteval-{1}-{2}'.format(epoch_dir, update_number, args.training_alias)
        os.makedirs(epoch_dir, exist_ok=True)
        training_bleu = decode_and_eval(training, args, joint_model, training_dir, mteval_dir)
        logging.info('Epoch %d (update %d) - Training BLEU %s', epoch, update_number, training_bleu)

    # decode validation set
    if validation:
        logging.info('Evaluating validation set')
        mteval_dir = '{0}/mteval-{1}-{2}'.format(epoch_dir, update_number, args.validation_alias)
        os.makedirs(epoch_dir, exist_ok=True)
        validation_bleu = decode_and_eval(validation, args, joint_model, validation_dir, mteval_dir)
        logging.info('Epoch %d (update %d) - Validation BLEU %s', epoch, update_number, validation_bleu)

    return training_bleu, validation_bleu


def cooling_action(args, iteration, temperature):
    if args.cooling_schedule > 0 and iteration > 0 and iteration % args.cooling_schedule == 0:
        temperature /= args.cooling_factor  # cool it
        if temperature < args.min_temperature:  # check for lower bound
            temperature = args.min_temperature
    return temperature


def prepare_conf_dir(workspace, args):
    """
    Creates a directory associated with a specific training configuration.

    :param workspace:
    :param args:
    :return: directory path
    """
    conf_dirs = glob.glob('{0}/conf*'.format(workspace))
    n_dir = len(conf_dirs) + 1
    new_dir = '{0}/conf{1}'.format(workspace, n_dir)
    os.makedirs(new_dir, exist_ok=False)
    with open('{0}/args.ini'.format(new_dir), 'w') as fo:
        for k, v in vars(args).items():
            print('%s=%r' % (k, v), file=fo)

    if args.local_gaussian_mean:
        shutil.copy(args.local_gaussian_mean, '{0}/local_gaussian_mean'.format(new_dir))
    if args.nonlocal_gaussian_mean:
        shutil.copy(args.nonlocal_gaussian_mean, '{0}/nonlocal_gaussian_mean'.format(new_dir))

    return new_dir


def core(args):

    workspace = make_dirs(args)
    training_dir = '{0}/{1}'.format(workspace, args.training_alias)
    validation_dir = '{0}/{1}'.format(workspace, args.validation_alias)

    if not sanity_checks(args):
        raise FileNotFoundError('One or more files could not be found')

    # We load the complete model
    if args.resume > 0:
        model = pipeline.load_model(args.model, '{0}/epoch{1}/weights.txt'.format(workspace, args.resume),
                                    args.init)
    else:
        model = pipeline.load_model(args.model, args.weights, args.init)

    # Then we get a joint factorisation of the model and a conditional factorisation of the model
    # each factorisation defines a local model and a nonlocal model
    # we assume that exact inference is tractable wrt local models
    # and for nonlocal models we resort to slice sampling
    joint_model, conditional_model = pipeline.get_factorised_models(model, args.factorisation)

    logging.info('Model:\n%s', model)
    logging.info('Joint view:\n%s', joint_model)
    logging.info('Conditional view:\n%s', conditional_model)

    fnames = model.fnames()
    weights = np.array(list(model.weights().densify()), dtype=ptypes.weight)
    dimensionality = len(weights)
    logging.info('Dimensionality: %d', dimensionality)

    # Check consistency with workspace
    if not os.path.exists('{0}/model.ini'.format(workspace)):
        save_model(model, '{0}/model.ini'.format(workspace))
    else:
        # compare it
        if not compare_models(model, '{0}/model.ini'.format(workspace)):
            raise ValueError('You have changed the model into something incompatible with the current workspace.'
                             " Compare '%s' and '%s'" % (args.model, '{0}/model.ini'.format(workspace)))

    if not os.path.exists('{0}/factorisation.ini'.format(workspace)):
        save_factorisation(joint_model, conditional_model, '{0}/factorisation.ini'.format(workspace))
    else:
        if not compare_factorisations(joint_model, conditional_model, '{0}/factorisation.ini'.format(workspace)):
            raise ValueError('You have factorised the model in a way which is incompatible with the current workspace.'
                             " Check '%s'" % '{0}/factorisation.ini'.format(workspace))
    # TODO: in principle the workspace also varies according to options under Parser and Grammar, for now I will leave it to the experimenter

    if args.local_gaussian_mean:
        local_gaussian_prior = AsymmetricGuassianPrior(mean=pipeline.read_weights(args.local_gaussian_mean),
                                                       var=args.local_gaussian_variance)
    else:
        local_gaussian_prior = SymmetricGuassianPrior(mean=0.0, var=args.local_gaussian_variance)

    if args.nonlocal_gaussian_mean:
        nonlocal_gaussian_prior = AsymmetricGuassianPrior(mean=pipeline.read_weights(args.nonlocal_gaussian_mean),
                                                          var=args.nonlocal_gaussian_variance)
    else:
        nonlocal_gaussian_prior = SymmetricGuassianPrior(mean=0.0, var=args.nonlocal_gaussian_variance)

    # make factorisations
    # e.g. joint model contains lookup and stateless
    # e.g. conditional model contains lookup, stateless

    # 1. Load/Parse
    # 1a. Training data (biparsable sentences)
    unconstrained_training, length_constrained_training, biparsable_training = preprocess_dataset(args,
                                                                                                  args.training,
                                                                                                  args.training_grammars,
                                                                                                  training_dir,
                                                                                                  joint_model,
                                                                                                  conditional_model,
                                                                                                  preparse=args.preparse)

    # 1b. Validation data (if applicable)
    if args.validation:  # we do not need to biparse the validation set
        _validation, validation, _ = preprocess_dataset(args, args.validation, args.validation_grammars, validation_dir,
                                                     joint_model, conditional_model=None, preparse=True)
    else:
        _validation, validation = [], []

    if biparsable_training is not None:
        logging.info('This model can generate %d out of %d training instances', len(biparsable_training),
                     len(length_constrained_training))
        training = biparsable_training
    else:
        logging.info('We selected %d training instances', len(length_constrained_training))
        training = length_constrained_training

    logging.info('We selected %d out of %d validation instances', len(validation),
                 len(_validation))

    # 2. SGD
    # create folder confN where epochs will be stored
    conf_dir = prepare_conf_dir(workspace, args)

    prior_mean, prior_var = get_prior_mean_and_variance(joint_model, fnames,
                                                        local_gaussian_prior, nonlocal_gaussian_prior)
    avg = np.zeros(dimensionality, dtype=ptypes.weight)

    logging.info('Parameters: %s', npvec2str(weights, fnames))
    logging.info('Prior mean: %s', npvec2str(prior_mean, fnames))
    logging.info('Prior var:  %s', npvec2str(prior_var, fnames))

    t = 0
    n_batches = np.ceil(len(training) / float(args.batch_size))

    if args.resume == 0:
        # save initial weights
        epoch_dir = '{0}/epoch{1}'.format(conf_dir, 0)
        os.makedirs(epoch_dir, exist_ok=True)
        # save weights
        with smart_wopen('{0}/weights.txt'.format(epoch_dir)) as fw:
            for fname, fvalue in zip(fnames, weights):
                print('{0} {1}'.format(fname, repr(fvalue)), file=fw)

        # evaluate with initial weights
        if args.assess_epoch0 and validation:
            logging.info('Evaluating validation set')
            mteval_dir = '{0}/mteval-{1}'.format(epoch_dir, args.validation_alias)
            os.makedirs(epoch_dir, exist_ok=True)
            bleu = decode_and_eval(validation, args, joint_model, validation_dir, mteval_dir)
            logging.info('Epoch %d - Validation BLEU %s', 0, bleu)

    temperature = args.max_temperature

    optimiser = get_optimiser(weights, prior_mean, prior_var,
                              args.learning_rate, args.ada_epsilon, args.ada_rho,
                              optimiser_type=args.optimiser)

    last_assessment = 0

    best_bleu = None

    # udpate parameters
    for epoch in range(1, args.maxiter + 1):

        # where we store everything related to this iteration
        epoch_dir = '{0}/epoch{1}'.format(conf_dir, epoch)
        os.makedirs(epoch_dir, exist_ok=True)

        # first we get a random permutation of the training data
        shuffle(training)

        # then we operate in mini-batches
        for b, first in enumerate(range(0, len(training), args.batch_size), 1):
            # gather segments in this batch
            batch = [training[ith] for ith in range(first, min(first + args.batch_size, len(training)))]
            # batch importance
            batch_coeff = float(len(batch)) / len(training)
            # learning rate
            #learning_rate = get_learning_rates(args.decaying_lrate, gamma0, t, prior_var,
            #                                   batch_coeff=1.0) # I am deliberately not adjusting for batch size
            logging.info('Epoch %d/%d - Time %d - Batch %d/%d (%.4f) - Temperature %f', epoch, args.maxiter,
                         t,
                         b,
                         n_batches,
                         batch_coeff,
                         temperature)
            #logging.info('Learning rate %d: %s', t, npvec2str(learning_rate, fnames))
            #print([seg.id for seg in batch])
            #batch_dir = '{0}/batch{0}'.format(epoch, b)
            #os.makedirs(batch_dir, exist_ok=True)

            likelihood_gradient_vectors, negative_entropy_gradient_vectors = gradient(batch, args, training_dir, joint_model, conditional_model)
            if len(likelihood_gradient_vectors) == 0:
                logging.info('Epoch %d - Batch %d - skipping batch for no segment was parsable', epoch, b)
                # when we skip a batch we do not make parameter updates, thus we do not advance t
                continue

            likelihood_gradient = likelihood_gradient_vectors.mean(0)  # normalise for batch size
            negative_entropy_gradient = negative_entropy_gradient_vectors.mean(0)  # normalise for batch size

            # The gradient of the prior with respect to component k is: - (w_k - mean_k) / var
            # we also normalise for the importance of the batch
            prior_gradient = - (weights - prior_mean) / prior_var * batch_coeff
            # The gradient of the objective function decomposes as the gradient of the (log) likelihood plus
            # the gradient of the (log) prior
            gradient_vec = likelihood_gradient + prior_gradient
            # if we have entropy regularisation, then we also sum the gradient of the negative entropy
            if temperature != 0:
                gradient_vec += temperature * negative_entropy_gradient

            # finally, the SGD (maximisation) update is w = w + learning_rate * gradient
            #weights += learning_rate * gradient_vec
            weights = optimiser.update(weights, gradient_vec)

            print('{0} ||| batch{1} ||| L2={2} ||| {3} ||| '.format(epoch, b,
                                                                    np.linalg.norm(weights - prior_mean, 2),
                                                                    npvec2str(weights, fnames)))
            # recursive average (Polyak and Juditsky, 1992)
            avg = t / (t + 1) * avg + 1.0 / (t + 1) * weights
            t += 1
            print('{0} ||| avg{1} ||| L2={2} ||| {3}'.format(epoch, t, np.linalg.norm(avg - prior_mean, 2),
                                                             npvec2str(avg, fnames)))

            # update models
            model = make_models(dict(zip(model.fnames(), avg)), model.extractors())
            joint_model, conditional_model = pipeline.get_factorised_models(model, args.factorisation)

            # sometimes we opt for assessing the model after a number of parameter updates
            if args.eval_schedule > 0 and t % args.eval_schedule == 0:
                _, validation_bleu = run_assessments(args, epoch, t, fnames, avg, joint_model, training, validation,
                                                     epoch_dir, training_dir, validation_dir)
                last_assessment = t
                if validation_bleu is not None:
                    if best_bleu is None or validation_bleu >= best_bleu.bleu:
                        best_bleu = SimpleNamespace()
                        best_bleu.epoch = epoch
                        best_bleu.update = t
                        best_bleu.bleu = validation_bleu
                        best_bleu.weights = avg

            # sometimes we opt for cooling after a number of parameter updates
            if args.cooling_regime > 0:
                temperature = cooling_action(args, t, temperature)

        # we always run assessments at the end of an epoch
        if last_assessment != t:  # except if eval_schedule coincided with the end of the epoch
            _, validation_bleu = run_assessments(args, epoch, t, fnames, avg, joint_model, training, validation,
                                                 epoch_dir, training_dir, validation_dir)
            if validation_bleu is not None:
                if best_bleu is None or validation_bleu >= best_bleu.bleu:
                    best_bleu = SimpleNamespace()
                    best_bleu.epoch = epoch
                    best_bleu.update = t
                    best_bleu.bleu = validation_bleu
                    best_bleu.weights = avg

        # sometimes we opt for cooling at the end of epochs
        if args.cooling_regime <= 0:
            temperature = cooling_action(args, epoch, temperature)

    # save final weights
    with smart_wopen('{0}/weights-final.txt'.format(conf_dir)) as fw:
        for fname, fvalue in zip(fnames, avg):
            print('{0} {1}'.format(fname, repr(fvalue)), file=fw)

    if best_bleu is not None:
        with smart_wopen('{0}/weights-best.txt'.format(conf_dir)) as fw:
            print('# epoch=%d update=%d BLEU=%f' % (best_bleu.epoch, best_bleu.update, best_bleu.bleu), file=fw)
            for fname, fvalue in zip(fnames, best_bleu.weights):
                print('{0} {1}'.format(fname, repr(fvalue)), file=fw)


def main():
    args = get_argparser().parse_args()

    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    core(args)


if __name__ == '__main__':
    main()
