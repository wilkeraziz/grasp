"""
:Authors: - Wilker Aziz
"""
import logging
import argparse
import sys

"""

:Authors: - Wilker Aziz
"""

import argparse
import logging
import sys
import itertools
import os
import numpy as np
import traceback
from multiprocessing import Pool
from functools import partial

import grasp.ptypes as ptypes

from grasp.recipes import smart_ropen, smart_wopen, make_unique_directory, pickle_it, unpickle_it

from grasp.scoring.scorer import TableLookupScorer, StatelessScorer, StatefulScorer
from grasp.scoring.model import Model, make_models
from grasp.scoring.util import read_weights

from grasp.mt.cdec_format import load_grammar
from grasp.mt.util import load_feature_extractors, GoalRuleMaker
from grasp.mt.util import save_forest, save_ffs, load_ffs, make_dead_srule, make_batches, number_of_batches
from grasp.mt.segment import SegmentMetaData
from grasp.mt.input import make_input

import grasp.semiring as semiring
from grasp.semiring.operator import FixedRHS

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
from grasp.alg.chain import apply_filters, group_by_identity, group_by_projection

from grasp.scoring.frepr import FComponents

from grasp.io.results import save_mcmc_yields, save_mcmc_derivations, save_markov_chain

from random import shuffle
from numpy import linalg as LA
from scipy.optimize import minimize
from time import time, strftime


def npvec2str(nparray, fnames=None):
    """converts an array of feature values into a string (fnames can be provided)"""
    if fnames is None:
        return ' '.join(str(fvalue) for fvalue in nparray)
    else:
        return ' '.join('{0}={1}'.format(fname, fvalue) for fname, fvalue in zip(fnames, nparray))


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
    parser.add_argument('--init', type=str, default=None,
                        help="use 'uniform' for uniform weights, 'random' for random weights, or choose a default weight")
    parser.add_argument("--resume", type=int, default=0,
                        help="Resume from a certain iteration (requires the config file of the preceding run)")


def cmd_external(parser):
    parser.add_argument('--scoring-tool', type=str,
                        default='/Users/waziz/workspace/github/cdec/mteval/fast_score',
                        help='a scoring tool such as fast_score')


def cmd_sgd(parser):
    parser.add_argument("--sgd", type=int, nargs=2, default=[10, 10],
                        help="Number of iterations and function evaluations for target optimisation")
    parser.add_argument("--tol", type=float, nargs=2, default=[1e-9, 1e-9],
                        help="f-tol and g-tol in target optimisation")
    parser.add_argument("--L2", type=float, default=0.0,
                        help="Weight of L2 regulariser in target optimisation")


def cmd_logging(parser):
    parser.add_argument('--save-chain',
                        action='store_true', default=0,
                        help='store the complete Markov chain')
    parser.add_argument('--save-d',
                        action='store_true', default=0,
                        help='store sampled derivations (after MCMC filters apply)')
    parser.add_argument('--save-y',
                        action='store_true', default=0,
                        help='store sampled translations (after MCMC filters apply)')
    parser.add_argument('--verbose', '-v',
                        action='count', default=0,
                        help='increase the verbosity level')


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


def cmd_slice(group):
    group.add_argument('--chains',
                       type=int, default=1, metavar='K',
                       help='number of random restarts')
    group.add_argument('--samples',
                       type=int, default=100, metavar='N',
                       help='number of samples')
    group.add_argument('--lag',
                       type=int, default=1, metavar='I',
                       help='lag between samples')
    group.add_argument('--burn',
                       type=int, default=10, metavar='N',
                       help='number of initial samples to be discarded (applies after lag)')
    group.add_argument('--within',
                       type=str, default='importance', choices=['exact', 'importance', 'uniform', 'cimportance'],
                       help='how to sample within the slice')
    group.add_argument('--batch',
                       type=int, default=100, metavar='K',
                       help='number of samples per slice (for importance and uniform sampling)')
    group.add_argument('--initial',
                       type=str, default='uniform', choices=['uniform', 'local'],
                       help='how to sample the initial state of the Markov chain')
    group.add_argument('--temperature0',
                       type=float, default=1.0,
                       help='flattens the distribution from where we obtain the initial derivation (for local initialisation only)')
    group.add_argument('--prior', nargs=2,
                       default=['sym', '0.1'],
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
                            "The later are computed based on the local distribution over incoming edges.")


def get_argparser():
    parser = argparse.ArgumentParser(description='Training by MLE',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #parser.add_argument('config', type=str, help="configuration file")
    parser.add_argument("workspace",
                        type=str, default=None,
                        help="where samples can be found and where decisions are placed")
    parser.add_argument("dev", type=str,
                        help="development set")
    parser.add_argument("--alias", type=str,
                        help="an alias for the experiment")
    parser.add_argument("--jobs", type=int, default=2, help="number of processes")
    parser.add_argument('--dev-alias', type=str, default='dev',
            help='Change the alias of the dev set')
    parser.add_argument("--devtest", type=str,
                        help="devtest set")
    parser.add_argument('--devtest-alias', type=str, default='devtest',
            help='Change the alias of the devtest set')
    parser.add_argument('--experiment',
                        type=str,
                        help='folder within the workspace where results are stored'
                             'by default we use a timestamp and a random suffix')
    parser.add_argument('--redo', action='store_true',
                        help='overwrite already computed files (by default we do not repeat computation)')
    cmd_model(parser.add_argument_group('Model'))
    cmd_parser(parser.add_argument_group('Parser'))
    cmd_grammar(parser.add_argument_group('Grammar'))
    cmd_optimisation(parser.add_argument_group('Parameter optimisation by coordinate descent'))
    cmd_sgd(parser.add_argument_group('Optimisation by SGD'))
    cmd_slice(parser.add_argument_group('Slice sampler'))
    cmd_external(parser.add_argument_group('External tools'))
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

    staticdir = '{0}/static'.format(outdir)
    os.makedirs(staticdir, exist_ok=exist_ok)
    dynamicdir = '{0}/iterations'.format(outdir)
    os.makedirs(dynamicdir, exist_ok=exist_ok)

    return outdir, staticdir, dynamicdir


def prepare_for_parsing(seg, args):
    """
    Load grammars (i.e. main, extra, glue, passthrough) and prepare input FSA.
    :param seg:
    :param args:
    :return: (normal grammars, glue grammars, input DFA)
    """
    logging.debug('[%d] Loading grammars', seg.id)
    # 1. Load grammars
    #  1a. additional grammars
    extra_grammars = []
    if args.extra_grammar:
        for grammar_path in args.extra_grammar:
            logging.debug('[%d] Loading additional grammar: %s', seg.id, grammar_path)
            grammar = load_grammar(grammar_path)
            logging.debug('[%d] Additional grammar: productions=%d', seg.id, len(grammar))
            extra_grammars.append(grammar)
    #  1b. glue grammars
    glue_grammars = []
    if args.glue_grammar:
        for glue_path in args.glue_grammar:
            logging.debug('[%d] Loading glue grammar: %s', seg.id, glue_path)
            glue = load_grammar(glue_path)
            logging.debug('[%d] Glue grammar: productions=%d', seg.id, len(glue))
            glue_grammars.append(glue)
    #  1c. main grammar
    main_grammar = load_grammar(seg.grammar)

    logging.debug('[%d] Preparing source FSA: %s', seg.id, seg.src)
    # 2. Make input FSA and a pass-through grammar for the given segment
    #  2a. pass-through grammar
    _, pass_grammar = make_input(seg,
                                 list(itertools.chain([main_grammar], extra_grammars, glue_grammars)),
                                 semiring.inside,
                                 args.default_symbol)
    #  2b. input FSA
    input_dfa = make_dfa(seg.src_tokens())
    logging.debug('[%d] Source FSA: states=%d arcs=%d', seg.id, input_dfa.n_states(), input_dfa.n_arcs())

    #  3a. put all (normal) grammars together
    if args.pass_through:
        grammars = list(itertools.chain([main_grammar], extra_grammars, [pass_grammar]))
    else:
        grammars = list(itertools.chain([main_grammar], extra_grammars))

    return grammars, glue_grammars, input_dfa


def t_biparse(seg, args, staticdir, model):
    try:
        return biparse(seg, args, staticdir, model)
    except:
        raise Exception('job={0} exception={1}'.format(seg.id,
                                                       ''.join(traceback.format_exception(*sys.exc_info()))))


def biparse(seg, args, staticdir, model):
    # this procedure produces the following files
    general_files = ['{0}/{1}.src.forest'.format(staticdir, seg.id),
                     '{0}/{1}.hyp.ffs.rule'.format(staticdir, seg.id),
                     '{0}/{1}.hyp.ffs.stateless'.format(staticdir, seg.id),
                     '{0}/{1}.hyp.forest'.format(staticdir, seg.id)]
    ref_files = ['{0}/{1}.ref.ffs.all'.format(staticdir, seg.id),
                 '{0}/{1}.ref.forest'.format(staticdir, seg.id)]
    # check for redundant work
    if all(os.path.exists(path) for path in general_files) and not args.redo:
        if all(os.path.exists(path) for path in ref_files):
            logging.info('Reusing forests for segment %d', seg.id)
            return True   # parsable
        else:
            return False  # not parsable

    grammars, glue_grammars, input_dfa = prepare_for_parsing(seg, args)

    # 3. Parse input
    input_grammar = make_hypergraph_from_input_view(grammars, glue_grammars, DummyConstant(semiring.inside.one))

    logging.debug('[%d] Parsing source', seg.id)
    #  3b. get a parser and intersect the source FSA
    goal_maker = GoalRuleMaker(args.goal, args.start)
    parser = NederhofParser(input_grammar, input_dfa, semiring.inside)
    root = input_grammar.fetch(Nonterminal(args.start))
    source_forest = parser.do(root, goal_maker.get_iview())
    pickle_it('{0}/{1}.src.forest'.format(staticdir, seg.id), source_forest)

    #  3c. compute target projection
    logging.debug('[%d] Computing target projection', seg.id)
    # Pass0
    target_forest = output_projection(source_forest, semiring.inside, TableLookupScorer(model.dummy))
    pickle_it('{0}/{1}.hyp.forest'.format(staticdir, seg.id), target_forest)

    # 4. Local scoring
    logging.debug('[%d] Computing local components', seg.id)
    #  4a. lookup components
    lookupffs = lookup_components(target_forest, model.lookup.extractors())
    #  4b. stateless components
    pickle_it('{0}/{1}.hyp.ffs.rule'.format(staticdir, seg.id), lookupffs)
    statelessffs = stateless_components(target_forest, model.stateless.extractors())
    pickle_it('{0}/{1}.hyp.ffs.stateless'.format(staticdir, seg.id), statelessffs)

    # 5. Parse references
    logging.debug('[%d] Preparing target FSA for %d references', seg.id, len(seg.refs))
    #  5a. make input FSA
    ref_dfa = make_dfa_set([ref.split() for ref in seg.refs], semiring.inside.one)
    #  5b. get a parser and intersect the target FSA
    logging.debug('[%d] Parsing references: states=%d arcs=%d', seg.id, ref_dfa.n_states(), ref_dfa.n_arcs())
    goal_maker.update()
    parser = EarleyParser(target_forest, ref_dfa, semiring.inside)
    ref_forest = parser.do(0, goal_maker.get_oview())

    if not ref_forest:
        logging.debug('[%d] The model cannot generate reference translations', seg.id)
        return False

    logging.debug('[%d] Rescoring reference forest: nodes=%d edges=%d', seg.id, ref_forest.n_nodes(), ref_forest.n_edges())
    goal_maker.update()

    # 6. Nonlocal scoring

    #  5c. complete model: thus, with stateful components
    rescorer = EarleyRescorer(ref_forest,
                              TableLookupScorer(model.lookup),
                              StatelessScorer(model.stateless),
                              StatefulScorer(model.stateful),
                              semiring.inside,
                              map_edges=False,
                              keep_frepr=True)
    ref_forest = rescorer.do(0, goal_maker.get_oview())
    pickle_it('{0}/{1}.ref.forest'.format(staticdir, seg.id), ref_forest)
    pickle_it('{0}/{1}.ref.ffs.all'.format(staticdir, seg.id), rescorer.components())

    raw_viterbi = viterbi_derivation(ref_forest, AcyclicTopSortTable(ref_forest))
    score = derivation_weight(ref_forest, raw_viterbi, semiring.inside)
    logging.debug('[%d] Viterbi derivation [%s]: %s', seg.id, score, yield_string(ref_forest, raw_viterbi))

    return True


def load_batch(args, path):
    """
    Load training batches
    :param args:
    :param path:
    :return:
    """
    return [SegmentMetaData.parse(input_str, grammar_dir=args.dev_grammars) for input_str in smart_ropen(path)]


def save_batch(args, workspace, batch):
    # 1. Log the segments in the batch
    os.makedirs(workspace, exist_ok=True)
    with smart_wopen('{0}/batch.gz'.format(workspace)) as fo:
        for seg in batch:
            print(seg.to_sgm(True), file=fo)


def prepare_batches(args, workspace, segments):
    parsable = list(segments)
    # how many batches we are preparing
    if args.mode == 'all':
        n_batches = 1
    elif args.mode == 'online':
        n_batches = len(parsable)
    else:
        n_batches = number_of_batches(len(parsable), float(args.mode)/100)
    logging.info('Spliting %d training instances into %d batches', len(parsable), n_batches)

    batches = []
    if all(os.path.exists('{0}/batch{1}/batch.gz'.format(workspace, b)) for b in range(n_batches)) and not args.redo:
        for b in range(n_batches):
            batches.append(load_batch(args, '{0}/batch{1}/batch.gz'.format(workspace, b)))
        return batches

    # make batches
    if args.shuffle:
        shuffle(parsable)
    if args.mode == 'all':
        batches = [parsable]  # a single batch with all data points
        logging.info('Single batch')
    elif args.mode == 'online':
        batches = [[seg] for seg in parsable]  # as many batches as training instances
        logging.info('Online: %d instances', len(batches))
    else:  # batch mode
        batches = make_batches(parsable, float(args.mode)/100)
        logging.info('%d batches: %s', len(batches), '/'.join(str(len(b)) for b in batches))

    assert len(batches) == n_batches, 'Mismatch in number of batches'

    # save batches
    for b, batch in enumerate(batches):
        batchdir = '{0}/batch{1}'.format(workspace, b)
        save_batch(args, batchdir, batch)

    return batches


def sample(args, staticdir, workspace, model, iteration, batch_number, batch):
    logging.info('[%d] Sampling from f(d|x) for x in batch %d', iteration, batch_number)

    workers = Pool(args.jobs)
    workers.map(partial(t_slice_sample,
                        args=args,
                        staticdir=staticdir,
                        workspace=workspace,
                        model=model),
                batch)


def t_slice_sample(seg, args, staticdir, workspace, model):
    try:
        return slice_sample(seg, args, staticdir, workspace, model)
    except:
        raise Exception('job={0} exception={1}'.format(seg.id,
                                                       ''.join(traceback.format_exception(*sys.exc_info()))))


def slice_sample(seg, args, staticdir, workspace, model):
    files = ['{0}/{1}.hyp-but-ref.ffs.all'.format(workspace, seg.id),
             '{0}/{1}.hyp.ffs.all'.format(workspace, seg.id)]

    if all(os.path.exists(path) for path in files) and not args.redo:
        logging.info('Reusing samples for segment %d', seg.id)
        return

    # 1. Load pickled objects
    logging.debug('[%d] Loading target forest', seg.id)
    forest = unpickle_it('{0}/{1}.hyp.forest'.format(staticdir, seg.id))
    logging.debug('[%d] Loading local components', seg.id)
    lookupffs = unpickle_it('{0}/{1}.hyp.ffs.rule'.format(staticdir, seg.id))
    statelessffs = unpickle_it('{0}/{1}.hyp.ffs.stateless'.format(staticdir, seg.id))

    # 2. Compute l(d)
    # there is a guarantee that lookup components and stateless components were computed over the same forest
    # that is, with the same nodes/edges structure
    # this is crucial to compute l(d) as below
    logging.debug('[%d] Computing l(d)', seg.id)
    lfunc = TableLookupFunction(np.array([semiring.inside.times(model.lookup.score(ff1),
                                                                model.stateless.score(ff2))
                                          for ff1, ff2 in zip(lookupffs, statelessffs)], dtype=ptypes.weight))

    # 3. Sample from f(d) = n(d) * l(d)
    logging.debug('[%d] Sampling from f(d) = n(d) * l(d)', seg.id)
    tsort = AcyclicTopSortTable(forest)
    goal_maker = GoalRuleMaker(args.goal, args.start, n=2)

    sampler = SlicedRescoring(forest,
                              lfunc,
                              tsort,
                              TableLookupScorer(model.dummy),
                              StatelessScorer(model.dummy),
                              StatefulScorer(model.stateful),
                              semiring.inside,
                              goal_maker.get_oview(),
                              OutputView(make_dead_srule()),
                              temperature0=args.temperature0)

    # here samples are represented as sequences of edge ids
    d0, markov_chain = sampler.sample(args)

    samples = apply_filters(markov_chain,
                            burn=args.burn,
                            lag=args.lag)

    projections = group_by_projection(samples, lambda d: yield_string(forest, d.edges))

    # 4. Complete feature vectors
    hypcomps = []
    #refcomps = []
    refset = frozenset(seg.refs)
    hypexp = model.constant(semiring.prob.zero)
    #refexp = model.constant(semiring.prob.zero)
    n_samples = 0
    complement = []
    for y_group in projections:
        y_comps = model.constant(semiring.inside.one)
        for d_group in group_by_identity(y_group.values):
            sample = d_group.key
            lookup_comps = model.lookup.constant(semiring.inside.one)
            stateless_comps = model.stateless.constant(semiring.inside.one)
            for e in sample.edges:
                lookup_comps = lookup_comps.hadamard(lookupffs[e], semiring.inside.times)
                stateless_comps = stateless_comps.hadamard(statelessffs[e], semiring.inside.times)
            # complete components (lookup, stateless, stateful)
            sample.components = FComponents([lookup_comps, stateless_comps, sample.components])
            # incorporate sample frequency
            n_samples += d_group.count
            hypcomps.append(sample.components.power(d_group.count, semiring.inside))
            hypexp = hypexp.hadamard(hypcomps[-1], semiring.prob.plus)
        if y_group.key not in refset:
            complement.append(sample.components)

        #if y_group.key in refset:
            #refcomps.append(hypcomps[-1])
            #refexp = refexp.hadamard(refcomps[-1], semiring.prob.plus)

    hypexp = hypexp.prod(1.0/n_samples)
    #refexp = refexp.prod(1.0/n_samples)

    #print('{0} ref-estimated {1}'.format(seg.id, refexp))
    #print('{0} hyp-estimated {1}'.format(seg.id, hypexp))

    pickle_it('{0}/{1}.hyp.ffs.all'.format(workspace, seg.id), hypcomps)
    pickle_it('{0}/{1}.hyp-but-ref.ffs.all'.format(workspace, seg.id), complement)
    #pickle_it('{0}/{1}.ss-ref.ffs.all'.format(workspace, seg.id), refcomps)

    # 5. Log stuff

    if args.save_d:
        derivations = group_by_identity(samples)
        save_mcmc_derivations('{0}/{1}.hyp.d.gz'.format(workspace, seg.id),
                              derivations,
                              valuefunc=lambda d: d.score,
                              #compfunc=lambda d: d.components,
                              derivation2str=lambda d: bracketed_string(forest, d.edges))

    if args.save_y:
        save_mcmc_yields('{0}/{1}.hyp.y.gz'.format(workspace, seg.id),
                         projections)

    if args.save_chain:
        markov_chain.appendleft(d0)
        save_markov_chain('{0}/{1}.hyp.chain.gz'.format(workspace, seg.id),
                          markov_chain,
                          flat=True,
                          valuefunc=lambda d: d.score,
                          #compfunc=lambda d: d.components,  # TODO: complete feature vectors of all derivations
                          derivation2str=lambda d: bracketed_string(forest, d.edges))


def t_hyp_expectations(seg, args, workspace, model, n_samples):
    try:
        return hyp_expectations(seg, args, workspace, model, n_samples)
    except:
        raise Exception('job={0} exception={1}'.format(seg.id,
                                                       ''.join(traceback.format_exception(*sys.exc_info()))))


def hyp_expectations(seg, args, workspace, model, n_samples):

    # 1. Load ffs
    components = unpickle_it('{0}/{1}.hyp.ffs.all'.format(workspace, seg.id))

    # 2. Re-estimate probabilities
    logging.debug('[%d] Computing f(d|x,y)', seg.id)

    # estimate p(d) by renormalising f(d) [which already incorporates sample frequency]
    fd = np.array([model.score(comp) for comp in components], dtype=ptypes.weight)
    pd = semiring.inside.normalise(fd)
    # estimate Z(x) by averaging f(d)
    Zx = semiring.inside.divide(semiring.inside.plus.reduce(fd),
                                semiring.inside.from_real(n_samples))
    # estimate <phi(d)> wrt f(d)
    expff = model.constant(semiring.prob.zero)
    # here we use the renormalised distribution
    for p, comp in zip(pd, components):
        expff = expff.hadamard(comp.power(semiring.inside.as_real(p), semiring.inside), semiring.prob.plus)

    # we may also estimate Z(x, not ref) by simply summing over (x, not ref) without taking sample frequency into account
    complement = unpickle_it('{0}/{1}.hyp-but-ref.ffs.all'.format(workspace, seg.id))
    Z_butref = semiring.inside.plus.reduce([model.score(comp) for comp in complement])

    return Zx, expff, Z_butref


def t_ref_expectations(seg, args, staticdir, model):
    try:
        return ref_expectations(seg, args, staticdir, model)
    except:
        raise Exception('job={0} exception={1}'.format(seg.id,
                                                       ''.join(traceback.format_exception(*sys.exc_info()))))


def ref_expectations(seg, args, staticdir, model, forest=None, components=None, tsort=None):

    # 1. Load pickled objects if necessary
    if forest is None:
        logging.debug('[%d] Loading pickled reference forest and components', seg.id)
        forest = unpickle_it('{0}/{1}.ref.forest'.format(staticdir, seg.id))
    if components is None:
        components = unpickle_it('{0}/{1}.ref.ffs.all'.format(staticdir, seg.id))
    if tsort is None:
        tsort = AcyclicTopSortTable(forest)

    # 2. Compute f(d|x, y)
    logging.debug('[%d] Computing f(d|x,y)', seg.id)
    weights = np.array([model.score(components[e]) for e in range(forest.n_edges())], dtype=ptypes.weight)
    fd = TableLookupFunction(weights)

    # 3. Compute expectations
    logging.debug('[%d] Computing expectations', seg.id)
    values = acyclic_value_recursion(forest, tsort, semiring.inside, omega=fd)
    reversed_values = acyclic_reversed_value_recursion(forest, tsort, semiring.inside, values, omega=fd)
    edge_expectations = compute_edge_expectation(forest, semiring.inside, values, reversed_values,
                                                 omega=fd, normalise=True)
    expff = model.constant(semiring.prob.zero)
    for e in range(forest.n_edges()):
        ue = semiring.inside.as_real(edge_expectations[e])
        expff = expff.hadamard(components[e].power(ue, semiring.inside),
                               semiring.prob.plus)
    #logging.info('[%d] Z(ref)=%s u(ref)=%s', seg.id, values[tsort.root()], expff)

    return values[tsort.root()], expff


def update_reference_stats(args, staticdir, workspace, model, batch):
    workers = Pool(args.jobs)
    results = workers.map(partial(t_ref_expectations,
                                  args=args,
                                  staticdir=staticdir,
                                  model=model),
                          batch)
    Z = semiring.inside.one
    ff = model.constant(semiring.inside.one)
    for z, u in results:
        ff = ff.hadamard(u, semiring.inside.times)
        Z = semiring.inside.times(Z, z)
    return Z, ff


def update_hypotheses_stats(args, staticdir, workspace, model, batch):
    workers = Pool(args.jobs)
    results = workers.map(partial(hyp_expectations,
                                  args=args,
                                  workspace=workspace,
                                  model=model,
                                  n_samples=args.samples),
                          batch)

    Z = semiring.inside.one
    ff = model.constant(semiring.inside.one)
    Z_butref = semiring.inside.one
    for z, u, z_butref in results:
        ff = ff.hadamard(u, semiring.inside.times)
        Z = semiring.inside.times(Z, z)
        Z_butref = semiring.inside.times(Z_butref, z_butref)

    return Z, ff, Z_butref


def objective_and_derivatives(args, staticdir, workspace, model, batch):

    # 1. Update expectations wrt p(d|x,y)
    Z_xy, ff_xy = update_reference_stats(args, staticdir, workspace, model, batch)

    # 2. Update expectation wrt to p(d,y|x)
    Z_x, ff_x, Z_x_not_y = update_hypotheses_stats(args, staticdir, workspace, model, batch)

    logging.info('Z_xy=%f Z_x=%f Z_x_not_y=%f', Z_xy, Z_x, Z_x_not_y
                 )
    # TODO: normalise by batch length?
    #if len(batch) > 1:
    #    refexp = refexp.prod(1.0/len(batch))  # TODO use hadamard and semiring.pow
    #    hypexp = hypexp.prod(1.0/len(batch))

    # TODO: turn semiring.divide into a binary operator
    #assert all(semiring.inside.gt(zh, zr) for zr, zh in zip(refprob, hypprob)), 'Ops'

    # 3. Compute likelihood and derivative
    likelihood = semiring.inside.divide(Z_xy, Z_x)
    # TODO: use Z_x_not_y?
    derivative = ff_xy.hadamard(ff_x.elementwise(semiring.inside.times.inverse), semiring.inside.times)

    return likelihood, derivative


def optimise(args, staticdir, workspace, model, iteration, batch_number, batch):
    logging.info('[%d] Optimising model on batch %d', iteration, batch_number)

    def f(theta):

        model_t = make_models(dict(zip(model.fnames(), theta)), model.extractors())

        logprob, derivatives = objective_and_derivatives(args, staticdir, workspace, model_t, batch)
        obj = -logprob
        jac = -np.array(list(derivatives.densify()), dtype=ptypes.weight)
        logging.info('O=%f', obj)
        logging.info('J=%s', npvec2str(jac, model.fnames()))
        if args.L2 == 0.0:
            logging.info('[%d:%d] L=%f',  iteration, batch_number, obj)
            return obj, jac
        else:
            r_obj = obj
            r_jac = jac.copy()

            if args.L2 != 0.0:  # L2-regularised
                regulariser = LA.norm(theta, 2) ** 2
                r_obj += args.L2 * regulariser
                r_jac += 2 * args.L2 * theta
                logging.info('[%d:%d] L=%f L2-regularised=%f', iteration, batch_number, obj, r_obj)

            return r_obj, r_jac

    def callback(theta):
        logging.info('[%d:%d] New theta: %s', iteration, batch_number, npvec2str(theta, fnames=model.fnames()))

    t0 = time()
    logging.info('[%d:%d] Optimising likelihood', iteration, batch_number)
    initial = np.array(list(model.weights().densify()), dtype=ptypes.weight)
    logging.info('[%d:%d] Initial: %s', iteration, batch_number, npvec2str(initial, fnames=model.fnames()))
    result = minimize(f,
                      initial,
                      # method='BFGS',
                      method='L-BFGS-B',
                      jac=True,
                      callback=callback,
                      options={'maxiter': args.sgd[0],
                               'ftol': args.tol[0],
                               'gtol': args.tol[1],
                               'maxfun': args.sgd[1],
                               'disp': False})
    dt = time() - t0
    logging.info('[%d:%d] Target SGD: function=%f nfev=%d nit=%d success=%s message="%s" minutes=%s',
                 iteration, batch_number,
                 result.fun, result.nfev, result.nit, result.success, result.message, dt / 60)
    logging.info('[%d:%d] Final: %s', iteration, batch_number, npvec2str(result.x, fnames=model.fnames()))
    return result.x


def core(args):

    workspace, staticdir, dynamicdir = make_dirs(args)

    # 1. Make model
    logging.info('Loading feature extractors')
    # Load feature extractors
    extractors = load_feature_extractors(args)
    logging.info('Making model')
    # Load the model
    if args.init is None:
        model = make_models(read_weights(args.weights, temperature=args.temperature),
                            extractors)
    if args.init == 'random':
        model = make_models(read_weights(args.weights, random=True),
                            extractors)
    elif args.init == 'uniform':
        model = make_models(read_weights(args.weights),
                            extractors,
                            uniform_weights=True)
    else:
        model = make_models(read_weights(args.weights, default=float(args.init)),
                            extractors)

    fnames = model.fnames()
    logging.debug('Model\n%s', model)
    print('{0} ||| init ||| {1}'.format(0, npvec2str(model.weights().densify(), fnames)))

    # 2. Parse data
    segments = [SegmentMetaData.parse(input_str, grammar_dir=args.dev_grammars) for input_str in smart_ropen(args.dev)]
    logging.info('Parsing %d training instances using %d workers', len(segments), args.jobs)
    workers = Pool(args.jobs)
    feedback = workers.map(partial(t_biparse,
                                   args=args,
                                   staticdir=staticdir,
                                   model=model),
                           segments)
    parsable = [seg for seg, status in zip(segments, feedback) if status]
    logging.info(' %d out of %d training instances are bi-parsable', len(parsable), len(segments))

    # 3. Optimise
    dimensionality = len(fnames)
    for iteration in range(1, args.maxiter + 1):
        iterdir = '{0}/{1}'.format(dynamicdir, iteration)
        os.makedirs(iterdir, exist_ok=True)
        batches = prepare_batches(args, iterdir, parsable)

        # 3b. process each batch in turn
        avg = np.zeros(dimensionality, dtype=ptypes.weight)
        for b, batch in enumerate(batches):
            batchdir = '{0}/batch{1}'.format(iterdir, b)

            # i. sample
            sample(args, staticdir, batchdir, model, iteration, b, batch)

            # ii. optimise
            weights = optimise(args, staticdir, batchdir, model, iteration, b, batch)

            print('{0} ||| {1} ||| {2}'.format(iteration, b, npvec2str(avg, fnames)))
            avg += weights
        avg /= len(batches)
        print('{0} ||| mean ||| {1}'.format(iteration, npvec2str(avg, fnames)))


def main():
    args = get_argparser().parse_args()

    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    core(args)


if __name__ == '__main__':
    main()