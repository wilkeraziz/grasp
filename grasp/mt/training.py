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

import grasp.ptypes as ptypes

from grasp.recipes import smart_ropen, smart_wopen, make_unique_directory, pickle_it, unpickle_it, traceit

from grasp.scoring.scorer import TableLookupScorer, StatelessScorer, StatefulScorer
from grasp.scoring.model import Model, make_models
from grasp.scoring.util import read_weights

from grasp.mt.cdec_format import load_grammar
from grasp.mt.util import load_feature_extractors, GoalRuleMaker
from grasp.mt.util import save_forest, save_ffs, load_ffs, make_dead_srule, make_batches, number_of_batches
from grasp.mt.segment import SegmentMetaData
from grasp.mt.input import make_input

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
    parser.add_argument('--init', type=str, default=None,
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


def cmd_external(parser):
    parser.add_argument('--scoring-tool', type=str,
                        default='/Users/waziz/workspace/cdec/mteval/fast_score',
                        help='a scoring tool such as fast_score')


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
                       help='include rule table features (both indicators and log-transformed probabilites)')
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
    group.add_argument('--samples', type=int, nargs=2, default=[100, 100],
                       metavar='N N',
                       help='number of samples for training (estimation of expectations) and testing (consensus decoding)')
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
    cmd_optimisation(parser.add_argument_group('Parameter optimisation by SGD'))
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

    devdir = '{0}/{1}'.format(outdir, args.dev_alias)
    os.makedirs(devdir, exist_ok=exist_ok)

    if args.devtest:
        devtestdir = '{0}/{1}'.format(outdir, args.devtest_alias)
        os.makedirs(devtestdir, exist_ok=exist_ok)

    dynamicdir = '{0}/iterations'.format(outdir)
    os.makedirs(dynamicdir, exist_ok=exist_ok)

    return outdir, devdir


def prepare_for_parsing(seg, args):
    """
    Load grammars (i.e. main, extra, glue, passthrough) and prepare input FSA.
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


def get_target_forest(seg, args, model):
    """
    Load grammars with `prepare_for_parsing`,
        parse and produce target forest.

    :return: forest, GoalRuleMaker
    """
    grammars, glue_grammars, input_dfa = prepare_for_parsing(seg, args)

    # 3. Parse input
    input_grammar = make_hypergraph_from_input_view(grammars, glue_grammars, DummyConstant(semiring.inside.one))

    #  3b. get a parser and intersect the source FSA
    goal_maker = GoalRuleMaker(args.goal, args.start)
    parser = NederhofParser(input_grammar, input_dfa, semiring.inside)
    root = input_grammar.fetch(Nonterminal(args.start))
    source_forest = parser.do(root, goal_maker.get_iview())

    #  3c. compute target projection
    target_forest = output_projection(source_forest, semiring.inside, TableLookupScorer(model.dummy))
    return target_forest, goal_maker


def get_localffs(forest, model):
    """
    Return a list of lookup components and stateless components.
    """
    # 4. Local scoring
    #  4a. lookup components
    lookupffs = lookup_components(forest, model.lookup.extractors())
    #  4b. stateless components
    statelessffs = stateless_components(forest, model.stateless.extractors())
    return lookupffs, statelessffs


@traceit
def decode(seg, args, model, workspace=None, ranking_path=None):
    """
    Sample and rank solutions by consensus BLEU
    """
    files = ['{0}/{1}.hyp.forest'.format(workspace, seg.id),
             '{0}/{1}.hyp.ffs.rule'.format(workspace, seg.id),
             '{0}/{1}.hyp.ffs.stateless'.format(workspace, seg.id)]
    if args.redo or workspace is None or not all(os.path.exists(path) for path in files):
        logging.info('[%d] Parsing and local scoring', seg.id)
        # parse
        forest, goal_maker = get_target_forest(seg, args, model)
        goal_maker.update()
        # local ffs
        lookupffs, statelessffs = get_localffs(forest, model)
        if workspace is not None:  # save objects
            pickle_it('{0}/{1}.hyp.forest'.format(workspace, seg.id), forest)
            pickle_it('{0}/{1}.hyp.ffs.rule'.format(workspace, seg.id), lookupffs)
            pickle_it('{0}/{1}.hyp.ffs.stateless'.format(workspace, seg.id), statelessffs)
    else:  # load
        logging.info('[%d] Reusing forest and local components', seg.id)
        forest = unpickle_it('{0}/{1}.hyp.forest'.format(workspace, seg.id))
        goal_maker = GoalRuleMaker(args.goal, args.start, n=2)
        lookupffs = unpickle_it('{0}/{1}.hyp.ffs.rule'.format(workspace, seg.id))
        statelessffs = unpickle_it('{0}/{1}.hyp.ffs.stateless'.format(workspace, seg.id))

    logging.info('[%d] Slice sampling', seg.id)
    # l(d)
    lfunc = TableLookupFunction(np.array([semiring.inside.times(model.lookup.score(ff1),
                                                                model.stateless.score(ff2))
                                          for ff1, ff2 in zip(lookupffs, statelessffs)], dtype=ptypes.weight))
    # slice sample
    tsort = AcyclicTopSortTable(forest)

    sampler = SlicedRescoring(forest,
                              lfunc,
                              tsort,
                              TableLookupScorer(model.dummy),
                              StatelessScorer(model.dummy),
                              StatefulScorer(model.stateful),
                              semiring.inside,
                              goal_maker.get_oview(),
                              OutputView(make_dead_srule()))

    # here samples are represented as sequences of edge ids
    d0, markov_chain = sampler.sample(n_samples=args.samples[1], batch_size=args.batch, within=args.within,
                                      initial=args.initial, prior=args.prior, burn=args.burn, lag=args.lag,
                                      temperature0=args.temperature0)

    samples = apply_filters(markov_chain,
                            burn=args.burn,
                            lag=args.lag)
    # total number of samples kept
    N = len(samples)
    projections = group_by_projection(samples, lambda d: yield_string(forest, d.edges))

    logging.info('[%d] Consensus decoding', seg.id)
    # translation strings
    support = [group.key for group in projections]
    # empirical distribution
    posterior = np.array([float(group.count)/N for group in projections], dtype=ptypes.weight)
    # consensus decoding
    scorer = DecodingBLEU(support, posterior)
    losses = np.array([scorer.loss(y) for y in support], dtype=ptypes.weight)
    # order samples by least loss, then by max prob
    ranking = sorted(range(len(support)), key=lambda i: (losses[i], -posterior[i]))
    if ranking_path is not None:
        # write all decisions to file
        with smart_wopen('{0}/{1}.gz'.format(ranking_path, seg.id)) as fo:
            print('# co-loss ||| posterior ||| solution', file=fo)
            for i in ranking:
                print('{0} ||| {1} ||| {2}'.format(losses[i], posterior[i], support[i]), file=fo)
    best = ranking[0]
    return support[best]


def mteval(args, staticdir, model, segments, hyp_path, ref_path, eval_path, ranking_path):
    """
    Decode and evaluate with an external tool.
    :return: BLEU score.
    """

    if ranking_path:
        os.makedirs(ranking_path, exist_ok=True)

    # decode
    with Pool(args.jobs) as workers:
        results = workers.map(partial(decode,
                                      args=args,
                                      model=model,
                                      workspace=staticdir,
                                      ranking_path=ranking_path),
                              segments)

    # write best decisions to file
    with smart_wopen(hyp_path) as fo:
        for y in results:
            print(y, file=fo)

    # call scoring tool
    cmd_str = '{0} -r {1}'.format(args.scoring_tool, ref_path)
    logging.info('Scoring: %s', cmd_str)
    # prepare args
    cmd_args = shlex.split(cmd_str)
    # assess
    score = None
    with smart_ropen(hyp_path) as fin:
        with smart_wopen('{0}.stdout'.format(eval_path)) as fout:
            with smart_wopen('{0}.stderr'.format(eval_path)) as ferr:
                with sp.Popen(cmd_args, stdin=fin, stdout=fout, stderr=ferr) as proc:
                    proc.wait()
    try:
        with smart_ropen('{0}.stdout'.format(eval_path)) as fi:
            line = next(fi)
            score = float(line.strip())
    except:
        logging.error('Problem reading %s.stdout', eval_path)

    return score


@traceit
def biparse(seg, args, staticdir, model):
    # this procedure produces the following files
    general_files = ['{0}/{1}.hyp.forest'.format(staticdir, seg.id),
                     '{0}/{1}.hyp.ffs.rule'.format(staticdir, seg.id),
                     '{0}/{1}.hyp.ffs.stateless'.format(staticdir, seg.id)]
    ref_files = ['{0}/{1}.ref.ffs.all'.format(staticdir, seg.id),
                 '{0}/{1}.ref.forest'.format(staticdir, seg.id)]
    # check for redundant work
    if all(os.path.exists(path) for path in general_files) and not args.redo:
        if all(os.path.exists(path) for path in ref_files):
            logging.info('Reusing forests for segment %d', seg.id)
            return True   # parsable
        else:
            return False  # not parsable

    target_forest, goal_maker = get_target_forest(seg, args, model)
    pickle_it('{0}/{1}.hyp.forest'.format(staticdir, seg.id), target_forest)

    # 4. Local scoring
    logging.debug('[%d] Computing local components', seg.id)
    lookupffs, statelessffs = get_localffs(target_forest, model)
    pickle_it('{0}/{1}.hyp.ffs.rule'.format(staticdir, seg.id), lookupffs)
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

    tsort = AcyclicTopSortTable(ref_forest)
    n_derivations = int(semiring.inside.as_real(acyclic_value_recursion(ref_forest, tsort, semiring.inside)[tsort.root()]))
    logging.info('[%d] %d derivations in refset', seg.id, n_derivations)

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

    tsort = AcyclicTopSortTable(ref_forest)
    raw_viterbi = viterbi_derivation(ref_forest, tsort)
    score = derivation_weight(ref_forest, raw_viterbi, semiring.inside)
    logging.info('[%d] Viterbi derivation in refset [%s]: %s', seg.id, score, bracketed_string(ref_forest, raw_viterbi))

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


def get_empirical_support(model, refset, forest, lookupffs, statelessffs, markov_chain):
    # complete feature vectors (with local components) and estimate an empirical support
    support = []
    seen = set()
    for derivation in markov_chain:
        if derivation.edges in seen:
            continue
        seen.add(derivation.edges)  # no duplicates
        # reconstruct components
        lookup_comps = model.lookup.constant(semiring.inside.one)
        stateless_comps = model.stateless.constant(semiring.inside.one)
        for e in derivation.edges:
            lookup_comps = lookup_comps.hadamard(lookupffs[e], semiring.inside.times)
            stateless_comps = stateless_comps.hadamard(statelessffs[e], semiring.inside.times)
        # complete components (lookup, stateless, stateful)
        components = FComponents([lookup_comps, stateless_comps, derivation.components])
        # get yield
        y = yield_string(forest, derivation.edges)
        # update estimated support
        support.append((y in refset, derivation.edges, components))
    return support


@traceit
def slice_sample(seg, args, staticdir, supportdir, workspace, model):
    files = ['{0}/{1}.D.ffs.all'.format(supportdir, seg.id),
             '{0}/{1}.hyp.ffs.all'.format(workspace, seg.id)]

    if all(os.path.exists(path) for path in files) and not args.redo:
        logging.info('Reusing samples for segment %d', seg.id)
        return

    # 1. Load pickled objects
    logging.debug('[%d] Loading target forest', seg.id)
    forest = unpickle_it('{0}/{1}.hyp.forest'.format(staticdir, seg.id))
    # TODO: store top sort table
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
                              OutputView(make_dead_srule()))

    # here samples are represented as sequences of edge ids
    d0, markov_chain = sampler.sample(n_samples=args.samples[0], batch_size=args.batch, within=args.within,
                                      initial=args.initial, prior=args.prior, burn=args.burn, lag=args.lag,
                                      temperature0=args.temperature0)

    # save empirical support
    pickle_it('{0}/{1}.D.ffs.all'.format(supportdir, seg.id),
              get_empirical_support(model, frozenset(seg.refs), forest, lookupffs, statelessffs, markov_chain))


    # apply usual MCMC filters to the Markov chain
    samples = apply_filters(markov_chain,
                            burn=args.burn,
                            lag=args.lag)

    n_samples = len(samples)

    # 4. Complete feature vectors and compute expectation
    hypcomps = []
    hypexp = model.constant(semiring.prob.zero)
    d_groups = group_by_identity(samples)
    for d_group in d_groups:
        derivation = d_group.key
        # reconstruct components
        lookup_comps = model.lookup.constant(semiring.inside.one)
        stateless_comps = model.stateless.constant(semiring.inside.one)
        for e in derivation.edges:
            lookup_comps = lookup_comps.hadamard(lookupffs[e], semiring.inside.times)
            stateless_comps = stateless_comps.hadamard(statelessffs[e], semiring.inside.times)
        # complete components (lookup, stateless, stateful)
        # note that here we are updating derivation.components!
        derivation.components = FComponents([lookup_comps, stateless_comps, derivation.components])
        # incorporate sample frequency
        hypcomps.append(derivation.components.power(float(d_group.count)/n_samples, semiring.inside))
        hypexp = hypexp.hadamard(hypcomps[-1], semiring.prob.plus)

    # save feature vectors
    pickle_it('{0}/{1}.hyp.ffs.all'.format(workspace, seg.id), hypcomps)

    # 5. Log stuff
    if args.save_d:
        save_mcmc_derivations('{0}/{1}.hyp.d.gz'.format(workspace, seg.id),
                              d_groups,
                              valuefunc=lambda d: d.score,
                              compfunc=lambda d: d.components,
                              derivation2str=lambda d: bracketed_string(forest, d.edges))

    if args.save_y:
        projections = group_by_projection(samples, lambda d: yield_string(forest, d.edges))
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


def sample(args, staticdir, supportdir, workspace, model, iteration, batch_number, batch):
    logging.info('[%d] Slice sampling batch %d', iteration, batch_number)

    with Pool(args.jobs) as workers:
        workers.map(partial(slice_sample,
                            args=args,
                            staticdir=staticdir,
                            supportdir=supportdir,
                            workspace=workspace,
                            model=model),
                    batch)


@traceit
def ref_expectations(seg, args, staticdir, model):
    """
    Return Z(x, y \in ref) and the expected feature vector.
    """

    # 1. Load pickled objects if necessary

    logging.debug('[%d] Loading pickled reference forest and components', seg.id)
    forest = unpickle_it('{0}/{1}.ref.forest'.format(staticdir, seg.id))
    components = unpickle_it('{0}/{1}.ref.ffs.all'.format(staticdir, seg.id))
    tsort = AcyclicTopSortTable(forest)

    # 2. Compute f(d|x, y)
    logging.debug('[%d] Computing f(d|x,y)', seg.id)
    weights = np.array([model.score(components[e]) for e in range(forest.n_edges())], dtype=ptypes.weight)
    fe = TableLookupFunction(weights)

    # 3. Compute expectations
    logging.debug('[%d] Computing expectations', seg.id)
    Z, mean = expected_components(forest, fe, tsort, semiring.inside, model, components)

    return Z, mean


@traceit
def hyp_expectations(seg, args, workspace, model):
    """
    Return Z(x) and the expected feature vector.
    """

    # 1. Load ffs
    components = unpickle_it('{0}/{1}.hyp.ffs.all'.format(workspace, seg.id))

    # 2. Re-estimate probabilities
    logging.debug('[%d] Computing f(d|x,y)', seg.id)

    # estimate p(d) by renormalising f(d) [which already incorporates sample frequency]
    fd = np.array([model.score(comp) for comp in components], dtype=ptypes.weight)
    pd = semiring.inside.normalise(fd)
    # estimate Z(x) = \sum_d f(d)
    Z = semiring.inside.plus.reduce(fd)
    # estimate <phi(d)> wrt f(d)
    mean = model.constant(semiring.prob.zero)
    # here we use the renormalised distribution to compute expected features
    for p, comp in zip(pd, components):
        mean = mean.hadamard(comp.elementwise(FixedLHS(semiring.inside.as_real(p),
                                                       semiring.prob.times)),
                             semiring.prob.plus)

    return Z, mean


@traceit
def estimate_partition_function(seg, model, merging):
    """
    Returns Z(x, y \in refset) and Z(x, y \not\in refset).
    """

    # 1. Load unique derivations separating them depending on whether or not they belong to the reference set

    components_R = []  # references
    components_C = []  # complement (not reference)

    seen = set()
    for supportdir in merging:
        for is_ref, edges, components in unpickle_it('{0}/{1}.D.ffs.all'.format(supportdir, seg.id)):
            if edges in seen:  # no duplicates
                continue
            seen.add(edges)
            if is_ref:
                components_R.append(components)
            else:
                components_C.append(components)
    logging.info('[%d] D(x) |R|=%d |C|=%d', seg.id, len(components_R), len(components_C))

    # 2. Re-estimate probabilities
    logging.debug('[%d] Computing f(d|x) to estimate partition function', seg.id)

    if len(components_R):
        Z_R = semiring.inside.plus.reduce(np.array([model.score(comp) for comp in components_R], dtype=ptypes.weight))
    else:
        Z_R = semiring.inside.zero
    if len(components_C):
        Z_C = semiring.inside.plus.reduce(np.array([model.score(comp) for comp in components_C], dtype=ptypes.weight))
    else:
        Z_C = semiring.inside.zero

    return Z_R, Z_C


def update_reference_stats(args, staticdir, workspace, model, batch):
    """
    Return an array containing Z(x, y \in refset) for every training instance,
        as well as the expected feature vector.
    """
    with Pool(args.jobs) as workers:
        results = workers.map(partial(ref_expectations,
                                      args=args,
                                      staticdir=staticdir,
                                      model=model),
                              batch)
    # results contain (partition function, expected features) for each training instance
    # the total in the training set is obtained by reduction using inside.times
    ff = model.constant(semiring.inside.one)
    for z, u in results:
        ff = ff.hadamard(u, semiring.inside.times)

    return np.array([z for z, u in results], dtype=ptypes.weight), ff


def update_hypotheses_stats(args, staticdir, workspace, model, batch):
    """
    Return an array containing Z(x)
        as well as the expected feature vector.
    """
    with Pool(args.jobs) as workers:
        results = workers.map(partial(hyp_expectations,
                                      args=args,
                                      workspace=workspace,
                                      model=model),
                              batch)

    # results contain (partition function, expected features) for each training instance
    # the total in the training set is obtained by reduction using inside.times
    ff = model.constant(semiring.inside.one)
    for z, u in results:
        ff = ff.hadamard(u, semiring.inside.times)

    return np.array([z for z, u in results], dtype=ptypes.weight), ff


def update_support_stats(args, staticdir, workspace, model, batch, merging):
    """
    Return two arrays representing the batch. The first array contains Z(x, y \in refset),
        the second array contains Z(x, y \not\in refset).
    """
    with Pool(args.jobs) as workers:
        results = workers.map(partial(estimate_partition_function,
                                      model=model,
                                      merging=merging),
                              batch)

    Z_R = np.array([z_r for z_r, z_c in results], dtype=ptypes.weight)
    Z_C = np.array([z_c for z_r, z_c in results], dtype=ptypes.weight)

    return Z_R, Z_C


def objective_and_derivatives(args, staticdir, workspace, model, batch, merging):

    factor = 1.0 / len(batch)

    # 1. Update Z(x, y \in refset) and Z(x, y \not\in refset)
    logging.info('Updating D(x)')
    Z_R, Z_C = update_support_stats(args, staticdir, workspace, model, batch, merging)

    # 1. Update expectations wrt p(d|x,y)
    # here we keep Z_xy because these are exact quantities
    logging.info('Updating f(d|x,y)')
    Z_xy, ff_xy = update_reference_stats(args, staticdir, workspace, model, batch)

    # Z_R <= Z_xy
    # typically, Z_R < Z_xy because the slice sampler that estimates Z_R might miss all the references

    # 2. Update expectation wrt to p(d,y|x)
    logging.info('Updating f(d,y|x)')
    Z_x, ff_x = update_hypotheses_stats(args, staticdir, workspace, model, batch)

    # 3. Estimate the likelihood \prod_i  Z(x_i, y_i) / Z(x_i)
    num = semiring.inside.times.reduce(Z_xy)
    # to approximate the denominator we sum Z_xy (from reference forests) and
    # Z_C (the complement, estimate by merging supports from different iterations)
    for i, (a, b, c) in enumerate(zip(Z_xy, Z_C, Z_x)):
        logging.info('(%d) Zxy=%f Zc=%f Zyx+Zc=%f Zx=%f', i, a, b, semiring.inside.plus(a, b), c)
    den = semiring.inside.times.reduce([semiring.inside.plus(a, b) for a, b in zip(Z_xy, Z_C)])
    likelihood = semiring.inside.divide(num, den)

    logging.info('Zxy=%f Zxy+Zc=%f | Zc=%f | discarded: Zr=%f Zx=%f',
                 num,
                 den,
                 semiring.inside.times.reduce(Z_C),
                 semiring.inside.times.reduce(Z_R),
                 semiring.inside.times.reduce(Z_x))

    # 4. Estimate the derivative <phi(d)>_p(d|x,y) - <phi(d)>_p(d|x)
    derivative = ff_xy.hadamard(ff_x.elementwise(semiring.inside.times.inverse), semiring.inside.times)

    # normalise by batch size
    if len(batch) > 1:
        factor = 1.0 / len(batch)
        likelihood *= factor
        derivative = derivative.prod(factor)

    return likelihood, derivative


def optimise(args, staticdir, workspace, model, iteration, batch_number, batch, merging):
    logging.info('[I=%d] Optimising model on batch %d', iteration, batch_number)

    def f(theta):

        model_t = make_models(dict(zip(model.fnames(), theta)), model.extractors())

        logprob, derivatives = objective_and_derivatives(args, staticdir, workspace, model_t, batch, merging)
        obj = -logprob
        jac = -np.array(list(derivatives.densify()), dtype=ptypes.weight)

        if args.L2 == 0.0:
            logging.info('[I=%d, b=%d] O=%f',  iteration, batch_number, obj)
            logging.debug('[I=%d, b=%d] Derivatives\n%s', iteration, batch_number, npvec2str(jac, model.fnames(), '\n'))
            return obj, jac
        else:
            r_obj = obj
            r_jac = jac.copy()

            if args.L2 != 0.0:  # L2-regularised
                regulariser = LA.norm(theta, 2) ** 2
                r_obj += args.L2 * regulariser
                r_jac += 2 * args.L2 * theta
                logging.info('[I=%d, b=%d] O=%f L=%f', iteration, batch_number, r_obj, obj)
                logging.debug('[I=%d, b=%d] Derivatives\n%s', iteration, batch_number, npvec2str(r_jac, model.fnames()))

            return r_obj, r_jac

    def callback(theta):
        logging.info('[I=%d, b=%d] New theta\n%s', iteration, batch_number, npvec2str(theta, model.fnames()))

    t0 = time()
    logging.info('[I=%d, b=%d] Optimising likelihood', iteration, batch_number)
    initial = np.array(list(model.weights().densify()), dtype=ptypes.weight)
    logging.info('[I=%d, b=%d] Initial: %s', iteration, batch_number, npvec2str(initial, model.fnames()))
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
    logging.info('[I=%d, b=%d] Target SGD: function=%f nfev=%d nit=%d success=%s message="%s" minutes=%s',
                 iteration, batch_number,
                 result.fun, result.nfev, result.nit, result.success, result.message, dt / 60)
    logging.info('[I=%d, b=%d] Final\n%s', iteration, batch_number, npvec2str(result.x, model.fnames()))
    return result.x


def eval_devtest(args, workspace, iterdir, model, segments):
    logging.info('Assessing loss on validation set')
    evaldir = '{0}/mteval'.format(iterdir)
    os.makedirs(evaldir, exist_ok=True)
    staticdir = '{0}/{1}'.format(workspace, args.devtest_alias)
    bleu = mteval(args, staticdir, model, segments,
                  '{0}/{1}.hyps'.format(evaldir, args.devtest_alias),
                  '{0}/refs'.format(staticdir, args.devtest_alias),
                  '{0}/{1}.bleu'.format(evaldir, args.devtest_alias),
                  '{0}/{1}.ranking'.format(evaldir, args.devtest_alias))
    logging.info('BLEU %s %s', args.devtest_alias, bleu)
    return bleu


def read_segments(path, grammar_dir):
    return tuple(SegmentMetaData.parse(input_str, grammar_dir=grammar_dir)
                 for input_str in smart_ropen(path))


def parse_training(args, staticdir, model, segments):
    logging.info('Parsing %d training instances using %d workers', len(segments), args.jobs)
    with Pool(args.jobs) as workers:
        feedback = workers.map(partial(biparse,
                                       args=args,
                                       staticdir=staticdir,
                                       model=model),
                               segments)
    return tuple([seg for seg, status in zip(segments, feedback) if status])


def core(args):

    workspace, devdir = make_dirs(args)

    # 1. Make model
    logging.info('Loading feature extractors')
    # Load feature extractors
    extractors = load_feature_extractors(args)
    logging.info('Making model')
    # Load the model
    if args.init is None:
        model = make_models(read_weights(args.weights, temperature=args.temperature),
                            extractors)
    elif args.init == 'random':
        model = make_models(read_weights(args.weights, random=True, temperature=args.temperature),
                            extractors)
    elif args.init == 'uniform':
        model = make_models(read_weights(args.weights, temperature=args.temperature),
                            extractors,
                            uniform_weights=True)
    else:
        model = make_models(read_weights(args.weights, default=float(args.init), temperature=args.temperature),
                            extractors)

    fnames = model.fnames()
    logging.debug('Model\n%s', model)

    # 2. Parse data
    segments = read_segments(args.dev, args.dev_grammars)
    parsable = parse_training(args, devdir, model, segments)
    logging.info(' %d out of %d training instances are bi-parsable', len(parsable), len(segments))

    # Validation set
    devtest = None
    if args.devtest:
        devtest = read_segments(args.devtest, args.devtest_grammars)
        # store references for evaluation purposes
        with smart_wopen('{0}/{1}/refs'.format(workspace, args.devtest_alias)) as fo:
            for seg in devtest:
                print(' ||| '.join(seg.refs), file=fo)
        # evaluate the initial model
        bleu = eval_devtest(args, workspace, '{0}/iterations/0'.format(workspace), model, devtest)
        print('{0} ||| init ||| {1}={2} ||| {3}'.format(0, args.devtest_alias, bleu, npvec2str(model.weights().densify(), fnames)))
    else:
        print('{0} ||| init |||  ||| {3}'.format(0, args.devtest_alias, npvec2str(model.weights().densify(), fnames)))

    # 3. Optimise
    dimensionality = len(fnames)
    merging = deque()
    for iteration in range(1, args.maxiter + 1):
        # where we store everything related to this iteration
        iterdir = '{0}/iterations/{1}'.format(workspace, iteration)
        os.makedirs(iterdir, exist_ok=True)

        # prepare batches
        batches = prepare_batches(args, iterdir, parsable)

        # we need to manage a view of the support of each training instance
        supportdir = '{0}/support'.format(iterdir)
        os.makedirs(supportdir, exist_ok=True)
        merging.append(supportdir)
        # we might not be interested in merge a complete history
        if len(merging) > args.merge > 0:
            merging.popleft()

        # 3b. process each batch in turn
        avg = np.zeros(dimensionality, dtype=ptypes.weight)
        for b, batch in enumerate(batches):
            batchdir = '{0}/batch{1}'.format(iterdir, b)

            # i. sample
            sample(args, devdir, supportdir, batchdir, model, iteration, b, batch)

            # ii. optimise
            weights = optimise(args, devdir, batchdir, model, iteration, b, batch, merging)

            print('{0} ||| batch{1} |||  ||| {2}'.format(iteration, b, npvec2str(weights, fnames)))
            avg += weights

        avg /= len(batches)
        model = make_models(dict(zip(model.fnames(), avg)), model.extractors())

        # check loss on validation set
        if devtest:
            bleu = eval_devtest(args, workspace, iterdir, model, devtest)
            print('{0} ||| avg ||| {1}={2} ||| {3}'.format(iteration,
                                                           args.devtest_alias,
                                                           bleu,
                                                           npvec2str(avg, fnames)))
        else:
            print('{0} ||| avg |||  ||| {3}'.format(iteration,
                                                    args.devtest_alias,
                                                    bleu,
                                                    npvec2str(avg, fnames)))


def main():
    args = get_argparser().parse_args()

    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    core(args)


if __name__ == '__main__':
    main()
