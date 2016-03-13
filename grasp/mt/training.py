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

import grasp.ptypes as ptypes

from grasp.recipes import smart_ropen, smart_wopen, make_unique_directory, pickle_it, unpickle_it

from grasp.scoring.scorer import TableLookupScorer, StatelessScorer, StatefulScorer
from grasp.scoring.model import Model, make_models
from grasp.scoring.util import read_weights

from grasp.mt.cdec_format import load_grammar
from grasp.mt.util import load_feature_extractors, GoalRuleMaker
from grasp.mt.util import save_forest, save_ffs, load_ffs, make_dead_srule
from grasp.mt.segment import SegmentMetaData
from grasp.mt.input import make_input


import grasp.semiring as semiring

from grasp.formal.scfgop import output_projection
from grasp.formal.fsa import make_dfa, make_dfa_set, make_dfa_set2
from grasp.formal.scfgop import make_hypergraph_from_input_view, output_projection
from grasp.formal.scfgop import lookup_components, stateless_components
from grasp.formal.topsort import AcyclicTopSortTable
from grasp.formal.traversal import bracketed_string, yield_string

from grasp.cfg.model import DummyConstant
from grasp.cfg.symbol import Nonterminal
from grasp.cfg.symbol import Terminal
from grasp.cfg.srule import OutputView

from grasp.alg.deduction import NederhofParser, EarleyParser, EarleyRescorer
from grasp.alg.inference import viterbi_derivation, AncestralSampler
from grasp.alg.value import LookupFunction, ConstantFunction
from grasp.alg.value import acyclic_value_recursion, acyclic_reversed_value_recursion
from grasp.alg.value import derivation_value, compute_edge_expectation
from grasp.alg.rescoring import weight_edges
from grasp.alg.rescoring import SlicedRescoring

from grasp.scoring.frepr import FComponents



def cmd_optimisation(parser):
    # Optimisation
    parser.add_argument("--maxiter", '-M', type=int, default=10,
                        help="Maximum number of iterations")
    parser.add_argument('--mode', type=str, default='10%',
                        help="use 'all' for all data, use 'online' for online updates, use 'S' to specify batch size in percentage")
    parser.add_argument('--shuffle',
                        action='store_true',
                        help='shuffle training instances')
    parser.add_argument('--samples', type=int, default=1000,
            help='Sampling schedule: number of samples')
    parser.add_argument('--default', type=float, default=None,
                        help='initialise all weights with a default value, if not given, we start from the values already specified in the config file')
    parser.add_argument("--resume", type=int, default=0,
                        help="Resume from a certain iteration (requires the config file of the preceding run)")

def cmd_external(parser):
    parser.add_argument('--scoring-tool', type=str,
            default='/Users/waziz/workspace/github/cdec/mteval/fast_score',
            help='a scoring tool such as fast_score')

def cmd_sgd(parser):
    parser.add_argument("--sgd", type=int, nargs=2, default=[10, 20],
                        help="Number of iterations and function evaluations for target optimisation")
    parser.add_argument("--tol", type=float, nargs=2, default=[1e-4, 1e-4],
                        help="f-tol and g-tol in target optimisation")
    parser.add_argument("--L2", type=float, default=0.0,
                        help="Weight of L2 regulariser in target optimisation")
    parser.add_argument("--T", type=float, default=0.0,
            help="Temperature parameter for target's entropic prior")


def cmd_logging(parser):
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
    cmd_model(parser.add_argument_group('Model'))
    cmd_parser(parser.add_argument_group('Parser'))
    cmd_grammar(parser.add_argument_group('Grammar'))
    cmd_optimisation(parser.add_argument_group('Parameter optimisation by coordinate descent'))
    cmd_sgd(parser.add_argument_group('Optimisation by SGD'))
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


    os.makedirs('{0}/forest'.format(outdir), exist_ok=exist_ok)

    return outdir


def update(data):
    pass


def prepare_for_parsing(seg, args, workspace):
    logging.info('[%d] Loading grammars', seg.id)
    # 1. Load grammars
    #  1a. additional grammars
    extra_grammars = []
    if args.extra_grammar:
        for grammar_path in args.extra_grammar:
            logging.info('[%d] Loading additional grammar: %s', seg.id, grammar_path)
            grammar = load_grammar(grammar_path)
            logging.info('[%d] Additional grammar: productions=%d', seg.id, len(grammar))
            extra_grammars.append(grammar)
    #  1b. glue grammars
    glue_grammars = []
    if args.glue_grammar:
        for glue_path in args.glue_grammar:
            logging.info('[%d] Loading glue grammar: %s', seg.id, glue_path)
            glue = load_grammar(glue_path)
            logging.info('[%d] Glue grammar: productions=%d', seg.id, len(glue))
            glue_grammars.append(glue)
    #  1c. main grammar
    main_grammar = load_grammar(seg.grammar)

    logging.info('[%d] Preparing source FSA: %s', seg.id, seg.src)
    # 2. Make input FSA and a pass-through grammar for the given segment
    #  2a. pass-through grammar
    _, pass_grammar = make_input(seg,
                                 list(itertools.chain([main_grammar], extra_grammars, glue_grammars)),
                                 semiring.inside,
                                 args.default_symbol)
    #  2b. input FSA
    input_dfa = make_dfa(seg.src_tokens())
    logging.info('[%d] Source FSA: states=%d arcs=%d', seg.id, input_dfa.n_states(), input_dfa.n_arcs())

    #  3a. put all (normal) grammars together
    if args.pass_through:
        grammars = list(itertools.chain([main_grammar], extra_grammars, [pass_grammar]))
    else:
        grammars = list(itertools.chain([main_grammar], extra_grammars))

    return grammars, glue_grammars, input_dfa


def biparse(seg, args, workspace, model):

    grammars, glue_grammars, input_dfa = prepare_for_parsing(seg, args, workspace)

    # 3. Parse input
    input_grammar = make_hypergraph_from_input_view(grammars, glue_grammars, DummyConstant(semiring.inside.one))

    logging.info('[%d] Parsing source', seg.id)
    #  3b. get a parser and intersect the source FSA
    goal_maker = GoalRuleMaker(args.goal, args.start)
    parser = NederhofParser(input_grammar, input_dfa, semiring.inside)
    root = input_grammar.fetch(Nonterminal(args.start))
    source_forest = parser.do(root, goal_maker.get_iview())
    pickle_it('{0}/forest/source.{1}'.format(workspace, seg.id), source_forest)

    #  3c. compute target projection
    logging.info('[%d] Computing target projection', seg.id)
    # Pass0
    target_forest = output_projection(source_forest, semiring.inside, TableLookupScorer(model.dummy))
    pickle_it('{0}/forest/target.{1}'.format(workspace, seg.id), target_forest)

    # 4. Local scoring
    logging.info('[%d] Computing local components', seg.id)
    #  4a. lookup components
    lookupffs = lookup_components(target_forest, model.lookup.extractors())
    #  4b. stateless components
    pickle_it('{0}/forest/lookup-ffs.{1}'.format(workspace, seg.id), lookupffs)
    statelessffs = stateless_components(target_forest, model.stateless.extractors())
    pickle_it('{0}/forest/stateless-ffs.{1}'.format(workspace, seg.id), statelessffs)

    # 5. Parse references
    logging.info('[%d] Preparing target FSA for %d references', seg.id, len(seg.refs))
    #  5a. make input FSA
    ref_dfa = make_dfa_set([ref.split() for ref in seg.refs], semiring.inside.one)
    #  5b. get a parser and intersect the target FSA
    logging.info('[%d] Parsing references: states=%d arcs=%d', seg.id, ref_dfa.n_states(), ref_dfa.n_arcs())
    goal_maker.update()
    parser = EarleyParser(target_forest, ref_dfa, semiring.inside)
    ref_forest = parser.do(0, goal_maker.get_oview())

    if not ref_forest:
        logging.info('[%d] The model cannot generate reference translations', seg.id)
        return False

    logging.info('[%d] Rescoring reference forest: nodes=%d edges=%d', seg.id, ref_forest.n_nodes(), ref_forest.n_edges())
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
    pickle_it('{0}/forest/ref.{1}'.format(workspace, seg.id), ref_forest)
    pickle_it('{0}/forest/all-ffs.{1}'.format(workspace, seg.id), rescorer.components())

    raw_viterbi = viterbi_derivation(ref_forest, AcyclicTopSortTable(ref_forest))
    score = derivation_value(ref_forest, raw_viterbi, semiring.inside)
    logging.info('[%d] Viterbi derivation [%s]: %s', seg.id, score, yield_string(ref_forest, raw_viterbi))

    return True


def expectations(seg, args, workspace, model):

    logging.info('[%d] Loading pickled reference forest and components', seg.id)
    forest = unpickle_it('{0}/forest/ref.{1}'.format(workspace, seg.id))
    components = unpickle_it('{0}/forest/all-ffs.{1}'.format(workspace, seg.id))
    tsort = AcyclicTopSortTable(forest)

    logging.info('[%d] Computing f(d|x,y)', seg.id)
    #components = [FComponents(itertools.chain(c1, c2, c3)) for c1, c2, c3 in zip(lookupffs, statelessffs, statefulffs)]
    #print('ALL')
    #for comp in components:
    #    print(comp)

    #weights = np.array([model.score(FComponents(itertools.chain(*components[e]))) for e in range(forest.n_edges())], dtype=ptypes.weight)
    weights = np.array([model.score(components[e]) for e in range(forest.n_edges())], dtype=ptypes.weight)
    fd = LookupFunction(weights)

    logging.info('[%d] Computing expectations', seg.id)
    values = acyclic_value_recursion(forest, tsort, semiring.inside, omega=fd)
    reversed_values = acyclic_reversed_value_recursion(forest, tsort, semiring.inside, values, omega=fd)
    edge_expectations = compute_edge_expectation(forest, semiring.inside, values, reversed_values,
                                                 omega=fd, normalise=True)
    expff = model.constant(semiring.prob.zero)

    for e in sorted(range(forest.n_edges()), key=lambda x: forest.head(x)):
        ue = semiring.inside.as_real(edge_expectations[e])
        #print('{0} | {1}'.format(ue, components[e]))
        expff = expff.hadamard(components[e].prod(ue), semiring.prob.plus)
    logging.info('[%d] Expected features\n%s', seg.id, expff)


def decode(seg, args, workspace, model):

    logging.info('[%d] Loading target forest', seg.id)
    forest = unpickle_it('{0}/forest/target.{1}'.format(workspace, seg.id))
    logging.info('[%d] Loading local components', seg.id)
    lookupffs = unpickle_it('{0}/forest/lookup-ffs.{1}'.format(workspace, seg.id))
    statelessffs = unpickle_it('{0}/forest/stateless-ffs.{1}'.format(workspace, seg.id))
    tsort = AcyclicTopSortTable(forest)
    #ehg = stateless_rescoring(ehg, StatelessScorer(model.stateless), semiring.inside)
    goal_maker = GoalRuleMaker(args.goal, args.start, n=2)

    rescorer = SlicedRescoring(forest, tsort,
                    StatelessScorer(model.dummy), StatefulScorer(model.stateful),
                    semiring.inside,
                    goal_maker.get_oview(),
                    OutputView(make_dead_srule()),
                    Terminal('<dead-end>'),
                    temperature0=args.temperature0)

    # here samples are represented as sequences of edge ids
    d0, markov_chain = rescorer.sample(args)



def core(args):

    workspace = make_dirs(args)

    logging.info('Loading feature extractors')
    # Load feature extractors
    extractors = load_feature_extractors(args)
    logging.info('Making model')
    # Load the model
    model = make_models(read_weights(args.weights), extractors)
    logging.debug('Model\n%s', model)

    logging.info('Parsing training data')
    segments = [SegmentMetaData.parse(input_str, grammar_dir=args.dev_grammars) for input_str in smart_ropen(args.dev)]
    # get D(x) for each and every segment
    # get D(x, y) for each and every segment and store it
    parsable = []
    for seg in segments:
        status = biparse(seg, args, workspace, model)
        if status:
            parsable.append(seg)

    logging.info('%d out of %d segments are bi-parsable', len(parsable), len(segments))

    for seg in parsable:
        expectations(seg, args, workspace, model)

    #if args.mode == 'all':
    #    batches = [segments]  # a single batch with all data points
    #elif args.mode == 'online':
    #    batches = list(segments)  # one data point per batch
    #else:
    #    perc = float(args.mode)/100
    #    b_size = len(segments) * perc





def main():
    args = get_argparser().parse_args()

    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    core(args)


if __name__ == '__main__':
    main()
