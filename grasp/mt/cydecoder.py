"""
Command line interface for hierarchical MT decoding.

:Authors: - Wilker Aziz
"""

import logging
import sys
import random
from itertools import chain
from multiprocessing import Pool
from functools import partial
import traceback
from types import SimpleNamespace

from grasp.io.report import EmptyReport, IterationReport
from grasp.io.results import save_kbest, save_viterbi
from grasp.io.results import save_mc_derivations, save_mc_yields
from grasp.io.results import save_mcmc_yields, save_mcmc_derivations, save_markov_chain

from grasp.cfg.symbol import Nonterminal, make_span
from grasp.cfg import CFG, TopSortTable, LazyTopSortTable, Terminal
from grasp.cfg.rule import NewCFGProduction as CFGProduction
from grasp.cfg.srule import SCFGProduction
from grasp.cfg.projection import DerivationYield
from grasp.cfg.model import DummyConstant
from grasp.cfg.srule import InputGroupView, OutputView

from grasp.mt.cmdline import argparser
from grasp.mt.cdec_format import load_grammar
from grasp.mt.segment import SegmentMetaData
from grasp.mt.input import make_input
from grasp.mt.workspace import make_dirs
from grasp.mt.util import load_feature_extractors, GoalRuleMaker

from grasp.recipes import timeit, smart_wopen

from grasp.scoring.scorer import StatefulScorer, StatelessScorer, TableLookupScorer
from grasp.scoring.model import Model, make_models
from grasp.scoring.util import read_weights

import grasp.semiring as semiring

from grasp.formal.fsa import make_dfa
from grasp.formal.scfgop import make_hypergraph_from_input_view
from grasp.formal.topsort import AcyclicTopSortTable
from grasp.formal.hg import make_derivation
from grasp.formal.traversal import bracketed_string, yield_string
from grasp.formal.scfgop import output_projection

from grasp.alg.rescoring import stateless_rescoring
from grasp.alg.deduction import NederhofParser
from grasp.alg.deduction import EarleyRescorer
from grasp.alg.inference import viterbi_derivation, AncestralSampler
from grasp.alg.chain import group_by_projection, group_by_identity, apply_batch_filters, apply_filters
from grasp.alg.value import derivation_value


def save_forest(hg, outdir, sid, name):
    with smart_wopen('{0}/forest/{1}.{2}.gz'.format(outdir, name, sid)) as fo:
            print('# FOREST', file=fo)
            print(hg, file=fo)


def my_save_viterbi(hg, tsort, omega_d, outdir, sid, name):
    raw_viterbi = viterbi_derivation(hg, tsort)
    viterbi = make_derivation(hg, raw_viterbi)
    score = omega_d(viterbi)
    logging.info('Viterbi derivation: %s', score)
    logging.info('Saving...')
    save_viterbi('{0}/viterbi/{1}.{2}.gz'.format(outdir, name, sid),
                 SimpleNamespace(derivation=viterbi,
                                 count=1,
                                 value=score),
                 get_projection=DerivationYield.derivation,
                 derivation2str=DerivationYield.string)

def traced_load_and_decode(seg, args, outdir):
    """Load grammars and decode. This version traces exception for convenience."""
    try:

        # Load feature extractors
        extractors = load_feature_extractors(args)
        # Load the model
        model = make_models(read_weights(args.weights, args.temperature), extractors)
        logging.debug('Model\n%s', model)
        # Load additional grammars
        extra_grammars = []
        if args.extra_grammar:
            for grammar_path in args.extra_grammar:
                logging.info('Loading additional grammar: %s', grammar_path)
                grammar = load_grammar(grammar_path)
                logging.info('Additional grammar: productions=%d', len(grammar))
                extra_grammars.append(grammar)

        # Load glue grammars
        glue_grammars = []
        if args.glue_grammar:
            for glue_path in args.glue_grammar:
                logging.info('Loading glue grammar: %s', glue_path)
                glue = load_grammar(glue_path)
                logging.info('Glue grammar: productions=%d', len(glue))
                glue_grammars.append(glue)

        print('[%d] decoding...' % seg.id, file=sys.stdout)
        dt, others = t_cy_decode(seg, extra_grammars, glue_grammars, model, args, outdir)
        print('[%d] decoding time: %s' % (seg.id, dt), file=sys.stdout)
        return dt, others
    except:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))




@timeit
def t_cy_decode(seg, extra_grammars, glue_grammars, model, args, outdir):
    """
    Decode (and time it).

    :param seg: a input segment
    :param extra_grammars: additional (normal) grammars
    :param glue_grammars: glue grammars
    :param model: a linear model
    :param args: command line options
    :param outdir: where to save results
    :return:
    """

    logging.info('Loading grammar: %s', seg.grammar)
    # load main SCFG from file
    main_grammar = load_grammar(seg.grammar)
    logging.info('Preparing input (%d): %s', seg.id, seg.src)
    # make input FSA and a pass-through grammar for the given segment
    _, pass_grammar = make_input(seg, list(chain([main_grammar], extra_grammars, glue_grammars)), semiring.inside, args.default_symbol)
    # put all (normal) grammars together
    grammars = list(chain([main_grammar], extra_grammars, [pass_grammar])) if args.pass_through else list(chain([main_grammar], extra_grammars))

    dfa = make_dfa(seg.src_tokens())

    logging.info('Input: states=%d arcs=%d', dfa.n_states(), dfa.n_arcs())

    hg = make_hypergraph_from_input_view(grammars, glue_grammars, DummyConstant(semiring.inside.one))
    #hg = make_hypergraph(grammars, glue_grammars, semiring.inside)

    goal_maker = GoalRuleMaker(args.goal, args.start)


    logging.info('Parsing input')
    # 1) get a parser
    parser = NederhofParser(hg, dfa, semiring.inside)
    root = hg.fetch(Nonterminal(args.start))
    hg = parser.do(root, goal_maker.get_iview())
    if args.forest:
        save_forest(hg, outdir, seg.id, "source")



    logging.info('Output projection and lookup scorers')
    # Pass0
    scorer = TableLookupScorer(model.lookup)
    ehg = output_projection(hg, semiring.inside, scorer)
    if args.forest:
        save_forest(ehg, outdir, seg.id, "pass0")

    # Topsort
    tsort = AcyclicTopSortTable(ehg)
    omega_d = lambda d: semiring.inside.times.reduce(d.weights())

    if True:  # Pass0 Viterbi
        my_save_viterbi(ehg, tsort, omega_d, outdir, seg.id, "pass0")

    if model.stateless:  # Pass1 Viterbi
        #from grasp.parsing.exact._rescoring import stateless_rescoring

        logging.info('Stateless rescoring...')
        # TODO: do we need "do_nothing" ?
        fast_rescorer = False

        if fast_rescorer:
            ehg = stateless_rescoring(ehg, StatelessScorer(model.stateless), semiring.inside)
        else:

            goal_maker.update()

            rescorer = EarleyRescorer(ehg,
                                      TableLookupScorer(model.dummy),
                                      StatelessScorer(model.stateless),
                                      StatefulScorer(model.dummy),
                                      semiring.inside,
                                      map_edges=False,
                                      keep_frepr=False)
            ehg = rescorer.do(tsort.root(), goal_maker.get_oview())
            tsort = AcyclicTopSortTable(ehg)  # EarleyRescorer does not preserve topsort

        if args.forest:
            save_forest(ehg, outdir, seg.id, "pass1")

        # Pass1 Viterbi
        my_save_viterbi(ehg, tsort, omega_d, outdir, seg.id, "pass1")


    if model.stateful:
        if args.framework == 'exact':
            logging.info('Stateful rescoring...')
            rescorer = EarleyRescorer(ehg, TableLookupScorer(model.dummy),
                                      StatelessScorer(model.dummy), StatefulScorer(model.stateful), semiring.inside)

            goal_maker.update()

            ehg = rescorer.do(tsort.root(), goal_maker.get_oview())

            if args.forest:
                save_forest(ehg, outdir, seg.id, "pass2")

            tsort = AcyclicTopSortTable(ehg)  # stateless_rescoring preserves topsort
            logging.info('Done!')

            # Pass2 Viterbi
            my_save_viterbi(ehg, tsort, omega_d, outdir, seg.id, "pass2")

            if args.samples > 0:
                sampler = AncestralSampler(ehg, tsort)
                samples = sampler.sample(args.samples)
                derivations = group_by_identity(samples)
                save_mc_derivations('{0}/exact/derivations/{1}.gz'.format(outdir, seg.id),
                                    derivations, sampler.Z,
                                    valuefunc=lambda d: derivation_value(ehg, d, semiring.inside),
                                    derivation2str=lambda d: bracketed_string(ehg, d))
                projections = group_by_projection(samples, lambda d: yield_string(ehg, d))
                save_mc_yields('{0}/exact/yields/{1}.gz'.format(outdir, seg.id), projections)

        else:
            from grasp.alg.rescoring import SlicedRescoring

            dead_rule = SCFGProduction(Nonterminal('X'), (Terminal('<dead-end>'),), (Terminal('<dead-end>'),), [], {'DeadRule': 1.0})

            rescorer = SlicedRescoring(ehg, tsort,
                            StatelessScorer(model.dummy), StatefulScorer(model.stateful),
                            semiring.inside,
                            OutputView(goal_maker.get_next_srule()),
                            OutputView(dead_rule),
                            Terminal('<dead-end>'),
                            temperature0=args.temperature0)

            # here samples are represented as sequences of edge ids
            d0, markov_chain = rescorer.sample(args)

            # top-down-left-right traversal including nonterminals
            # bracket it

            # apply usual MCMC heuristics (e.g. burn-in, lag)
            samples = apply_filters(markov_chain,
                                burn=args.burn,
                                lag=args.lag)

            # group by derivation (now a sample is represented by a Derivation object)
            derivations = group_by_identity(samples)
            save_mcmc_derivations('{0}/slice/derivations/{1}.gz'.format(outdir, seg.id),
                                  derivations,
                                  valuefunc=lambda d: d.score,
                                  derivation2str=lambda d: bracketed_string(ehg, d.edges))
            projections = group_by_projection(samples, lambda d: yield_string(ehg, d.edges))
            save_mcmc_yields('{0}/slice/yields/{1}.gz'.format(outdir, seg.id),
                             projections)

            if args.save_chain:
                markov_chain.appendleft(d0)
                save_markov_chain('{0}/slice/chain/{1}.gz'.format(outdir, seg.id),
                                  markov_chain,
                                  flat=True,
                                  valuefunc=lambda d: d.score,
                                  derivation2str=lambda d: bracketed_string(ehg, d.edges))




            #save_mcmc_yields('{0}/slice/trees/{1}.gz'.format(outdir, seg.id), trees)




    # TODO:
    # 1. Cythonize sliced rescoring
    # 2. Cleanup

    #if args.framework == 'exact':
    #    return exact_rescoring(seg, ehg, e_root, model, semiring.inside, args, outdir)
    #else:
    #    return sliced_rescoring(seg, ehg, e_root, model, semiring.inside, args, outdir)

    return {}



def core(args):
    """
    The main pipeline including multiprocessing.
    """

    outdir = make_dirs(args)

    # Load the segments
    segments = [SegmentMetaData.parse(input_str, grammar_dir=args.grammars) for input_str in args.input]
    if args.shuffle:
        random.shuffle(segments)

    args.input = None  # necessary because we cannot pickle the input stream (TODO: get rid of this ugly thing!)

    pool = Pool(args.cpus)
    results = pool.map(partial(traced_load_and_decode,
                               args=args,
                               outdir=outdir), segments)

    # time report
    time_report = IterationReport('{0}/time'.format(outdir))
    for seg, (dt, dt_detail) in sorted(zip(segments, results), key=lambda seg_info: seg_info[0].id):
        time_report.report(seg.id, total=dt, **dt_detail)
    time_report.save()

    print('Check output files in:', outdir, file=sys.stdout)


def main():
    args = argparser().parse_args()

    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    if args.profile:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
        core(args)
        pr.disable()
        pr.dump_stats(args.profile)
    else:
        core(args)


if __name__ == '__main__':
    main()
