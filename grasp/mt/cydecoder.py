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
from grasp.cfg.symbol import Nonterminal, make_span
from grasp.cfg import CFG, TopSortTable, LazyTopSortTable, Terminal
from grasp.cfg.rule import NewCFGProduction as CFGProduction
from grasp.cfg.srule import SCFGProduction
from grasp.cfg.projection import DerivationYield


from grasp.mt.cmdline import argparser
from grasp.mt.cdec_format import load_grammar
from grasp.mt.segment import SegmentMetaData
from grasp.mt.input import make_input
from grasp.mt.workspace import make_dirs
from grasp.io.results import save_kbest, save_viterbi
from grasp.io.results import save_mc_derivations, save_mc_yields
from grasp.io.results import save_mcmc_yields, save_mcmc_derivation, save_markov_chain
from grasp.recipes import timeit, smart_wopen


from grasp.scoring.stateless import WordPenalty, ArityPenalty
from grasp.scoring.lm import KenLM
from grasp.scoring.lm import StatelessLM
from grasp.scoring.lookup import RuleTable
from grasp.scoring.scorer import StatefulScorer, StatelessScorer, TableLookupScorer
from grasp.scoring.scorer import LogLinearModel
from grasp.scoring.util import read_weights
from grasp.ff.scorer import apply_scorers
import grasp.semiring as semiring

from grasp.formal.fsa import make_dfa
from grasp.formal.scfgop import make_hypergraph_from_input_view
from grasp.cfg.model import DummyConstant
from grasp.cfg.srule import InputGroupView, OutputView
from grasp.formal.topsort import AcyclicTopSortTable

from grasp.formal.hg import make_derivation

#from grasp.parsing.exact.deduction import Nederhof


from grasp.parsing.exact.rescoring import EarleyRescoring


from grasp.alg.rescoring import stateless_rescoring
from grasp.alg.deduction import NederhofParser
from grasp.alg.deduction import EarleyRescorer
from grasp.alg.inference import viterbi_derivation, AncestralSampler
from grasp.alg.chain import group_by_projection, group_by_identity, apply_batch_filters, apply_filters


def load_feature_extractors(args) -> 'list of extractors':  # TODO: generalise it and use a configuration file
    """
    Load feature extractors depending on command line options.

    For now we have the following extractors:

        * RuleTable: named features in the rule table (a lookup scorer)
        * WordPenalty
        * ArityPenalty
        * KenLMScorer

    :param args:
    :return: a vector of Extractor objects
    """
    extractors = []

    if args.rt:
        extractor = RuleTable(uid=len(extractors),
                              name='RuleTable')
        extractors.append(extractor)
        logging.debug('Scorer: %r', extractor)

    if args.wp:
        extractor = WordPenalty(uid=len(extractors),
                                name=args.wp[0],
                                penalty=float(args.wp[1]))
        extractors.append(extractor)
        logging.debug('Scorer: %r', extractor)

    if args.ap:
        extractor = ArityPenalty(uid=len(extractors),
                                 name=args.ap[0],
                                 penalty=float(args.ap[1]))
        extractors.append(extractor)
        logging.debug('Scorer: %r', extractor)

    if args.slm:
        extractor = StatelessLM(uid=len(extractors),
                                name=args.slm[0],
                                order=int(args.slm[1]),
                                path=args.slm[2])
        extractors.append(extractor)
        logging.debug('Scorer: %r', extractor)

    if args.lm:
        extractor = KenLM(uid=len(extractors),
                          name=args.lm[0],
                          order=int(args.lm[1]),
                          path=args.lm[2])
        extractors.append(extractor)
        logging.debug('Scorer: %r', extractor)

    return extractors


@timeit
def t_stateless_rescoring(forest, root, model, semiring) -> 'forest and its start symbol':
    """Locally rescore a forest."""
    if model.stateless:
        logging.info('Stateless rescoring...')
        forest = stateless_rescoring(forest, StatelessScorer(model), semiring, do_nothing={root})
        logging.info('Done!')
    return forest, root


@timeit
def t_earley_rescoring(forest, root, model, semiring) -> 'forest and its start symbol':
    """Exactly rescore a forest using a variant of the Earley algorithm."""
    logging.info('Stateful rescoring: %s', model.stateful)
    rescorer = EarleyRescoring(forest,
                               semiring=semiring,
                               stateful=StatefulScorer(model),
                               stateless=StatelessScorer(model),
                               do_nothing={root})
    forest = rescorer.do(root=root)
    root = make_span(root)
    return forest, root


def exact_rescoring(seg, forest, root, model, semiring, args, outdir):
    """
    Exact decoding strategies.

    :param seg: input segment
    :param forest: a CFG forest
    :param root: the forest's starty symbol
    :param model: a linear model
    :param semiring: a semiring
    :param args: command line options
    :param outdir: where to save results
    """

    # rescore and compute rescoring time
    if not model.stateful:
        dt, (forest, root) = t_stateless_rescoring(forest, root, model, semiring)
        derivation2str = DerivationYield.derivation
    else:
        dt, (forest, root) = t_earley_rescoring(forest, root, model, semiring)
        derivation2str = DerivationYield.tree  # we do not need to show states to users

    time_report = {'rescoring': dt}

    if args.forest:
        with smart_wopen('{0}/forest/rescored.{1}.gz'.format(outdir, seg.id)) as fo:
            print('# FOREST terminals=%d nonterminals=%d rules=%d' % (forest.n_terminals(),
                                                                      forest.n_nonterminals(),
                                                                      len(forest)), file=fo)
            print(forest, file=fo)

    tsorter = LazyTopSortTable(forest)

    @timeit
    def t_viterbi():
        d = viterbi_derivation(forest,
                               tsorter.do(),
                               generations=args.generations)

        logging.info('Viterbi derivation: %s %s', derivation_value(d), DerivationYield.tree(d))
        logging.info('Viterbi translation: %s', DerivationYield.string(d))
        save_viterbi('{0}/viterbi/{1}.gz'.format(outdir, seg.id),
                     d,
                     omega_d=derivation_value,
                     get_projection=DerivationYield.string,
                     derivation2str=derivation2str)

    @timeit
    def t_kbest():
        logging.info('K-best...')
        kbestparser = KBest(forest,
                            root,
                            args.kbest).do()
        logging.info('Done!')
        derivations = list(kbestparser.iterderivations())
        save_kbest('{0}/kbest/{1}.gz'.format(outdir, seg.id),
                   derivations,
                   omega_d=derivation_value,
                   get_projection=DerivationYield.string,
                   derivation2str=derivation2str)

    @timeit
    def t_ancestral():
        logging.info('Sampling...')
        sampler = AncestralSampler(forest,
                                   tsorter.do(),
                                   generations=args.generations)
        samples = list(sampler.sample(args.samples))
        # group samples by derivation and yield
        derivations = group_by_identity(samples)
        yields = group_by_projection(samples, DerivationYield.string)
        # save the empirical distribution over derivations
        save_mc_derivations('{0}/ancestral/derivations/{1}.gz'.format(outdir, seg.id),
                            derivations,
                            inside=sampler.Z,
                            omega_d=derivation_value,
                            derivation2str=derivation2str)
        # save the empirical distribution over strings
        save_mc_yields('{0}/ancestral/yields/{1}.gz'.format(outdir, seg.id),
                       yields)

    if args.viterbi:
        dt, _ = t_viterbi()
        time_report['viterbi'] = dt

    if args.kbest > 0:
        dt, _ = t_kbest()
        time_report['kbest'] = dt

    if args.samples > 0:
        dt, _ = t_ancestral()
        time_report['ancestral'] = dt

    return time_report


def sliced_rescoring(seg, forest, root, model, semiring, args, outdir):
    """
    Decode by slice sampling.

    :param seg: an input segment
    :param forest: a CFG forest
    :param root: the forest's start symbol
    :param model: a linear model
    :param semiring: a semiring
    :param args: command line options
    :param outdir: where to save results
    """

    # sometimes we want a very detailed report
    if args.report:
        report = IterationReport('{0}/slice/report/{1}'.format(outdir, seg.id))
    else:
        report = EmptyReport()

    # make scorers
    stateless = StatelessScorer(model)
    stateful = StatefulScorer(model)

    @timeit
    def t_sample():
        rescorer = SlicedRescoring(forest,
                                   TopSortTable(forest),
                                   stateful=stateful,
                                   stateless=stateless,
                                   do_nothing={root},
                                   generations=args.generations,
                                   temperature0=args.temperature0,
                                   report=report)
        # construct a Markov chain
        return rescorer.sample(args)

    # model score
    omega_d = lambda der: semiring.times(derivation_value(der, semiring),  # the local part
                                         apply_scorers(der, stateless, stateful, semiring, {root}))  # the global part
    samples = []
    total_dt = 0
    for run in range(1, args.chains + 1):  # TODO: run chains in parallel?
        logging.info('Chain %d/%d', run, args.chains)
        dt, markov_chain = t_sample()
        total_dt += dt

        # get the final samples from the chain (after burn-in, lag, resampling, etc.)
        samples.extend(apply_filters(markov_chain, burn=args.burn, lag=args.lag))

        if args.save_chain:
            logging.info('Saving Markov chain (might take some time)...')
            save_markov_chain('{0}/slice/chain/{1}-{2}.gz'.format(outdir, seg.id, run),
                              markov_chain,
                              omega_d)

    # group by derivation
    derivations = group_by_identity(samples)
    # group by string
    yields = group_by_projection(samples, DerivationYield.string)

    # save everything
    save_mcmc_derivation('{0}/slice/derivations/{1}.gz'.format(outdir, seg.id),
                         derivations,
                         omega_d)
    save_mcmc_yields('{0}/slice/yields/{1}.gz'.format(outdir, seg.id),
                     yields)

    # save a customised report
    report.save()

    return {'slice': total_dt}


def make_forest(hg):
    forest = CFG()
    for e in range(hg.n_edges()):
        lhs = hg.label(hg.head(e))
        rhs = [hg.label(n) for n in hg.tail(e)]
        forest.add(CFGProduction(lhs, rhs, hg.weight(e)))
    return forest


class GoalRuleMaker:

    def __init__(self, goal_str, start_str):
        self._n = 0
        self._goal_str = goal_str
        self._start_str = start_str

    def _make_goal_str(self, n=None):
        return '{0}{1}'.format(self._goal_str, self._n) if n is None else '{0}{1}'.format(self._goal_str, n)

    def update(self):
        self._n += 1

    def get_srule(self):
        if self._n == 0:
            rhs = [Nonterminal(self._start_str)]
        else:
            rhs = [Nonterminal(self._make_goal_str(self._n - 1))]
        return SCFGProduction(Nonterminal(self._make_goal_str()), rhs, rhs, [1], {'GoalRule': 1.0})

    def get_iview(self):
        return InputGroupView([self.get_srule()])

    def get_oview(self):
        return OutputView(self.get_srule())


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
    input_fsa, pass_grammar = make_input(seg, list(chain([main_grammar], extra_grammars, glue_grammars)), semiring.inside, args.default_symbol)
    # put all (normal) grammars together
    grammars = list(chain([main_grammar], extra_grammars, [pass_grammar])) if args.pass_through else list(chain([main_grammar], extra_grammars))

    logging.info('Input: states=%d arcs=%d', input_fsa.n_states(), input_fsa.n_arcs())



    from grasp.formal.scfgop import output_projection

    dfa = make_dfa(seg.src_tokens())

    hg = make_hypergraph_from_input_view(grammars, glue_grammars, DummyConstant(semiring.inside.one))
    #hg = make_hypergraph(grammars, glue_grammars, semiring.inside)

    goal_maker = GoalRuleMaker(args.goal, args.start)

    # 1) get a parser
    parser = NederhofParser(hg, dfa, semiring.inside)
    root = hg.fetch(Nonterminal(args.start))
    hg = parser.do(root, goal_maker.get_iview())
    if args.forest:
        save_forest(hg, outdir, seg.id, "source")

    # Pass0
    ehg = output_projection(hg, semiring.inside, TableLookupScorer(model))
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
        fast_rescorer = True

        if fast_rescorer:
            ehg = stateless_rescoring(ehg, StatelessScorer(model), semiring.inside)
        else:

            goal_maker.update()


            rescorer = EarleyRescorer(ehg, StatelessScorer(model), None, semiring.inside)
            ehg = rescorer.do(tsort.root(), goal_maker.get_oview())
            tsort = AcyclicTopSortTable(ehg)  # EarleyRescorer does not preserve topsort

        if args.forest:
            save_forest(ehg, outdir, seg.id, "pass1")

        logging.info('Done!')

        # Pass1 Viterbi
        my_save_viterbi(ehg, tsort, omega_d, outdir, seg.id, "pass1")

    if model.stateful:
        logging.info('Stateful rescoring...')
        rescorer = EarleyRescorer(ehg, None, StatefulScorer(model), semiring.inside)

        goal_maker.update()

        ehg = rescorer.do(tsort.root(), goal_maker.get_oview())

        if args.forest:
            save_forest(ehg, outdir, seg.id, "pass2")

        tsort = AcyclicTopSortTable(ehg)  # stateless_rescoring preserves topsort
        logging.info('Done!')

        # Pass2 Viterbi
        my_save_viterbi(ehg, tsort, omega_d, outdir, seg.id, "pass2")



    # TODO:
    # 1. Cythonize sliced rescoring
    # 2. Cleanup

    #if args.framework == 'exact':
    #    return exact_rescoring(seg, ehg, e_root, model, semiring.inside, args, outdir)
    #else:
    #    return sliced_rescoring(seg, ehg, e_root, model, semiring.inside, args, outdir)

    return {}

@timeit
def t_decode(seg, extra_grammars, glue_grammars, model, args, outdir):
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
    #semiring = SumTimes

    logging.info('Loading grammar: %s', seg.grammar)
    # load main SCFG from file
    main_grammar = load_grammar(seg.grammar)
    logging.info('Preparing input (%d): %s', seg.id, seg.src)
    # make input FSA and a pass-through grammar for the given segment
    input_fsa, pass_grammar = make_input(seg, list(chain([main_grammar], extra_grammars, glue_grammars)), semiring.inside, args.default_symbol)
    # put all (normal) grammars together
    grammars = list(chain([main_grammar], extra_grammars, [pass_grammar])) if args.pass_through else list(chain([main_grammar], extra_grammars))
    # get input projection
    igrammars = [g.input_projection(semiring.inside, weighted=False) for g in grammars]
    iglue = [g.input_projection(semiring.inside, weighted=False) for g in glue_grammars]

    logging.info('Input: states=%d arcs=%d', input_fsa.n_states(), input_fsa.n_arcs())

    # 1) get a parser
    from grasp.parsing.exact.nederhof import Nederhof
    parser = Nederhof(igrammars,
                      input_fsa,
                      glue_grammars=iglue,
                      semiring=semiring.inside)

    # 2) make a forest
    logging.info('Parsing input...')
    f_forest = parser.do(root=Nonterminal(args.start), goal=Nonterminal(args.goal))
    f_root = make_span(Nonterminal(args.goal))

    if not f_forest:
        logging.error('NO PARSE FOUND')
        return
    elif args.forest:
        with smart_wopen('{0}/forest/source.{1}.gz'.format(outdir, seg.id)) as fo:
            print('# FOREST terminals=%d nonterminals=%d rules=%d' % (f_forest.n_terminals(), f_forest.n_nonterminals(), len(f_forest)), file=fo)
            print(f_forest, file=fo)

    logging.info('Target projection...')

    #TODO: clean up this (target projection)
    lookupscorer = TableLookupScorer(model)

    e_forest = CFG()
    for f_rule in f_forest:
        base_lhs = f_rule.lhs.base
        base_rhs = tuple(s if isinstance(s, Terminal) else s.base for s in f_rule.rhs)
        e_lhs = f_rule.lhs
        for grammar in chain(grammars, glue_grammars):
            for r in grammar.iteroutputrules(base_lhs, base_rhs):
                alignment = iter(r.alignment)
                f_nts = tuple(filter(lambda s: isinstance(s, Nonterminal), f_rule.rhs))
                e_rhs = [s if isinstance(s, Terminal) else f_nts[next(alignment) - 1] for s in r.orhs]
                e_forest.add(CFGProduction(e_lhs, e_rhs, semiring.inside.times(f_rule.weight, lookupscorer.score(r))))

    # goal rules are treated independently because they are not part of the original grammar.
    for goal_rule in f_forest.iterrules(f_root):
        gr = CFGProduction(goal_rule.lhs, [s for s in goal_rule.rhs], goal_rule.weight)
        e_forest.add(gr)
        logging.info('Goal rule: %s', gr)

    # the target forest has exactly the same root symbol as the source forest
    e_root = f_root

    if args.forest:
        with smart_wopen('{0}/forest/target.{1}.gz'.format(outdir, seg.id)) as fo:
            print('# FOREST terminals=%d nonterminals=%d rules=%d' % (e_forest.n_terminals(), e_forest.n_nonterminals(), len(e_forest)), file=fo)
            print(e_forest, file=fo)

    local_d = viterbi_derivation(e_forest, TopSortTable(e_forest), generations=args.generations)
    logging.info('Local Viterbi derivation: %s %s', derivation_value(local_d), DerivationYield.derivation(local_d))
    logging.info('Local Viterbi translation: %s', DerivationYield.string(local_d))

    if args.framework == 'exact':
        return exact_rescoring(seg, e_forest, e_root, model, semiring.inside, args, outdir)
    else:
        return sliced_rescoring(seg, e_forest, e_root, model, semiring.inside, args, outdir)


def traced_load_and_decode(seg, args, outdir):
    """Load grammars and decode. This version traces exception for convenience."""
    try:

        # Load feature extractors
        extractors = load_feature_extractors(args)
        # Load the model
        model = LogLinearModel(read_weights(args.weights, args.temperature), extractors)

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
