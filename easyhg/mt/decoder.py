"""
:Authors: - Wilker Aziz
"""

import logging
import sys
import random
from itertools import chain
from multiprocessing import Pool
from functools import partial
import traceback

from easyhg.report import EmptyReport, IterationReport

from easyhg.grammar.semiring import SumTimes, MaxTimes
from easyhg.grammar.symbol import Nonterminal, make_span
from easyhg.grammar.cfg import CFG, TopSortTable, LazyTopSortTable, Terminal
from easyhg.grammar.rule import CFGProduction
from easyhg.grammar.utils import make_nltk_tree, inlinetree
from easyhg.grammar.projection import get_leaves

from easyhg.alg import Nederhof, Earley
from easyhg.alg.exact import EarleyRescoring, KBest, viterbi, AncestralSampler
from easyhg.alg.exact.rescoring import stateless_rescoring
from easyhg.alg.exact.inference import total_weight
from easyhg.alg.sliced import SlicedRescoring
from easyhg.alg.sliced.sampling import apply_filters, group_by_projection

from easyhg.mt.cmdline import argparser
from easyhg.mt.config import configure
from easyhg.mt.cdec_format import load_grammar
from easyhg.mt.segment import SegmentMetaData

from easyhg.ff.lm import KenLMScorer
from easyhg.ff.lookup import RuleTable
from easyhg.ff.stateless import WordPenalty, ArityPenalty

from easyhg.ff.scorer import StatefulScorer, StatelessScorer, TableLookupScorer, apply_scorers
from easyhg.ff.loglinear import LogLinearModel, read_weights
from easyhg.recipes import timeit, smart_wopen

from .input import make_input
from .workspace import make_dirs
from .report import save_kbest, save_viterbi
from .report import save_mc_derivations, save_mc_yields
from .report import save_mcmc_yields, save_mcmc_derivation, save_markov_chain




def load_feature_extractors(args):  # TODO: generalise it and use a configuration file
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

    if args.lm:
        extractor = KenLMScorer(uid=len(extractors),
                             name=args.lm[0],
                             order=int(args.lm[1]),
                             path=args.lm[2],
                             bos=Terminal(KenLMScorer.DEFAULT_BOS_STRING),
                             eos=Terminal(KenLMScorer.DEFAULT_EOS_STRING))
        extractors.append(extractor)
        logging.debug('Scorer: %r', extractor)

    return extractors


@timeit
def t_earley_rescoring(forest, root, model, semiring):
    if not model.stateful:
        if model.stateless:  # stateless scorers only
            forest = stateless_rescoring(forest, StatelessScorer(model), semiring, do_nothing={root})
    else:  # at least stateful scorers
        rescorer = EarleyRescoring(forest,
                                   semiring=semiring,
                                   stateful=StatefulScorer(model),
                                   stateless=StatelessScorer(model),
                                   do_nothing={root})
        forest = rescorer.do(root=root)
        root = make_span(root, None, None)
    return forest, root


def exact_rescoring(seg, forest, root, model, semiring, args, outdir):

    # rescore and compute rescoring time
    dt, (forest, root) = t_earley_rescoring(forest, root, model, semiring)
    time_report = {'rescoring': dt}

    if args.forest:
        with smart_wopen('{0}/forest/rescored.{1}.gz'.format(outdir, seg.id)) as fo:
            print('# FOREST terminals=%d nonterminals=%d rules=%d' % (forest.n_terminals(), forest.n_nonterminals(), len(forest)), file=fo)
            print(forest, file=fo)

    tsorter = LazyTopSortTable(forest)

    @timeit
    def t_viterbi():
        d, score = viterbi(forest,
                           tsorter.do(),
                           generations=args.generations)

        t = make_nltk_tree(d)
        logging.info('Viterbi derivation: %s %s', score, inlinetree(t))
        logging.info('Viterbi translation: %s', ' '.join(t.leaves()))
        save_viterbi('{0}/viterbi/{1}.gz'.format(outdir, seg.id),
                     d,
                     total_weight(d, MaxTimes))

    @timeit
    def t_kbest():
        logging.info('K-best...')
        kbestparser = KBest(forest,
                            root,
                            args.kbest).do()
        logging.info('Done!')
        derivations = list(kbestparser.iterderivations())
        omega_d = lambda der: total_weight(der, MaxTimes)
        save_kbest('{0}/kbest/{1}.gz'.format(outdir, seg.id),
                   derivations,
                   omega_d)

    @timeit
    def t_ancestral():
        logging.info('Sampling...')
        sampler = AncestralSampler(forest,
                                   tsorter.do(),
                                   generations=args.generations)
        samples = list(sampler.sample(args.samples))
        # group samples by derivation and yield
        derivations = group_by_projection(samples)
        yields = group_by_projection(samples, get_leaves)
        omega_d = lambda der: total_weight(der, semiring)
        # save the empirical distribution over derivations
        save_mc_derivations('{0}/ancestral/derivations/{1}.gz'.format(outdir, seg.id),
                            derivations, sampler.Z, omega_d, semiring)
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
    omega_d = lambda der: semiring.times(total_weight(der, semiring),  # the local part
                                         apply_scorers(der, stateless, stateful, semiring, {root}))  # the global part
    samples = []
    total_dt = 0
    for run in range(1, args.chains + 1):  # TODO: run chains in parallel?
        logging.info('Chain %d/%d', run, args.chains)
        dt, markov_chain = t_sample()
        total_dt += dt

        # get the final samples from the chain (after burn-in, lag, resampling, etc.)
        samples.extend(apply_filters(markov_chain, burn=args.burn, lag=args.lag, resample=args.resample))

        if args.save_chain:
            logging.info('Saving Markov chain (might take some time)...')
            save_markov_chain('{0}/slice/chain/{1}-{2}.gz'.format(outdir, seg.id, run),
                              markov_chain,
                              omega_d)

    # group by derivation
    derivations = group_by_projection(samples)
    # group by string
    yields = group_by_projection(samples, get_leaves)

    # save everything
    save_mcmc_derivation('{0}/slice/derivations/{1}.gz'.format(outdir, seg.id),
                         derivations,
                         omega_d)
    save_mcmc_yields('{0}/slice/yields/{1}.gz'.format(outdir, seg.id),
                     yields)


    # save a customised report
    report.save()

    return {'slice': total_dt}


@timeit
def t_decode(seg, extra_grammars, glue_grammars, model, args, outdir):
    semiring = SumTimes

    logging.info('Loading grammar: %s', seg.grammar)
    # load main SCFG from file
    main_grammar = load_grammar(seg.grammar)
    logging.info('Preparing input (%d): %s', seg.id, seg.src)
    # make input FSA and a pass-through grammar for the given segment
    input_fsa, pass_grammar = make_input(seg, list(chain([main_grammar], extra_grammars, glue_grammars)), semiring, args.default_symbol)
    # put all (normal) grammars together
    grammars = list(chain([main_grammar], extra_grammars, [pass_grammar])) if args.pass_through else list(chain([main_grammar], extra_grammars))
    # get input projection
    igrammars = [g.input_projection(semiring, weighted=False) for g in grammars]
    iglue = [g.input_projection(semiring, weighted=False) for g in glue_grammars]

    logging.info('Input: states=%d arcs=%d', input_fsa.n_states(), input_fsa.n_arcs())

    # 1) get a parser
    parser = Nederhof(igrammars,
                      input_fsa,
                      glue_grammars=iglue,
                      semiring=semiring)

    # 2) make a forest
    logging.info('Parsing input...')
    f_forest = parser.do(root=Nonterminal(args.start), goal=Nonterminal(args.goal))
    f_root = make_span(Nonterminal(args.goal), None, None)

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
                e_forest.add(CFGProduction(e_lhs, e_rhs, semiring.times(f_rule.weight, lookupscorer.score(r))))

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

    local_d, local_d_score = viterbi(e_forest, TopSortTable(e_forest), generations=args.generations)
    local_t = make_nltk_tree(local_d)
    logging.info('Local Viterbi derivation: %s %s', local_d_score, inlinetree(local_t))
    logging.info('Local Viterbi translation: %s', ' '.join(local_t.leaves()))

    if args.framework == 'exact':
        return exact_rescoring(seg, e_forest, e_root, model, semiring, args, outdir)
    else:
        return sliced_rescoring(seg, e_forest, e_root, model, semiring, args, outdir)


def traced_load_and_decode(seg, args, outdir):
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
        dt, others = t_decode(seg, extra_grammars, glue_grammars, model, args, outdir)
        print('[%d] decoding time: %s' % (seg.id, dt), file=sys.stdout)
        return dt, others
    except:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))


def core(args):

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
    #args, config = configure(argparser(), set_defaults=['Grammar', 'Parser'])  # TODO: use config file
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
        pass




if __name__ == '__main__':
    main()
