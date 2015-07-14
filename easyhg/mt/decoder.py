"""
:Authors: - Wilker Aziz
"""

import logging
import os
from itertools import chain
from collections import Counter

from easyhg.grammar.semiring import SumTimes, MaxTimes, Count
#from easyhg.grammar.symbol import make_flat_symbol, make_recursive_symbol,
from easyhg.grammar.symbol import Nonterminal, make_span
from easyhg.grammar.scfg import SCFG
from easyhg.grammar.cfg import CFG, TopSortTable, Terminal
from easyhg.grammar.fsa import WDFSA
from easyhg.grammar.rule import CFGProduction, SCFGProduction
from easyhg.grammar.utils import make_unique_directory, smart_wopen, make_nltk_tree, inlinetree
from easyhg.grammar.projection import get_leaves
from easyhg.grammar.result import Result
from easyhg.grammar.report import save_kbest, save_viterbi, save_mc, save_mcmc, save_sample_history, save_flat_history

from easyhg.alg import Nederhof, Earley
from easyhg.alg.exact import EarleyRescoring, KBest, viterbi, AncestralSampler
from easyhg.alg.exact.rescoring import stateless_rescoring
from easyhg.alg.exact.inference import robust_inside, optimise, total_weight
from easyhg.alg.sliced import SlicedRescoring, make_result, make_result_simple

from easyhg.mt.cmdline import argparser
from easyhg.mt.config import configure
from easyhg.mt.cdec_format import load_grammar
from easyhg.mt.segment import SegmentMetaData

from easyhg.ff.lm import KenLMScorer
#from easyhg.ff.scorer import StatefulScorerWrapper
from easyhg.ff.lookup import RuleTable
from easyhg.ff.stateless import WordPenalty, ArityPenalty

from easyhg.ff.scorer import StatefulScorer, StatelessScorer, TableLookupScorer
from easyhg.ff.loglinear import LogLinearModel, read_weights, cdec_basic
from easyhg.recipes import timeit


def make_input(seg, grammars, semiring, unk_lhs):
    """
    Make an input fsa for an input segment as well as its pass-through grammar.
    :param seg: a Segment object.
    :param grammars: a sequence of SCFGs.
    :param semiring: must provide `one`.
    :return: the input WDFSA, the pass-through grammar
    """
    fsa = WDFSA()
    pass_grammar = SCFG()
    unk = Nonterminal(unk_lhs)
    tokens = seg.src.split()
    for i, token in enumerate(tokens):
        word = Terminal(token)
        if any(g.in_ivocab(word) for g in grammars):
            pass_grammar.add(SCFGProduction.create(unk,
                                                   [word],
                                                   [word],
                                                   {'PassThrough': 1.0}))
        else:
            pass_grammar.add(SCFGProduction.create(unk,
                                                   [word],
                                                   [word],
                                                   {'PassThrough': 1.0,
                                                    'Unknown': 1.0}))
        fsa.add_arc(i, i + 1, word, semiring.one)
    fsa.make_initial(0)
    fsa.make_final(len(tokens))
    return fsa, pass_grammar


def count(forest, args):
    semiring = Count

    logging.info('Top-sorting...')
    tsort = TopSortTable(forest)
    logging.info('Top symbol: %s', tsort.root())
    root = tsort.root()

    logging.info('Inside semiring=%s ...', str(semiring.__name__))
    itable = robust_inside(forest, tsort, semiring, infinity=args.generations, omega=lambda e: semiring.one)
    logging.info('Inside goal-value=%f', itable[root])


def exact_rescoring(seg, forest, root, model, semiring, args, outdir):
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

    if args.forest:
        with smart_wopen('{0}/forest/rescored.{1}.gz'.format(outdir, seg.id)) as fo:
            print('# FOREST terminals=%d nonterminals=%d rules=%d' % (forest.n_terminals(), forest.n_nonterminals(), len(forest)), file=fo)
            print(forest, file=fo)

    if args.viterbi:
        d, score = viterbi(forest, TopSortTable(forest), generations=args.generations)
        t = make_nltk_tree(d)
        logging.info('Viterbi derivation: %s %s', score, inlinetree(t))
        logging.info('Viterbi translation: %s', ' '.join(t.leaves()))
        save_viterbi('{0}/viterbi/{1}.gz'.format(outdir, seg.id), Result([(d, 1, score)]))

    if args.kbest > 0:
        logging.info('K-best...')
        kbestparser = KBest(forest,
                            root,
                            args.kbest).do()
        logging.info('Done!')
        R = Result()
        for k, d in enumerate(kbestparser.iterderivations()):
            score = total_weight(d, MaxTimes)
            R.append(d, 1, score)
        save_kbest('{0}/kbest/{1}.gz'.format(outdir, seg.id), R)

    if args.samples > 0:
        logging.info('Sampling...')
        sampler = AncestralSampler(forest, TopSortTable(forest), generations=args.generations)
        #counts = Counter(get_leaves(d) for d in sampler.sample(args.samples))
        #for y, n in counts.most_common():
        #    print(float(n)/args.samples, ' '.join(x.surface for x in y))

        count = Counter(sampler.sample(args.samples))
        R = Result(Z=sampler.Z)
        for d, n in count.most_common():
            score = total_weight(d, SumTimes)
            R.append(d, n, score)
        save_mc('{0}/ancestral/{1}.gz'.format(outdir, seg.id), R)


@timeit
def decode(seg, extra_grammars, glue_grammars, model, args, outdir):
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


    # I need to treat the goal rules independently because they are not part of the original grammar.
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



    #count(e_forest, args)

    #kbest(e_forest, args)
    d, score = viterbi(e_forest, TopSortTable(e_forest), generations=args.generations)
    t = make_nltk_tree(d)
    logging.info('Viterbi derivation: %s %s', score, inlinetree(t))
    logging.info('Viterbi translation: %s', ' '.join(t.leaves()))
    logging.info('Viterbi score: %s', score)

    if args.framework == 'exact':
        exact_rescoring(seg, e_forest, e_root, model, semiring, args, outdir)
    else:
        #e_forest = stateless_rescoring(e_forest, StatelessScorer(model), semiring, {e_root})
        rescorer = SlicedRescoring(e_forest,
                                   TopSortTable(e_forest),
                                   stateful=StatefulScorer(model),
                                   stateless=StatelessScorer(model),
                                   do_nothing={e_root},
                                   generations=args.generations,
                                   temperature0=args.temperature0)

        history = rescorer.sample(args)

        results = make_result_simple(history, burn=args.burn, lag=args.lag, resample=args.resample)
        save_mcmc('{0}/slice-{1}/{2}.gz'.format(outdir, args.within, seg.id), results)
        if args.history:
            save_flat_history('{0}/slice-{1}/history/{2}.gz'.format(outdir, args.within, seg.id), history)


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

    # create output directories for the several inference algorithms
    if args.viterbi:
        os.makedirs('{0}/viterbi'.format(outdir), exist_ok=exist_ok)
    if args.kbest > 0:
        os.makedirs('{0}/kbest'.format(outdir), exist_ok=exist_ok)
    if args.samples > 0:
        if args.framework == 'exact':
            os.makedirs('{0}/ancestral'.format(outdir), exist_ok=exist_ok)
        elif args.framework == 'slice':
            os.makedirs('{0}/slice-{1}'.format(outdir, args.within), exist_ok=exist_ok)
            if args.history:
                os.makedirs('{0}/slice-{1}/history'.format(outdir, args.within), exist_ok=exist_ok)
    if args.forest:
        os.makedirs('{0}/forest'.format(outdir), exist_ok=exist_ok)
    if args.count:
        os.makedirs('{0}/count'.format(outdir), exist_ok=exist_ok)

    # write the command line arguments to an ini file
    args_ini = '{0}/args.ini'.format(outdir)
    logging.info('Writing command line arguments to: %s', args_ini)
    with open(args_ini, 'w') as fo:
        for k, v in sorted(vars(args).items()):
            print('{0}={1}'.format(k,repr(v)),file=fo)

    return outdir


def load_scorers(args):
    scorers = []

    if args.rt:
        scorer = RuleTable(uid=len(scorers),
                           name='RuleTable')
        scorers.append(scorer)
        logging.debug('Scorer: %r', scorer)

    if args.wp:
        scorer = WordPenalty(uid=len(scorers),
                             name=args.wp[0],
                             penalty=float(args.wp[1]))
        scorers.append(scorer)
        logging.debug('Scorer: %r', scorer)

    if args.ap:
        scorer = ArityPenalty(uid=len(scorers),
                              name=args.ap[0],
                              penalty=float(args.ap[1]))
        scorers.append(scorer)
        logging.debug('Scorer: %r', scorer)

    if args.lm:
        scorer = KenLMScorer(uid=len(scorers),
                             name=args.lm[0],
                             order=int(args.lm[1]),
                             path=args.lm[2],
                             bos=Terminal(KenLMScorer.DEFAULT_BOS_STRING),
                             eos=Terminal(KenLMScorer.DEFAULT_EOS_STRING))
        scorers.append(scorer)
        logging.debug('Scorer: %r', scorer)

    return scorers


def core(args):

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

    # Load scores
    scorers = load_scorers(args)

    # Load the model
    model = LogLinearModel(read_weights(args.weights, args.temperature), scorers)  # TODO: uniform initialisation
    outdir = make_dirs(args)

    # Parse sentence by sentence
    for input_str in args.input:
        # get an input automaton
        seg = SegmentMetaData.parse(input_str, grammar_dir=args.grammars)
        dt, _ = decode(seg, extra_grammars, glue_grammars, model, args, outdir)
        logging.info('Decoding time %d: %s', seg.id, dt)

    logging.info('Check output files in: %s', outdir)


def main():
    args, config = configure(argparser(), set_defaults=['Grammar', 'Parser'])  # TODO: use config file

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
