"""
:Authors: - Wilker Aziz
"""

import logging
import os
from itertools import chain

from easyhg.grammar.semiring import SumTimes, MaxTimes, Count
from easyhg.grammar.symbol import make_flat_symbol, make_recursive_symbol, Nonterminal, Terminal, flatten_symbol
from easyhg.grammar.scfg import SCFG
from easyhg.grammar.nederhof import Nederhof
from easyhg.grammar.earley import Earley
from easyhg.grammar.cfg import CFG, TopSortTable
from easyhg.grammar.inference import robust_inside
from easyhg.grammar.model import cdec_basic, load_cdef_file
from easyhg.mt.cmdline import argparser
from easyhg.mt.config import configure
from easyhg.mt.cdec_format import load_grammar
from easyhg.mt.segment import SegmentMetaData
from easyhg.grammar.symbol import Terminal
from easyhg.grammar.fsa import WDFSA
from easyhg.grammar.rule import CFGProduction, SCFGProduction
from easyhg.grammar.utils import make_unique_directory, smart_wopen
from easyhg.ff.lm import KenLMScorer
from easyhg.ff.scorer import StatefulScorerWrapper
from easyhg.ff.stateless import WordPenalty
from easyhg.grammar.rescore import EarleyRescoring
from easyhg.grammar.inference import optimise, total_weight
from easyhg.grammar.kbest import KBest
from easyhg.grammar.projection import string as projected_string
from easyhg.grammar.utils import make_nltk_tree, inlinetree


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



def viterbi(forest, args):
    semiring = MaxTimes

    logging.info('Top-sorting...')
    tsort = TopSortTable(forest)
    logging.info('Top symbol: %s', tsort.root())
    root = tsort.root()

    logging.info('Inside semiring=%s ...', str(semiring.__name__))
    itable = robust_inside(forest, tsort, semiring, infinity=args.generations)
    logging.info('Inside goal-value=%f', itable[root])

    logging.info('Viterbi...')
    d = optimise(forest, root, MaxTimes, Iv=itable)
    logging.info('Done!')
    for e in d:
        print(e)
    print(itable[root])
    print(inlinetree(make_nltk_tree(d)))

    #return Result([(d, 1, itable[root])])

def kbest(forest, args):
    root = Nonterminal(args.goal)
    logging.info('K-best...')
    kbestparser = KBest(forest,
                        root,
                        args.kbest,
                        MaxTimes,
                        traversal=projected_string,
                        uniqueness=False).do()
    logging.info('Done!')
    for k, d in enumerate(kbestparser.iterderivations()):
        score = total_weight(d, MaxTimes)
        print(inlinetree(make_nltk_tree(d)))
        print(score)


def decode(seg, extra_grammars, glue_grammars, model, scorers, args, outdir):
    semiring = SumTimes
    logging.info('Loading grammar: %s', seg.grammar)
    # load main SCFG from file
    main_grammar = load_grammar(seg.grammar)
    logging.info('Preparing input: %s', seg.src)
    # make input FSA and a pass-through grammar for the given segment
    input_fsa, pass_grammar = make_input(seg, list(chain([main_grammar], extra_grammars, glue_grammars)), semiring, args.default_symbol)
    # put all (normal) grammars together
    grammars = list(chain([main_grammar], extra_grammars, [pass_grammar])) if args.pass_through else list(chain([main_grammar], extra_grammars))
    # get input projection
    igrammars = [g.input_projection(semiring, weighted=False) for g in grammars]
    iglue = [g.input_projection(semiring, weighted=False) for g in glue_grammars]

    logging.info('Input: states=%d arcs=%d', input_fsa.n_states(), input_fsa.n_arcs())

    # 1) get a parser
    parser = Earley(igrammars,
                      input_fsa,
                      glue_grammars=iglue,
                      semiring=semiring,
                      make_symbol=make_recursive_symbol)

    # 2) make a forest
    logging.info('Parsing...')
    f_forest = parser.do(root=Nonterminal(args.start), goal=Nonterminal(args.goal))

    if not f_forest:
        logging.error('NO PARSE FOUND')
        print()
        return
    elif args.forest:
        with smart_wopen('{0}/forest/source.{1}.gz'.format(outdir, seg.id)) as fo:
            print('# FOREST terminals=%d nonterminals=%d rules=%d' % (f_forest.n_terminals(), f_forest.n_nonterminals(), len(f_forest)), file=fo)
            print(f_forest.pprint(flatten_symbol), file=fo)

    #TODO: clean up this (target projection)
    e_forest = CFG()
    for f_rule in f_forest:
        base_lhs = f_rule.lhs.label[0]
        base_rhs = tuple(s if isinstance(s, Terminal) else s.label[0] for s in f_rule.rhs)
        e_lhs = flatten_symbol(f_rule.lhs)
        for grammar in chain(grammars, glue_grammars):
            for r in grammar.iteroutputrules(base_lhs, base_rhs):
                alignment = iter(r.alignment)
                f_nts = tuple(filter(lambda s: isinstance(s, Nonterminal), f_rule.rhs))
                e_rhs = [s if isinstance(s, Terminal) else flatten_symbol(f_nts[next(alignment) - 1]) for s in r.orhs]
                e_forest.add(CFGProduction(e_lhs, e_rhs, model.dot(r.fvpairs)))

    for goal_rule in f_forest.iterrules(Nonterminal((Nonterminal(args.goal), None, None))):
        e_forest.add(CFGProduction(flatten_symbol(goal_rule.lhs), [flatten_symbol(s) for s in goal_rule.rhs], goal_rule.weight))

    if args.forest:
        with smart_wopen('{0}/forest/target.{1}.gz'.format(outdir, seg.id)) as fo:
            print('# FOREST terminals=%d nonterminals=%d rules=%d' % (e_forest.n_terminals(), e_forest.n_nonterminals(), len(e_forest)), file=fo)
            print(e_forest, file=fo)

    #count(e_forest, args)
    viterbi(e_forest, args)
    #kbest(e_forest, args)

    if args.lm:
        rescorer = EarleyRescoring(e_forest, StatefulScorerWrapper([scorers[0]]), semiring, make_flat_symbol)
        rescored, new_goal = rescorer.do(root=Nonterminal(args.goal), goal=Nonterminal('{0}{1}'.format(args.goal, scorers[0].id)))

        if args.forest:
            with smart_wopen('{0}/forest/rescored.{1}.gz'.format(outdir, seg.id)) as fo:
                print('# FOREST terminals=%d nonterminals=%d rules=%d' % (rescored.n_terminals(), rescored.n_nonterminals(), len(rescored)), file=fo)
                print(rescored, file=fo)

        viterbi(rescored, args)






def make_dirs(args):
    """
    Make output directories and saves the command line arguments for documentation purpose.

    :param args: command line arguments
    :return: main output directory within workspace (prefix is a timestamp and suffix is a unique random string)
    """

    # create the workspace if missing
    logging.info('Workspace: %s', args.workspace)
    if not os.path.exists(args.workspace):
        os.makedirs(args.workspace)

    # create a unique experiment area
    outdir = make_unique_directory(args.workspace)
    logging.info('Writing files to: %s', outdir)

    # create output directories for the several inference algorithms
    if args.viterbi:
        os.makedirs('{0}/viterbi'.format(outdir))
    if args.kbest > 0:
        os.makedirs('{0}/kbest'.format(outdir))
    if args.samples > 0:
        if args.framework == 'exact':
            os.makedirs('{0}/ancestral'.format(outdir))
        elif args.framework == 'slice':
            os.makedirs('{0}/slice'.format(outdir))
        elif args.framework == 'gibbs':
            os.makedirs('{0}/gibbs'.format(outdir))
    if args.forest:
        os.makedirs('{0}/forest'.format(outdir))
    if args.count:
        os.makedirs('{0}/count'.format(outdir))
    if args.history:
        os.makedirs('{0}/history'.format(outdir))

    # write the command line arguments to an ini file
    args_ini = '{0}/args.ini'.format(outdir)
    logging.info('Writing command line arguments to: %s', args_ini)
    with open(args_ini, 'w') as fo:
        for k, v in sorted(vars(args).items()):
            print('{0}={1}'.format(k,repr(v)),file=fo)

    return outdir


def load_scorers(linear_model, args):
    scorers = []
    if args.lm:
        logging.info('Loading language model: order=%s path=%s', args.lm[0], args.lm[1])

        scorer = KenLMScorer(uid=len(scorers),
                 name=KenLMScorer.DEFAULT_FNAME,
                 weights=[linear_model.get(KenLMScorer.DEFAULT_FNAME),
                          linear_model.get('{0}_OOV'.format(KenLMScorer.DEFAULT_FNAME))],
                 order=int(args.lm[0]),
                 path=args.lm[1],
                 bos=Terminal(KenLMScorer.DEFAULT_BOS_STRING),
                 eos=Terminal(KenLMScorer.DEFAULT_EOS_STRING))
        scorers.append(scorer)
    if args.wp:

        logging.info('Decoding with WordPenalty')
        scorer = WordPenalty(uid=len(scorers),
                             name='WordPenalty',
                             weights=[linear_model.get('WordPenalty')])
        scorers.append(scorer)

    return scorers


def core(args):

    # load the linear model
    if args.weights:
        linear_model = load_cdef_file(args.weights)
    else:
        linear_model = cdec_basic()
    logging.debug('Linear model: %s', linear_model)

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
    scorers = load_scorers(linear_model, args)

    outdir = make_dirs(args)

    # Parse sentence by sentence
    for input_str in args.input:
        # get an input automaton
        seg = SegmentMetaData.parse(input_str, grammar_dir=args.grammars)
        decode(seg, extra_grammars, glue_grammars, linear_model, scorers, args, outdir)

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
