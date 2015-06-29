"""
:Authors: - Wilker Aziz
"""

import logging
from itertools import chain

from easyhg.grammar.semiring import SumTimes
from easyhg.grammar.symbol import make_flat_symbol, Nonterminal
from easyhg.grammar.scfg import SCFG
from easyhg.grammar.nederhof import Nederhof
from easyhg.grammar.cfg import TopSortTable
from easyhg.grammar.inference import robust_inside
from easyhg.grammar.model import cdec_basic
from easyhg.mt.cmdline import argparser
from easyhg.mt.config import configure
from easyhg.mt.cdec_format import load_grammar
from easyhg.mt.segment import SegmentMetaData
from easyhg.grammar.symbol import Terminal
from easyhg.grammar.fsa import WDFSA
from easyhg.grammar.rule import SCFGProduction


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


def decode(seg, extra_grammars, glue_grammars, args):
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
    parser = Nederhof(igrammars,
                      input_fsa,
                      glue_grammars=iglue,
                      semiring=semiring,
                      make_symbol=make_flat_symbol)

    # 2) make a forest
    logging.info('Parsing...')
    forest = parser.do(root=Nonterminal(args.start), goal=Nonterminal(args.goal))
    if not forest:
        logging.error('NO PARSE FOUND')
        print()
        return
    elif args.forest:
        print('# FOREST terminals=%d nonterminals=%d rules=%d' % (forest.n_terminals(), forest.n_nonterminals(), len(forest)))
        print(forest)
        print()


    logging.info('Top-sorting...')
    tsort = TopSortTable(forest)
    logging.info('Top symbol: %s', tsort.root())
    root = tsort.root()

    logging.info('Inside semiring=%s ...', str(semiring.__name__))
    itable = robust_inside(forest, tsort, semiring, infinity=args.generations)
    logging.info('Inside goal-value=%f', itable[root])


def core(args):

    linear_model = cdec_basic()

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

    # Parse sentence by sentence
    for input_str in args.input:
        # get an input automaton
        seg = SegmentMetaData.parse(input_str, grammar_dir=args.grammars)
        decode(seg, extra_grammars, glue_grammars, args)


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
