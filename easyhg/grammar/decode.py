"""
:Authors: - Wilker Aziz
"""

import logging
from .cmdline import argparser
from itertools import chain
from .semiring import SumTimes
from .cdec_format import load_grammar
from .sentence import make_sentence
from .rule import get_oov_scfg_productions
from .symbol import make_flat_symbol, Nonterminal
from .scfg import SCFG
from .nederhof import Nederhof
from .cfg import TopSortTable
from .inference import robust_inside
from .model import cdec_basic

def core(args):
    semiring = SumTimes

    # Load main grammars
    logging.info('Loading main grammar...')
    if args.grammarfmt != 'cdec':
        raise ValueError("For now I only recognise cdec's format: %s" % args.grammarfmt)

    linear_model = cdec_basic()

    scfg = load_grammar(args.grammar, linear_model)
    logging.info('Main grammar: terminals=%d nonterminals=%d productions=%d', scfg.n_terminals(), scfg.n_nonterminals(), len(scfg))

    # Load additional grammars
    main_grammars = [scfg]
    if args.extra_grammar:
        for grammar_path in args.extra_grammar:
            logging.info('Loading additional grammar: %s', grammar_path)
            grammar = load_grammar(grammar_path, linear_model)
            logging.info('Additional grammar: terminals=%d nonterminals=%d productions=%d', grammar.n_terminals(), grammar.n_nonterminals(), len(grammar))
            main_grammars.append(grammar)

    # Load glue grammars
    glue_grammars = []
    if args.glue_grammar:
        for glue_path in args.glue_grammar:
            logging.info('Loading glue grammar: %s', glue_path)
            glue = load_grammar(glue_path, linear_model)
            logging.info('Glue grammar: terminals=%d nonterminals=%d productions=%d', glue.n_terminals(), glue.n_nonterminals(), len(glue))
            glue_grammars.append(glue)

    # Report information about the main grammar
    # report_info(cfg, args)

    # Make surface lexicon
    surface_lexicon = set()
    for grammar in chain(main_grammars, glue_grammars):
        surface_lexicon.update(t.surface for t in grammar.sigma)

    # Parse sentence by sentence
    for input_str in args.input:

        # get an input automaton
        sentence = make_sentence(input_str, semiring, surface_lexicon, args.unkmodel)
        grammars = list(main_grammars)

        if args.unkmodel == 'passthrough':
            grammars.append(SCFG(get_oov_scfg_productions(sentence.oovs, args.unklhs, semiring.one)))

        logging.info('Parsing %d words: %s', len(sentence), sentence)

        # 1) get a parser
        parser = Nederhof([g.f_projection(semiring, marginalise=False) for g in grammars],
                          sentence.fsa,
                          glue_grammars=[g.f_projection(semiring, marginalise=False) for g in glue_grammars],
                          semiring=semiring,
                          make_symbol=make_flat_symbol)

        # 2) make a forest
        logging.info('Parsing...')
        forest = parser.do(root=Nonterminal(args.start), goal=Nonterminal(args.goal))
        if not forest:
            logging.error('NO PARSE FOUND')
            print()
            continue

        print(forest)

        logging.info('Top-sorting...')
        tsort = TopSortTable(forest)
        logging.info('Top symbol: %s', tsort.root())
        root = tsort.root()

        logging.info('Inside semiring=%s ...', str(semiring.__name__))
        itable = robust_inside(forest, tsort, semiring, infinity=args.generations)
        logging.info('Inside goal-value=%f', itable[root])




def configure():
    args = argparser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

    return args


def main():
    args = configure()

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
