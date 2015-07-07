"""
Initialisation heuristics.

    - ITG
    - Coarse labels


TODO:
    - a switch for using slice sampling within heuristics

:Authors: - Wilker Aziz
"""

import logging
from collections import defaultdict
from .symbol import Nonterminal, make_recursive_symbol
from .semiring import SumTimes
from .inference import robust_inside, sample
from .cfg import TopSortTable, CFG
from .coarse import itercoarse, refine_conditions, iteritg
from .nederhof import Nederhof as ExactNederhof


def initialise_coarse(input_fsa, grammars, glue_grammars, options):
    semiring = SumTimes
    coarse_grammar = CFG()
    for g in grammars:
        for r in itercoarse(g, semiring):
            coarse_grammar.add(r)
    coarse_glue = CFG()
    for g in glue_grammars:
        for r in itercoarse(g, semiring):
            coarse_glue.add(r)

    logging.info('Coarse grammar: terminals=%d nonterminals=%d rules=%d', coarse_grammar.n_terminals(),
                 coarse_grammar.n_nonterminals(), len(coarse_grammar))
    parser = ExactNederhof([coarse_grammar], input_fsa,
                           glue_grammars=[coarse_glue],
                           semiring=semiring,
                           make_symbol=make_recursive_symbol)
    forest = parser.do(root=Nonterminal(options.start), goal=Nonterminal(options.goal))
    if not forest:
        logging.info('The coarse grammar cannot parse this input.')
        return None

    tsort = TopSortTable(forest)
    root = tsort.root()
    inside_nodes = robust_inside(forest, tsort, semiring, infinity=options.generations)
    d = list(sample(forest, root, semiring, Iv=inside_nodes, N=1))[0]

    spans = defaultdict(set)
    for r in d:
        spans[r.lhs.label[0]].add(r.lhs.label[1:])

    return refine_conditions(spans, grammars[0], semiring)


def initialise_itg(input_fsa, grammars, glue_grammars, options):
    semiring = SumTimes
    itg_grammar = CFG()
    for g in grammars:
        for r in iteritg(g):
            itg_grammar.add(r)
    itg_glue = CFG()
    for g in glue_grammars:
        for r in iteritg(g):
            itg_glue.add(r)

    logging.info('ITG grammar: terminals=%d nonterminals=%d rules=%d', itg_grammar.n_terminals(),
                 itg_grammar.n_nonterminals(), len(itg_grammar))
    parser = ExactNederhof([itg_grammar], input_fsa,
                           glue_grammars=[itg_glue],
                           semiring=semiring,
                           make_symbol=make_recursive_symbol)
    forest = parser.do(root=Nonterminal(options.start), goal=Nonterminal(options.goal))
    if not forest:
        logging.info('The ITG grammar cannot parse this input.')
        return None

    tsort = TopSortTable(forest)
    root = tsort.root()
    inside_nodes = robust_inside(forest, tsort, semiring, infinity=options.generations)
    d = list(sample(forest, root, semiring, Iv=inside_nodes, N=1))[0]

    return {r.lhs.label: SumTimes.as_real(r.weight) for r in d}


def attempt_initialisation(input_fsa, grammars, glue_grammars, options):
    if not options.heuristic:
        return None
    elif options.heuristic == 'itg':
        return initialise_itg(input_fsa, grammars, glue_grammars, options)
    elif options.heuristic == 'coarse':
        return initialise_coarse(input_fsa, grammars, glue_grammars, options)
    else:
        raise ValueError('Unknown initialisation heuristic: %s' % options.heuristic)