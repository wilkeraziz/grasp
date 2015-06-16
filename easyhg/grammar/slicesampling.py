"""
This module is an interface for parsing as intersection.
One can choose from all available implementations.

:Authors: - Wilker Aziz
"""

import logging
from collections import defaultdict, Counter
from itertools import chain

from .symbol import Nonterminal, make_recursive_symbol
from .semiring import SumTimes, Count
from .slicevars import SliceVariables
from .slicednederhof import Nederhof
from .inference import robust_inside, sample, total_weight
from .utils import make_nltk_tree, inlinetree
from . import heuristic
from .reader import load_grammar
from .sentence import make_sentence
from .cfg import TopSortTable, CFG
from .coarse import itercoarse, refine_conditions, iteritg
from .nederhof import Nederhof as ExactNederhof



def make_conditions(d, semiring):
    conditions = {r.lhs.label: semiring.as_real(r.weight) for r in d}
    return conditions


def make_batch_conditions(D, semiring):
    if len(D) == 1:
        d = D[0]
        conditions = {r.lhs.label: semiring.as_real(r.weight) for r in d}
    else:
        conditions = defaultdict(set)
        for d in D:
            [conditions[r.lhs.label].add(semiring.as_real(r.weight)) for r in d]
        conditions = {s: min(thetas) for s, thetas in conditions.items()}
    return conditions


def make_heuristic(args, cfg, semiring):
    if not args.heuristic:
        return None
    if args.heuristic == 'empdist':
        return heuristic.empdist(cfg, semiring, args.heuristic_empdist_alpha)
    elif args.heuristic == 'uniform':
        return heuristic.uniform(cfg, semiring, args.heuristic_uniform_params[0], args.heuristic_uniform_params[1])
    else:
        raise ValueError('Unknown heuristic')


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

    logging.info('Coarse grammar: terminals=%d nonterminals=%d rules=%d', coarse_grammar.n_terminals(), coarse_grammar.n_nonterminals(), len(coarse_grammar))
    parser = ExactNederhof([coarse_grammar], input_fsa,
              glue_grammars=[coarse_glue],
              semiring=semiring,
              make_symbol=make_recursive_symbol)
    forest = parser.do(root=Nonterminal(options.start), goal=Nonterminal(options.goal))
    if not forest:
        raise ValueError('The coarse grammar cannot parse this input')

    tsort = TopSortTable(forest)
    root = tsort.root()
    inside_nodes = robust_inside(forest, tsort, semiring, infinity=options.generations)
    d = list(sample(forest, root, semiring, Iv=inside_nodes, N=1))[0]
    t = make_nltk_tree(d)
    score = total_weight(d, semiring)
    #print('# exact=%f \n%s' % (semiring.as_real(semiring.divide(score, inside_nodes[root])), inlinetree(t)))

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

    logging.info('Coarse grammar: terminals=%d nonterminals=%d rules=%d', itg_grammar.n_terminals(), itg_grammar.n_nonterminals(), len(itg_grammar))
    parser = ExactNederhof([itg_grammar], input_fsa,
              glue_grammars=[itg_glue],
              semiring=semiring,
              make_symbol=make_recursive_symbol)
    forest = parser.do(root=Nonterminal(options.start), goal=Nonterminal(options.goal))
    if not forest:
        raise ValueError('The coarse grammar cannot parse this input')

    tsort = TopSortTable(forest)
    root = tsort.root()
    inside_nodes = robust_inside(forest, tsort, semiring, infinity=options.generations)
    d = list(sample(forest, root, semiring, Iv=inside_nodes, N=1))[0]
    t = make_nltk_tree(d)
    score = total_weight(d, semiring)
    #print('# exact=%f \n%s' % (semiring.as_real(semiring.divide(score, inside_nodes[root])), inlinetree(t)))

    return make_conditions(d, semiring)



def slice_sampling(input, grammars, glue_grammars, options):
    semiring=SumTimes
    # get a heuristic (for now it only uses the main grammar)
    heuristic = make_heuristic(options, grammars[0], semiring)
    # configure slice variables
    u = SliceVariables({}, a=options.beta_a[0], b=options.beta_b[0], heuristic=heuristic)
    samples = []
    goal = Nonterminal(options.goal)
    checkpoint = 10

    # TRYING SOMETHING HERE
    conditions = initialise_itg(input.fsa, grammars, glue_grammars, options)
    u.reset(conditions, a=options.beta_a[1], b=options.beta_b[1])
    #########################

    while len(samples) < options.samples + options.burn:
        parser = Nederhof(grammars, input.fsa,
                          glue_grammars=glue_grammars,
                          semiring=semiring,
                          slice_variables=u,
                          make_symbol=make_recursive_symbol)
        logging.debug('Parsing beta-a=%s beta-b=%s ...', u.a, u.b)
        forest = parser.do(root=Nonterminal(options.start), goal=goal)
        if not forest:
            logging.debug('NO PARSE FOUND')
            u.reset()  # reset the slice variables (keeping conditions unchanged if any)
            continue


        logging.debug('Topsort...')
        tsort = TopSortTable(forest)
        #logging.debug('Top symbol: %s', tsort.root())

        if options.count:
            Ic = robust_inside(forest, tsort, Count, omega=lambda e: Count.one, infinity=options.generations)
            logging.info('Done! Forest: %d edges, %d nodes and %d paths' % (len(forest), forest.n_nonterminals(), Ic[tsort.root()]))
        else:
            logging.debug('Done! Forest: %d edges, %d nodes' % (len(forest), forest.n_nonterminals()))

        logging.debug('Inside...')
        uniformdist = parser.reweight(forest)
        Iv = robust_inside(forest, tsort, semiring, omega=lambda e: uniformdist[e], infinity=options.generations)
        logging.debug('Sampling...')
        D = list(sample(forest, tsort.root(), semiring, Iv=Iv, N=options.batch, omega=lambda e: uniformdist[e]))
        assert D, 'The slice should never be empty'
        u.reset(make_batch_conditions(D, semiring), a=options.beta_a[1], b=options.beta_b[1])
        samples.extend(D)
        if len(samples) > checkpoint:
            logging.info('sampling... %d/%d', len(samples), options.samples + options.burn)
            checkpoint = len(samples) + 10

    count = Counter(samples[options.burn:])
    for d, n in count.most_common():
        t = make_nltk_tree(d)
        score = total_weight(d, SumTimes)
        print('# n=%d estimate=%f score=%f\n%s' % (n, float(n)/options.samples, score, inlinetree(t)))
    print()
