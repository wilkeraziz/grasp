"""
:Authors: - Wilker Aziz
"""

import logging
from collections import Counter
from easyhg.alg.exact import Earley, Nederhof, KBest
from easyhg.alg.exact.inference import robust_inside, sample, optimise, total_weight
from easyhg.recipes import smart_wopen

from .symbol import Nonterminal, make_span
from .semiring import SumTimes, MaxTimes, Count
from . import projection
from .cfg import TopSortTable
from .result import Result


class ParserState(object):

    @staticmethod
    def get_parser(input_fsa, main_grammars, glue_grammars, intersection='nederhof', semiring=SumTimes):
        """
        Construct a parsing algorithm.

        :param input_fsa: FSA
        :param main_grammars: list of CFG objects
        :param glue_grammars: list of CFG objects
        :return: Earley or Nederhof
        """
        if intersection == 'earley':
            return Earley(main_grammars, input_fsa,
                            glue_grammars=glue_grammars,
                            semiring=semiring)
        elif intersection == 'nederhof':
            return Nederhof(main_grammars, input_fsa,
                              glue_grammars=glue_grammars,
                              semiring=semiring)
        else:
            raise NotImplementedError("I don't know this intersection algorithm: %s" % intersection)

    def __init__(self, options, input_fsa, main_grammars, glue_grammars):
        self._parser = ParserState.get_parser(input_fsa, main_grammars, glue_grammars, options.intersection)
        self._forest = None
        self._tsort = None
        self._itables = {}
        self._options = options

    @property
    def options(self):
        return self._options

    @property
    def parser(self):
        return self._parser

    def forest(self):
        """
        Parse the input.
        :return: a forest (CFG)
        """
        if self._forest is None:
            # make a forest
            logging.info('Parsing...')
            forest = self.parser.do(root=Nonterminal(self._options.start), goal=Nonterminal(self._options.goal))
            self._forest = forest
        return self._forest

    def tsort(self, forest):
        """
        Topsort the forest.
        :param forest: CFG
        :return: TopSortTable
        """
        if self._tsort is None:
            logging.info('Top-sorting...')
            tsort = TopSortTable(forest)
            logging.info('Top symbol: %s', tsort.root())
            self._tsort = tsort
        return self._tsort

    def inside(self, forest, tsort, semiring, omega=None):
        """
        Compute inside weights for a given semiring.

        :param forest: CFG
        :param tsort: TopSortTable
        :param semiring: a Semiring class
        :param omega: a function of the edge weights (or None for edge weights)
        :return: inside weights (dict)
        """
        itable = self._itables.get(semiring, None)
        if itable is None:
            logging.info('Inside semiring=%s ...', str(semiring.__name__))
            if omega is None:
                itable = robust_inside(forest, tsort, semiring, infinity=self._options.generations)
            else:
                itable = robust_inside(forest, tsort, semiring, omega=omega, infinity=self._options.generations)
            logging.info('Inside goal-value=%f', itable[tsort.root()])
            self._itables[semiring] = itable
        return itable


def report_forest(state, outdir, uid):
    """
    Report information about the forest
    :param state: ParserState
    :return:
    """
    forest = state.forest()

    # count the number of derivations if necessary
    N = None
    if state.options.count:
        tsort = state.tsort(forest)
        itable = state.inside(forest, tsort, Count, omega=lambda e: Count.one)
        N = itable[tsort.root()]
        logging.info('Forest: edges=%d nodes=%d paths=%d', len(forest), forest.n_nonterminals(), N)
        with smart_wopen('{0}/count/{1}.gz'.format(outdir, uid)) as fo:
            print('terminals=%d nonterminals=%d edges=%d paths=%d' % (forest.n_terminals(),
                                                                      forest.n_nonterminals(),
                                                                      len(forest),
                                                                      N),
                  file=fo)
    else:
        logging.info('Forest: edges=%d nodes=%d', len(forest), forest.n_nonterminals())

    # write forest down as a CFG
    if state.options.forest:
        with smart_wopen('{0}/forest/{1}.gz'.format(outdir, uid)) as fo:
            print('# FOREST terminals=%d nonterminals=%d edges=%d' % (forest.n_terminals(),
                                                                      forest.n_nonterminals(),
                                                                      len(forest)),
                  file=fo)
            print(forest, file=fo)


def viterbi(state):
    """
    Viterbi derivation using the MaxTimes semiring.

    :param state: ParserState
    :return: a Result object containing the Viterbi derivation
    """
    forest = state.forest()
    tsort = state.tsort(forest)
    root = tsort.root()
    inside_nodes = state.inside(forest, tsort, MaxTimes)
    logging.info('Viterbi...')
    d = optimise(forest, root, MaxTimes, Iv=inside_nodes)
    logging.info('Done!')
    return Result([(d, 1, inside_nodes[root])])


def kbest(state):
    """
    K-best derivations using Huang and Chiang's algorithm.

    :param state: ParserState
    :return: a Result object containing the k-best derivations
    """
    root = make_span(Nonterminal(state.options.goal), None, None)
    logging.info('K-best...')
    kbestparser = KBest(state.forest(),
                        root,
                        state.options.kbest,
                        MaxTimes,
                        traversal=projection.string,
                        uniqueness=False).do()
    logging.info('Done!')
    R = Result()
    for k, d in enumerate(kbestparser.iterderivations()):
        score = total_weight(d, MaxTimes)
        R.append(d, 1, score)
    return R


def ancestral_sampling(state):
    """
    Samples N derivations by ancestral sampling (SumTimes semiring).

    :param state: ParserState
    :return: a Result object containing the samples grouped by derivation.
    """
    forest = state.forest()
    tsort = state.tsort(forest)
    root = tsort.root()
    inside_nodes = state.inside(forest, tsort, SumTimes)
    count = Counter(sample(forest, root, SumTimes, Iv=inside_nodes, N=state.options.samples))
    R = Result(Z=inside_nodes[root])
    for d, n in count.most_common():
        score = total_weight(d, SumTimes)
        R.append(d, n, score)
    return R


def exact(uid, input, grammars, glue_grammars, options, outdir):

    state = ParserState(options, input.fsa, grammars, glue_grammars)

    forest = state.forest()
    report_forest(state, outdir, uid)

    if not forest:
        logging.info('NO PARSE FOUND')
        return

    results = {}

    if options.viterbi:
        results['viterbi'] = viterbi(state)

    if options.kbest > 0:
        results['kbest'] = kbest(state)

    if options.samples > 0:
        results['ancestral'] = ancestral_sampling(state)

    logging.info('Finished!')
    return results



