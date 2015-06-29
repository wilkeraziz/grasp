"""
This module is an interface for parsing as intersection.
One can choose from all available implementations.

:Authors: - Wilker Aziz
"""

import logging
from collections import Counter
from .cmdline import argparser
from .sentence import make_sentence
from .symbol import Terminal, Nonterminal, make_flat_symbol
from .earley import Earley
from .nederhof import Nederhof
from .utils import make_nltk_tree, inlinetree
from .semiring import Prob, SumTimes, MaxTimes, Count
from .inference import inside, robust_inside, sample, optimise, total_weight
from .kbest import KBest
from . import projection
from .reader import load_grammar
from .cfg import CFG, TopSortTable
from .rule import get_oov_cfg_productions
from .slicesampling import slice_sampling
import sys
from itertools import chain


class ParserState(object):

    def __init__(self, options):
        self._parser = None
        self._forest = None
        self._tsort = None
        self._itables = {}
        self._options = options

    @property
    def options(self):
        return self._options

    def parser(self, input_fsa, main_grammars, glue_grammars):
        """

        :param input_fsa: FSA
        :param main_grammars: list of CFG objects
        :param glue_grammars: list of CFG objects
        :return: Earley or Nederhof
        """
        if self._parser is None:
            semiring = SumTimes
            make_symbol = make_flat_symbol
            if self._options.intersection == 'earley':
                self._parser = Earley(main_grammars, input_fsa,
                                glue_grammars=glue_grammars,
                                semiring=semiring,
                                make_symbol=make_symbol)
            elif self._options.intersection == 'nederhof':
                self._parser = Nederhof(main_grammars, input_fsa,
                                  glue_grammars=glue_grammars,
                                  semiring=semiring,
                                  make_symbol=make_symbol)
            else:
                raise NotImplementedError("I don't know this intersection algorithm: %s" % self._options.intersection)
        return self._parser

    def forest(self, parser):
        if self._forest is None:
            # make a forest
            logging.info('Parsing...')
            forest = parser.do(root=Nonterminal(self._options.start), goal=Nonterminal(self._options.goal))
            self._forest = forest
        return self._forest

    def tsort(self, forest):
        if self._tsort is None:
            logging.info('Top-sorting...')
            tsort = TopSortTable(forest)
            logging.info('Top symbol: %s', tsort.root())
            self._tsort = tsort
        return self._tsort

    def inside(self, forest, tsort, semiring, omega=None):
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


def report_info(cfg, args):
    if args.report_top or args.report_tsort or args.report_cycles:
        tsort = TopSortTable(cfg)
        if args.report_top:
            logging.info('TOP symbols={0} buckets={1}'.format(tsort.n_top_symbols(), tsort.n_top_buckets()))
            for bucket in tsort.itertopbuckets():
                print(' '.join(str(s) for s in bucket))
            sys.exit(0)
        if args.report_tsort:
            logging.info('TOPSORT levels=%d' % tsort.n_levels())
            print(str(tsort))
            sys.exit(0)
        if args.report_cycles:
            loopy = []
            for i, level in enumerate(tsort.iterlevels(skip=1)):
                loopy.append(set())
                for bucket in filter(lambda b: len(b) > 1, level):
                    loopy[-1].add(bucket)
            logging.info('CYCLES symbols=%d cycles=%d' % (tsort.n_loopy_symbols(), tsort.n_cycles()))
            for i, buckets in enumerate(loopy, 1):
                if not buckets:
                    continue
                print('level=%d' % i)
                for bucket in buckets:
                    print(' bucket-size=%d' % len(bucket))
                    print('\n'.join('  {0}'.format(s) for s in bucket))
                print()
            sys.exit(0)


def report_forest(forest, state):
    if state.options.count:
        tsort = state.tsort(forest)
        itable = state.inside(forest, tsort, Count, omega=lambda e: Count.one)
        logging.info('Forest: edges=%d nodes=%d paths=%d', len(forest), forest.n_nonterminals(), itable[tsort.root()])
        print('# FOREST edges=%d nodes=%d paths=%d' % (len(forest), forest.n_nonterminals(), itable[tsort.root()]))
    else:
        logging.info('Forest: edges=%d nodes=%d', len(forest), forest.n_nonterminals())

    if state.options.forest:
        print('# FOREST terminals=%d nonterminals=%d rules=%d' % (forest.n_terminals(), forest.n_nonterminals(), len(forest)))
        print(forest)  # TODO: write to a file
        print()


def viterbi(forest, state):
    tsort = state.tsort(forest)
    root = tsort.root()
    inside_nodes = state.inside(forest, tsort, MaxTimes)
    logging.info('Viterbi...')
    d = optimise(forest, root, MaxTimes, Iv=inside_nodes)
    t = make_nltk_tree(d)
    print('# VITERBI')
    print('# k={0} score={1}\n{2}'.format(1, inside_nodes[root], inlinetree(t)))
    print()


def kbest(forest, state):
    root = Nonterminal(state.options.goal)
    logging.info('K-best...')
    kbestparser = KBest(forest,
                        root,
                        state.options.kbest,
                        MaxTimes,
                        traversal=projection.string,
                        uniqueness=False).do()
    logging.info('Done!')
    print('# K-BEST: size=%d' % state.options.kbest)
    for k, d in enumerate(kbestparser.iterderivations()):
        t = make_nltk_tree(d)
        print('# k={0} score={1}\n{2}'.format(k + 1, total_weight(d, MaxTimes), inlinetree(t)))
    print()


def ancestral_sampling(forest, state):
    tsort = state.tsort(forest)
    root = tsort.root()
    inside_nodes = state.inside(forest, tsort, SumTimes)
    count = Counter(sample(forest, root, SumTimes, Iv=inside_nodes, N=state.options.samples))
    n_samples = state.options.samples
    print('# SAMPLE: size=%d Z=%s' % (n_samples, inside_nodes[root]))
    for d, n in count.most_common():
        t = make_nltk_tree(d)
        score = total_weight(d, SumTimes)
        p = SumTimes.divide(score, inside_nodes[root])
        print('# n={0} estimate={1} exact={2} score={3}\n{4}'.format(n, float(n)/n_samples, SumTimes.as_real(p), score, inlinetree(t)))
    print()


def exact(input, grammars, glue_grammars, options):

    state = ParserState(options)
    parser = state.parser(input.fsa, grammars, glue_grammars)
    forest = state.forest(parser)

    if not forest:
        logging.info('NO PARSE FOUND')
        return

    report_forest(forest, state)

    if options.viterbi:
        viterbi(forest, state)

    if options.kbest > 0:
        kbest(forest, state)

    if options.samples > 0:
        ancestral_sampling(forest, state)

    logging.info('Finished!')


def do(input, grammars, glue_grammars, options):
    if options.framework == 'exact':
        exact(input, grammars, glue_grammars, options)
    elif options.framework == 'slice':
        slice_sampling(input, grammars, glue_grammars, options)
    else:
        raise NotImplementedError('I do not yet know how to perform inference in this framework: %s' % self.options.framework)


def core(args):
    semiring = SumTimes

    # Load main grammars
    logging.info('Loading main grammar...')
    cfg = load_grammar(args.grammar, args.grammarfmt, args.log)
    logging.info('Main grammar: terminals=%d nonterminals=%d productions=%d', cfg.n_terminals(), cfg.n_nonterminals(), len(cfg))

    # Load additional grammars
    main_grammars = [cfg]
    if args.extra_grammar:
        for grammar_path in args.extra_grammar:
            logging.info('Loading additional grammar: %s', grammar_path)
            grammar = load_grammar(grammar_path, args.grammarfmt, args.log)
            logging.info('Additional grammar: terminals=%d nonterminals=%d productions=%d', grammar.n_terminals(), grammar.n_nonterminals(), len(grammar))
            main_grammars.append(grammar)

    # Load glue grammars
    glue_grammars = []
    if args.glue_grammar:
        for glue_path in args.glue_grammar:
            logging.info('Loading glue grammar: %s', glue_path)
            glue = load_grammar(glue_path, args.grammarfmt, args.log)
            logging.info('Glue grammar: terminals=%d nonterminals=%d productions=%d', glue.n_terminals(), glue.n_nonterminals(), len(glue))
            glue_grammars.append(glue)

    # Report information about the main grammar
    report_info(cfg, args)

    # Make surface lexicon
    surface_lexicon = set()
    for grammar in chain(main_grammars, glue_grammars):
        surface_lexicon.update(t.surface for t in grammar.iterterminals())

    # Parse sentence by sentence
    for input_str in args.input:
        # get an input automaton
        sentence = make_sentence(input_str, semiring, surface_lexicon, args.unkmodel)
        grammars = list(main_grammars)

        if args.unkmodel == 'passthrough':
            grammars.append(CFG(get_oov_cfg_productions(sentence.oovs, args.unklhs, semiring.one)))

        logging.info('Parsing %d words: %s', len(sentence), sentence)
        do(sentence, grammars, glue_grammars, args)


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
