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


class Parser(object):

    def __init__(self, grammars, input, options, glue_grammars=[]):
        self._grammars = grammars
        self._glue_grammars = glue_grammars
        self._input = input
        self._options = options

        self._start_symbol = Nonterminal(options.start)
        self._goal_symbol = Nonterminal(options.goal)

        # lazy properties
        self._forest = None
        self._tsort = None
        self._inside = {}

    @property
    def grammars(self):
        return self._grammars

    @property
    def glue_grammars(self):
        return self._glue_grammars

    @property
    def input(self):
        return self._input

    @property
    def options(self):
        return self._options

    @property
    def root(self):
        return self._goal_symbol

    def get_parser(self, algorithm):
        grammars = self.grammars
        fsa = self.input.fsa
        glue = self.glue_grammars
        semiring = SumTimes
        make_symbol = make_flat_symbol
        if algorithm == 'earley':
            parser = Earley(grammars, fsa, glue_grammars=glue, semiring=semiring, make_symbol=make_symbol)
        elif algorithm == 'nederhof':
            parser = Nederhof(grammars, fsa, glue_grammars=glue, semiring=semiring, make_symbol=make_symbol)
        else:
            raise NotImplementedError("I don't know this intersection algorithm: %s" % algorithm)
        return parser

    def forest(self):
        if self._forest is None:
            # get a parser for a given parsing algorithm
            parser = self.get_parser(self.options.intersection)
            # make a forest
            logging.info('Parsing...')
            forest = parser.do(root=self._start_symbol, goal=self._goal_symbol)
            if not forest:
                raise ValueError('No parse found')

            self._forest = forest

        return self._forest

    def tsort(self):
        if self._tsort is None:
            forest = self.forest()
            logging.info('Top-sorting...')
            tsort = TopSortTable(forest)
            logging.info('Top symbol: %s', tsort.root())
            self._tsort = tsort
        return self._tsort

    def inside(self, semiring):
        itable = self._inside.get(semiring, None)
        if itable is None:
            forest = self.forest()
            tsort = self.tsort()
            logging.info('Inside semiring=%s ...', str(semiring.__name__))
            itable = robust_inside(forest, tsort, semiring, infinity=self.options.generations)
            logging.info('Inside goal-value=%f', itable[self.root])
            self._inside[semiring] = itable
        return itable

    def viterbi(self):
        forest = self.forest()
        tsort = self.tsort()
        inside_nodes = self.inside(MaxTimes)
        logging.info('Viterbi...')
        d = optimise(forest, self.root, MaxTimes, Iv=inside_nodes)
        t = make_nltk_tree(d)
        print('# VITERBI')
        print('# k={0} score={1}\n{2}'.format(1, inside_nodes[self.root], inlinetree(t)))
        print()

    def kbest(self):
        forest = self.forest()
        logging.info('K-best...')
        kbestparser = KBest(forest,
                            self.root,
                            self.options.kbest,
                            MaxTimes,
                            traversal=projection.string,
                            uniqueness=False).do()
        logging.info('Done!')
        print('# K-BEST: size=%d' % self.options.kbest)
        for k, d in enumerate(kbestparser.iterderivations()):
            t = make_nltk_tree(d)
            print('# k={0} score={1}\n{2}'.format(k + 1, total_weight(d, MaxTimes), inlinetree(t)))
        print()

    def ancestral_sampling(self):
        forest = self.forest()
        inside_nodes = self.inside(SumTimes)
        count = Counter(sample(forest, self.root, SumTimes, Iv=inside_nodes, N=self.options.samples))
        n_samples = self.options.samples
        print('# SAMPLE: size=%d Z=%s' % (n_samples, inside_nodes[self.root]))
        for d, n in count.most_common():
            t = make_nltk_tree(d)
            score = total_weight(d, SumTimes)
            p = SumTimes.divide(score, inside_nodes[self.root])
            print('# n={0} estimate={1} exact={2} score={3}\n{4}'.format(n, float(n)/n_samples, SumTimes.as_real(p), score, inlinetree(t)))
        print()

    def count(self, forest, tsort):
        logging.info('Inside semiring=%s ...', str(Count.__name__))
        Ic = robust_inside(forest, tsort, Count, omega=lambda e: Count.one, infinity=self.options.generations)
        logging.info('Forest: edges=%d nodes=%d paths=%d', len(forest), forest.n_nonterminals(), Ic[self.root])
        print('# FOREST edges=%d nodes=%d paths=%d' % (len(forest), forest.n_nonterminals(), Ic[self.root]))

    def exact(self):
        try:
            forest = self.forest()
        except ValueError:
            logging.error('NO PARSE FOUND')
            return False

        if self.options.forest:  # dump forest
            print(forest)
            print()

        if self.options.count:  # compute derivation counts
            self.count(self.forest(), self.tsort())
        else:
            logging.info('Forest: edges=%d nodes=%d goal=%s', len(forest), forest.n_nonterminals(), self.root)
            print('# FOREST: edges=%d nodes=%d goal=%s' % (len(forest), forest.n_nonterminals(), self.root))

        if self.options.viterbi:
            self.viterbi()

        if self.options.kbest > 0:
            self.kbest()

        if self.options.samples > 0:
            self.ancestral_sampling()

        logging.info('Finished!')

    def slice_sampling(self):
        slice_sampling(self.grammars, self.glue_grammars, self.input, self.options)

    def do(self):
        if self.options.framework == 'exact':
            self.exact()
        elif self.options.framework == 'slice':
            self.slice_sampling()
        else:
            raise NotImplementedError('I do not yet know how to perform inference in this framework: %s' % self.options.framework)


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
        parser = Parser(grammars, sentence, args, glue_grammars=glue_grammars)
        parser.do()


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
