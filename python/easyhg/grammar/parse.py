"""
This module is an interface for parsing as intersection.
One can choose from all available implementations.

:Authors: - Wilker Aziz
"""

import logging
from collections import Counter

from .cmdline import argparser
from .sentence import make_sentence
from .symbol import Nonterminal, make_flat_symbol
from .earley import Earley
#from .cky import CKY
from .nederhof import Nederhof
from .utils import make_nltk_tree, inlinetree
from .semiring import Prob, SumTimes, MaxTimes, Count
from .inference import inside, sample, optimise, total_weight
from .kbest import KBest
from . import projection
from .reader import load_grammar


def get_parser(cfg, fsa, semiring, make_symbol, algorithm):
    if algorithm == 'earley':
        parser = Earley(cfg, fsa, semiring=semiring, make_symbol=make_symbol)
    elif algorithm == 'nederhof':
        parser = Nederhof(cfg, fsa, semiring=semiring, make_symbol=make_symbol)
    elif algorithm == 'cky':
        raise ValueError('Temporarily unavailable: %s' % algorithm)
        #parser = CKY(cfg, fsa, semiring=semiring, make_symbol=make_symbol)
    else:
        raise NotImplementedError("I don't know this intersection algorithm: %s" % algorithm)
    return parser


def ancestral_sampling(forest, topsorted, args, semiring=SumTimes):
    logging.info('Inside...')
    Iv = inside(forest, topsorted, semiring)
    logging.info('Done! Sampling...')
    count = Counter(sample(forest, topsorted[-1], semiring, Iv=Iv, N=args.samples))
    print('# SAMPLE: size=%d' % args.samples)
    for d, n in reversed(count.most_common()):
        t = make_nltk_tree(d)
        p = total_weight(d, SumTimes, Iv[topsorted[-1]])
        print('# n={0} emp={1} exact={2}\n{3}'.format(n, float(n)/args.samples, semiring.as_real(p), inlinetree(t)))
    print()


def viterbi(forest, topsorted, args, semiring=MaxTimes):
    logging.info('Inside...')
    Iv = inside(forest, topsorted, semiring)
    logging.info('Done! Viterbi...')
    d = optimise(forest, topsorted[-1], semiring, Iv=Iv)
    t = make_nltk_tree(d)
    print('# VITERBI')
    print('# k={0} score={1}\n{2}'.format(1, Iv[topsorted[-1]], inlinetree(t)))
    print()


def kbest(forest, topsorted, args, semiring=MaxTimes):
    logging.info('K-best...')
    kbest = KBest(forest, topsorted[-1], args.kbest, semiring, traversal=projection.string, uniqueness=False).do()
    print('# K-BEST: size=%d' % args.kbest)
    for k, d in enumerate(kbest.iterderivations()):
        t = make_nltk_tree(d)
        print('# k={0} score={1}\n{2}'.format(k + 1, total_weight(d, MaxTimes), inlinetree(t)))
    print()


def parse(cfg, sentence, semiring, args):
    # get a parser
    parser = get_parser(cfg, sentence.fsa, semiring, make_flat_symbol, args.intersection)
    # make a forest
    logging.info('Parsing...')
    forest = parser.do(root=Nonterminal(args.start), goal=Nonterminal(args.goal))
    if not forest:
        logging.error('NO PARSE FOUND')
        return False

    logging.info('Top-sorting...')
    #topsorted = list(chain(*topsort_cfg(forest)))
    topsorted = forest.topsort()

    logging.info('Topsorted=%d symbols=%d', len(topsorted), forest.n_symbols())
    print(topsorted[-1])
    #S = set(forest.iternonterminals())
    #S.update(forest.iterterminals())
    #for s in S.difference(set(topsorted)):
    #    print s


    if args.count:
        logging.info('Counting...')
        Ic = inside(forest, topsorted, Count, omega=lambda e: 1)
        logging.info('Forest: edges=%d nodes=%d paths=%d', len(forest), forest.n_nonterminals(), Ic[topsorted[-1]])
        print('# FOREST: edges=%d nodes=%d paths=%d' % (len(forest), forest.n_nonterminals(), Ic[topsorted[-1]]))
    else:
        logging.info('Forest: edges=%d nodes=%d', len(forest), forest.n_nonterminals())
        print('# FOREST: edges=%d nodes=%d' % (len(forest), forest.n_nonterminals()))

    if args.forest:
        print(forest)
        print()

    if args.samples > 0:
        ancestral_sampling(forest, topsorted, args)

    if args.kbest == 1:
        viterbi(forest, topsorted, args)

    if args.kbest > 1:
        kbest(forest, topsorted, args)

    logging.info('Finished!')


def fullparser(args):
    semiring = SumTimes

    logging.info('Loading grammar...')
    cfg = load_grammar(args.grammar, args.grammarfmt, args.log)
    logging.info('Done: rules=%d', len(cfg))

    for input_str in args.input:
        # get an input automaton
        sentence, extra_rules = make_sentence(input_str, semiring, cfg.lexicon, args.unkmodel, args.default_symbol)
        cfg.update(extra_rules)
        parse(cfg, sentence, semiring, args)


def configure():
    args = argparser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

    if args.sampler == 'ancestral':
        parser = fullparser
    else:
        from . import slicesampling
        parser = slicesampling.main

    return args, parser


def main():
    args, parser = configure()

    if args.profile:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
        parser(args)
        pr.disable()
        pr.dump_stats(args.profile)
    else:
        parser(args)




if __name__ == '__main__':
    main()
