"""
:Authors: - Wilker Aziz
"""
from .utils import smart_wopen, make_nltk_tree, inlinetree
from .semiring import SumTimes
from .projection import get_leaves
from easyhg.alg.exact.inference import total_weight


def save_mc(path, result):
    with smart_wopen(path) as out:
        Z = result.Z
        N = result.count()
        print('# MC samples={0} inside={1}'.format(N, Z), file=out)
        for i, (d, n, score) in enumerate(result, 1):
            t = make_nltk_tree(d)
            p = SumTimes.divide(score, Z)
            print('# k={0} n={1} estimate={2} exact={3} score={4}\n{5}'.format(i,
                                                                               n,
                                                                               float(n)/N,
                                                                               SumTimes.as_real(p),
                                                                               score,
                                                                               inlinetree(t)),
                  file=out)


def save_mcmc(path, result):
    with smart_wopen(path) as out:
        Z = result.estimate(SumTimes.plus)
        N = result.count()
        print('# MCMC samples={0} inside-estimate={1}'.format(N, Z), file=out)
        for i, (d, n, score) in enumerate(result, 1):
            t = make_nltk_tree(d)
            p = SumTimes.divide(score, Z)
            print('# k={0} n={1} estimate={2} normalized-score={3} score={4}\n{5}'.format(i,
                                                                               n,
                                                                               float(n)/N,
                                                                               SumTimes.as_real(p),
                                                                               score,
                                                                               inlinetree(t)),
                  file=out)


def save_kbest(path, result):
    with smart_wopen(path) as out:
        Z = result.estimate(SumTimes.plus)
        print('# KBEST size={0} inside-estimate={1}'.format(len(result), Z), file=out)
        for i, (d, n, score) in enumerate(result, 1):
            t = make_nltk_tree(d)
            p = SumTimes.divide(score, Z)
            print('# k={0} score={1} normalized-score={2}\n{3}'.format(i,
                                                                       score,
                                                                       SumTimes.as_real(p),
                                                                       inlinetree(t)),
                  file=out)


def save_viterbi(path, result):
    with smart_wopen(path) as out:
        d, n, score = result[0]
        t = make_nltk_tree(d)
        print('# score={0}\n{1}'.format(score, inlinetree(t)), file=out)


def save_sample_history(path, samples_by_iteration):
    with smart_wopen(path) as out:
        for i, samples in enumerate(samples_by_iteration, 1):
            print('# i={0} n={1}'.format(i, len(samples)), file=out)
            for d in samples:
                score = total_weight(d, SumTimes)
                t = make_nltk_tree(d)
                print('{0}\t{1}'.format(score, inlinetree(t)), file=out)

def save_flat_history(path, history):
    with smart_wopen(path) as out:
        for i, d in enumerate(history, 1):
            score = total_weight(d, SumTimes)
            t = make_nltk_tree(d)
            print('{0}\t{1}'.format(score, inlinetree(t)), file=out)