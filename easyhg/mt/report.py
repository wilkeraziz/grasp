"""
:Authors: - Wilker Aziz
"""

from easyhg.grammar.utils import make_nltk_tree, inlinetree
from easyhg.recipes import smart_wopen
from easyhg.grammar.projection import get_leaves


def save_viterbi(path, derivation, score):
    """

    :param path: where to save
    :param derivation: the best derivation
    :param score: its score
    """
    with smart_wopen(path) as out:
        y = get_leaves(derivation)
        tree = make_nltk_tree(derivation)
        print('# score\tyield\tderivation', file=out)
        print('{0}\t{1}\t{2}'.format(score,
                                     ' '.join(str(w.surface) for w in y),
                                     inlinetree(tree)),
              file=out)


def save_kbest(path, derivations, omega_d):
    """

    :param path: where to save
    :param derivations: sorted list of derivations
    :param omega: a function over derivations
    """
    with smart_wopen(path) as out:
        print('# score\tyield\tderivation', file=out)
        for d in derivations:
            y = get_leaves(d)
            tree = make_nltk_tree(d)
            score = omega_d(d)
            print('{0}\t{1}\t{2}'.format(score,
                                         ' '.join(str(w.surface) for w in y),
                                         inlinetree(tree)),
                  file=out)


def save_mc_derivations(path, samples, inside, omega_d, semiring):
    """

    :param path: where to save
    :param samples: sorted list of samples (obtained by group_by_projection)
    :param inside: inside at the root
    :param omega: a function over derivations
    :param semiring:
    """
    with smart_wopen(path) as out:
        total = sum(sample.count for sample in samples)
        print('# MC samples={0} inside={1}'.format(total, inside), file=out)
        print('# exact\testimate\tcount\tscore\tderivation', file=out)
        for sample in samples:
            score = omega_d(sample.projection)
            tree = make_nltk_tree(sample.projection)
            prob = semiring.as_real(semiring.divide(score, inside))
            print('{0}\t{1}\t{2}\t{3}\t{4}'.format(prob,
                                                   sample.count/total,
                                                   sample.count,
                                                   score,
                                                   inlinetree(tree)),
                  file=out)


def save_mc_yields(path, samples):
    """
    :param path: where to save
    :param samples: sorted list of samples (obtained by group_by_projection)
    """
    with smart_wopen(path) as out:
        total = sum(sample.count for sample in samples)
        print('# MC samples={0}'.format(total), file=out)
        print('# estimate\tcount\tderivations\tyield', file=out)
        for i, sample in enumerate(samples, 1):
            print('{0}\t{1}\t{2}\t{3}'.format(sample.count/total,
                                              sample.count,
                                              sample.derivations,
                                              ' '.join(str(t.surface) for t in sample.projection)),
                  file=out)


def save_mcmc_derivation(path, samples, omega_d):
    """

    :param path: where to save
    :param samples: sorted list of samples (obtained by group_by_projection)
    :param omega_d: a function over derivations
    """
    with smart_wopen(path) as out:
        total = sum(sample.count for sample in samples)
        print('# MCMC samples={0}'.format(total), file=out)
        print('# estimate\tcount\tscore\tderivation', file=out)
        for i, sample in enumerate(samples, 1):
            tree = make_nltk_tree(sample.projection)
            print('{0}\t{1}\t{2}\t{3}'.format(sample.count/total,  # estimate
                                              sample.count,
                                              omega_d(sample.projection),
                                              inlinetree(tree)),
                  file=out)


def save_mcmc_yields(path, samples):
    """

    :param path: where to save
    :param samples: sorted list of sampled (obtained by group_by_projection)
    """
    with smart_wopen(path) as out:
        total = sum(sample.count for sample in samples)
        print('# MCMC samples={0}\n# estimate\tcount\tderivations\tyield'.format(total), file=out)
        for i, sample in enumerate(samples, 1):
            print('{0}\t{1}\t{2}\t{3}'.format(float(sample.count)/total,
                                              sample.count,
                                              sample.derivations,
                                              ' '.join(str(t.surface) for t in sample.projection)),
                  file=out)


def save_markov_chain(path, markov_chain, omega_d=None):
    """

    :param path: where to save
    :param markov_chain: the original Markov chain
    :param omega_d: an optional function over derivations
    :return:
    """
    if omega_d is None:
        with smart_wopen(path) as out:
            for d in markov_chain:
                print(inlinetree(make_nltk_tree(d)), file=out)
    else:
        with smart_wopen(path) as out:
            for d in markov_chain:
                print('{0}\t{1}'.format(omega_d(d), inlinetree(make_nltk_tree(d))), file=out)