"""
This module contains functions that save the output of different decoding strategies.

:Authors: - Wilker Aziz
"""

from grasp.recipes import smart_wopen
from grasp.semiring import SumTimes
from grasp.cfg.projection import DerivationYield


def save_viterbi(path, derivation, omega_d, get_projection, derivation2str=DerivationYield.derivation):
    """

    :param path: where to save
    :param derivation: the best derivation
    :param omega_d: a function over derivations
    :param get_projection: a function which returns a projection of a derivation
    """
    with smart_wopen(path) as out:
        print('# score\tyield\tderivation', file=out)
        print('{0}\t{1}\t{2}'.format(omega_d(derivation),
                                     get_projection(derivation),
                                     derivation2str(derivation)),
              file=out)


def save_kbest(path, derivations, omega_d, get_projection, derivation2str=DerivationYield.derivation):
    """

    :param path: where to save
    :param derivations: sorted list of derivations
    :param omega_d: a function over derivations
    :param get_projection: a function which returns a projection of a derivation
    """
    with smart_wopen(path) as out:
        print('# score\tyield\tderivation', file=out)
        for d in derivations:
            print('{0}\t{1}\t{2}'.format(omega_d(d),
                                         get_projection(d),
                                         derivation2str(d)),
                  file=out)


def save_mc_derivations(path, samples, inside, omega_d, derivation2str=DerivationYield.derivation, semiring=SumTimes):
    """

    :param path: where to save
    :param samples: sorted list of samples (obtained by group_by_projection)
    :param inside: inside at the root
    :param omega_d: a function over derivations
    """
    with smart_wopen(path) as out:
        total = sum(sample.count for sample in samples)
        print('# MC samples={0} inside={1}'.format(total, inside), file=out)
        print('# exact\testimate\tcount\tscore\tderivation', file=out)
        for sample in samples:
            score = omega_d(sample.derivation)
            prob = semiring.as_real(semiring.divide(score, inside))
            print('{0}\t{1}\t{2}\t{3}\t{4}'.format(prob,
                                                   sample.count/total,
                                                   sample.count,
                                                   score,
                                                   derivation2str(sample.derivation)),
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
                                              len(sample.derivations),
                                              sample.projection),
                  file=out)


def save_mcmc_derivation(path, samples, omega_d, derivation2str=DerivationYield.derivation):
    """

    :param path: where to save
    :param samples: sorted list of samples (obtained by group_by_identity)
    :param omega_d: a function over derivations
    """
    with smart_wopen(path) as out:
        total = sum(sample.count for sample in samples)
        print('# MCMC samples={0}'.format(total), file=out)
        print('# estimate\tcount\tscore\tderivation', file=out)
        for i, sample in enumerate(samples, 1):
            print('{0}\t{1}\t{2}\t{3}'.format(sample.count/total,  # estimate
                                              sample.count,
                                              omega_d(sample.derivation),
                                              derivation2str(sample.derivation)),
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
                                              len(sample.derivations),
                                              sample.projection),
                  file=out)


def save_markov_chain(path, markov_chain, omega_d=None, flat=True, derivation2str=DerivationYield.derivation):
    """

    :param path: where to save
    :param markov_chain: the original Markov chain
    :param omega_d: an optional function over derivations
    :param flat: whether the Markov chain is flat (each state represents a single derivation) or not,
        in which case each state is a sequence of derivations.
    """
    if flat:
        if omega_d is None:
            with smart_wopen(path) as out:
                for d in markov_chain:
                    print(derivation2str(d), file=out)
        else:
            with smart_wopen(path) as out:
                for d in markov_chain:
                    print('{0}\t{1}'.format(omega_d(d), derivation2str(d)), file=out)
    else:
        if omega_d is None:
            with smart_wopen(path) as out:
                for i, batch in enumerate(markov_chain):
                    for d in batch:
                        print('{0}\t{1}'.format(i, derivation2str(d)), file=out)
        else:
            with smart_wopen(path) as out:
                for i, batch in enumerate(markov_chain):
                    for d in batch:
                        print('{0}\t{1}\t{2}'.format(i, omega_d(d), derivation2str(d)), file=out)