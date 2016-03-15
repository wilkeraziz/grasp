"""
This module contains functions that save the output of different decoding strategies.

:Authors: - Wilker Aziz
"""

from grasp.recipes import smart_wopen
import grasp.semiring as semiring
from grasp.cfg.projection import DerivationYield


def save_viterbi(path, viterbi, get_projection, derivation2str=DerivationYield.derivation):
    """

    :param path: where to save
    :param viterbi: the best derivation
    :param omega_d: a function over derivations
    :param get_projection: a function which returns a projection of a derivation
    """
    with smart_wopen(path) as out:
        print('# score\tyield\tderivation', file=out)
        rules = viterbi.derivation.rules()
        print('{0}\t{1}\t{2}'.format(viterbi.value,
                                     get_projection(rules),
                                     derivation2str(rules)),
              file=out)


def save_kbest(path, derivations, get_projection, derivation2str=DerivationYield.derivation):
    """

    :param path: where to save
    :param derivations: sorted list of derivations
    :param omega_d: a function over derivations
    :param get_projection: a function which returns a projection of a derivation
    """
    with smart_wopen(path) as out:
        print('# score\tyield\tderivation', file=out)
        for d in derivations:
            print('{0}\t{1}\t{2}'.format(d.value,
                                         get_projection(d),
                                         derivation2str(d)),
                  file=out)


def save_mc_derivations(path, samples, inside, valuefunc, derivation2str, semiring=semiring.inside):
    """

    :param path: where to save
    :param samples: sorted list of samples (obtained by group_by_identity)
    :param inside: inside at the root
    :param valuefunc: function to compute the value of the derivation
    :param derivation2str: how to obtain a string from a derivation
    :param semiring: semiring used to normalise probabilities
    """
    with smart_wopen(path) as out:
        total = sum(sample.count for sample in samples)
        print('# MC samples={0} inside={1} semiring={2}'.format(total, inside, semiring), file=out)
        print('# exact\testimate\tcount\tscore\tderivation', file=out)
        for sample in samples:
            score = valuefunc(sample.derivation)
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


def save_mcmc_derivations(path, groups, valuefunc, derivation2str, compfunc=None):  # =lambda d: DerivationYield.derivation(d.rules())
    """

    :param path: where to save
    :param samples: sorted list of samples (obtained by group_by_identity)
    :param valuefunc: compute the value of a derivation
    :param omega_d: a function over derivations
    """
    with smart_wopen(path) as out:
        total = sum(group.count for group in groups)
        print('# MCMC samples={0}'.format(total), file=out)
        if compfunc is not None:
            print('# estimate\tcount\tscore\tderivation\tfeatures', file=out)
            for i, group in enumerate(groups, 1):
                sample = group.key
                print('{0}\t{1}\t{2}\t{3}\t{4}'.format(group.count / total,  # estimate
                                                       group.count,
                                                       valuefunc(sample),
                                                       derivation2str(sample),
                                                       compfunc(sample)),
                      file=out)
        else:
            print('# estimate\tcount\tscore\tderivation', file=out)
            for i, group in enumerate(groups, 1):
                sample = group.key
                print('{0}\t{1}\t{2}\t{3}'.format(group.count/total,  # estimate
                                                  group.count,
                                                  valuefunc(sample),
                                                  derivation2str(sample)),
                      file=out)


def save_mcmc_yields(path, groups):
    """

    :param path: where to save
    :param groups: sorted list of sampled (obtained by group_by_projection)
    """
    with smart_wopen(path) as out:
        total = sum(group.count for group in groups)
        print('# MCMC samples={0}\n# estimate\tcount\tderivations\tyield'.format(total), file=out)
        for i, group in enumerate(groups, 1):
            sample = group
            print('{0}\t{1}\t{2}\t{3}'.format(float(group.count)/total,
                                              group.count,
                                              len(set(group.values)),
                                              group.key),
                  file=out)


def save_markov_chain(path, markov_chain, derivation2str, valuefunc=None, compfunc=None, flat=True):
    """

    :param path: where to save
    :param markov_chain: the original Markov chain
    :param valuefunc: an optional function over derivations that returns a score
    :param compfunc: an optional function over derivations that return feature components
    :param flat: whether the Markov chain is flat (each state represents a single derivation) or not,
        in which case each state is a sequence of derivations.
    """
    if flat:
        with smart_wopen(path) as out:
            for d in markov_chain:
                fields = []
                if valuefunc is not None:
                    fields.append(valuefunc(d))
                fields.append(derivation2str(d))
                if compfunc is not None:
                    fields.append(compfunc(d))
                print('\t'.join(str(x) for x in fields), file=out)
    else:
        with smart_wopen(path) as out:
            for i, batch in enumerate(markov_chain):
                for d in batch:
                    fields = [i]
                    if valuefunc is not None:
                        fields.append(valuefunc(d))
                    fields.append(derivation2str(d))
                    if compfunc is not None:
                        fields.append(compfunc(d))
                    print('\t'.join(str(x) for x in fields), file=out)
