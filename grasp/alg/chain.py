"""
:Authors: - Wilker Aziz
"""
import itertools
from collections import Counter, defaultdict
from types import SimpleNamespace
import numpy as np


def apply_filters(markov_chain, burn=0, lag=1, resample=0):
    """
    Filter a Markov chain using a few known tricks to reduce autocorrelation.

    :param markov_chain: the chain of samples (or batches of samples)
    :param burn: discard a number of states from the beginning of the chain
    :param lag: retains states constantly spaced
    :param resample: resamples with replacement a number of states
    :return: a stream of states
    """

    iterable = iter(markov_chain)
    total = len(markov_chain)

    # burn from the left end
    if burn > 0:
        iterable = itertools.islice(iterable, burn, len(markov_chain))
        total -= burn

    # evenly spaced samples
    if lag > 1 and total > 0:
        iterable = itertools.islice(iterable, lag - 1, total, lag)
        total //= lag

    filtered_chain = list(iterable)

    # resampling step
    if resample > 0:
        filtered_chain = [filtered_chain[i] for i in np.random.randint(0, len(filtered_chain), resample)]

    return filtered_chain


def apply_batch_filters(batches, resample=0):
    """
    After applying the usual filters to a Markov chain whose states are batches of samples,
     rather than single samples, we flatten these batches, making a stream of samples.
    In this case, one might want to resample from the batch instead of leaving it whole.

    :param batches: a sequence of batches of samples
    :param resample: resamples with replacement a number of elements from each batch
    :return: a stream of samples
    """
    if resample > 0:
        samples = []
        for batch in batches:
            choices = np.random.randint(0, len(batch), resample)
            samples.extend(batch[i] for i in choices)
    else:
        samples = list(itertools.chain(*batches))
    return samples


def group_by_identity(derivations):
    counts = Counter(derivations)
    output = []
    for d, n in counts.most_common():
        output.append(SimpleNamespace(key=d, count=n))
    return output


def group_by_projection(samples, get_projection):
    p2d = defaultdict(list)
    counts = Counter()
    for d in samples:
        y = get_projection(d)
        counts[y] += 1
        p2d[y].append(d)
    output = []
    for y, n in counts.most_common():
        output.append(SimpleNamespace(key=y, count=n, values=p2d[y]))
    return output