"""
This module implements BLEU for decoding and for training with chisel.

In decoding,
    we must be able to compute BLEU between two candidate translations (where once is seen as reference)

In training,
    we must be able to compute BLEU between a candidate translation and a set of references


>>> factory = NGramFactory(4)
>>> f1 = make_features('the black dog barked at the black cat'.split(), factory)
>>> f1 # doctest: +NORMALIZE_WHITESPACE
    [defaultdict(<type 'float'>, {}),
     defaultdict(<type 'float'>, {0: 2.0, 1: 2.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}),
     defaultdict(<type 'float'>, {0: 2.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}),
     defaultdict(<type 'float'>, {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}),
     defaultdict(<type 'float'>, {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0})]
>>> len(factory.ngrams(3))
6
>>> dict2array(f1[3], 6) # doctest: +NORMALIZE_WHITESPACE
array([ 1.,  1.,  1.,  1.,  1.,  1.])
>>> f2 = make_features('black dog barks at black cat'.split(), factory)
>>> f2 # doctest: +NORMALIZE_WHITESPACE
[defaultdict(<type 'float'>, {}),
 defaultdict(<type 'float'>, {1: 2.0, 2: 1.0, 4: 1.0, 5: 1.0, 6: 1.0}),
 defaultdict(<type 'float'>, {8: 1.0, 1: 1.0, 5: 1.0, 6: 1.0, 7: 1.0}),
 defaultdict(<type 'float'>, {8: 1.0, 9: 1.0, 6: 1.0, 7: 1.0}),
 defaultdict(<type 'float'>, {5: 1.0, 6: 1.0, 7: 1.0})]
>>> len(factory.ngrams(3))
10
>>> dict2array(f1[3], 10) # doctest: +NORMALIZE_WHITESPACE
array([ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.])
>>> dict2array(f2[3], 10) # doctest: +NORMALIZE_WHITESPACE
array([ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.])
>>> arrays = make_arrays(f1, factory)
>>> arrays # doctest: +NORMALIZE_WHITESPACE
array([array([], dtype=float64),
       array([ 2.,  2.,  1.,  1.,  1.,  1.,  0.]),
       array([ 2.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.]),
       array([ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.]),
       array([ 1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])], dtype=object)

:Authors: - Wilker Aziz
"""

from collections import defaultdict, deque
import numpy as np
import math
from functools import reduce


class NGramFactory(object):
    """
    Manages the mapping from ngrams (as tuples of strings) to unique ids (as pairs of ints).
    An n-gram is uniquely identified by two integers: (k, i) where k (1-based) is the n-gram order and i
    is a unique id (0-based) amongst all k-grams.


    >>> factory = NGramFactory(4)
    >>> factory.max_order
    4
    >>> factory.get('a b c'.split())
    (3, 0)
    >>> factory.get('a b c'.split())
    (3, 0)
    >>> factory.get('a'.split())
    (1, 0)
    >>> factory.get('b'.split())
    (1, 1)
    >>> factory[3,0]
    ('a', 'b', 'c')
    >>> factory[1,2]
    Traceback (most recent call last):
    ...
    IndexError: list index out of range
    >>> factory.ngrams(1)
    [('a',), ('b',)]

    """

    def __init__(self, max_order):
        self.max_order_ = max_order
        self.ngrams_ = [[] for _ in range(max_order + 1)]
        self.n2i_ = [defaultdict() for _ in range(max_order + 1)]

    @property
    def max_order(self):
        return self.max_order_

    def get(self, words):
        """get(words) -> NGram object"""
        key = tuple(words)
        order = len(words)
        nid = self.n2i_[order].get(key, None)
        if nid is None:
            nid = len(self.ngrams_[order])
            self.n2i_[order][key] = nid
            self.ngrams_[order].append(key)
        return order, nid

    def __getitem__(self, order_nid):
        """self[nid] -> NGram object"""
        order, nid = order_nid
        return self.ngrams_[order][nid]

    def ngrams(self, order):
        return self.ngrams_[order]

    def __str__(self):
        return '\n'.join('{0}: {1}'.format(order, ' | '.join(' '.join(ngram) for ngram in ngrams))
                         for order, ngrams in enumerate(self.ngrams_[1:], 1))


def closest(v, L):
    """
    Returns the element in L (a sorted numpy array) which is closest to v

    >>> R = np.array([9,3,6])
    >>> R.sort()
    >>> R
    array([3, 6, 9])
    >>> [closest(i, R) for i in range(1,12)]
    [3, 3, 3, 3, 6, 6, 6, 9, 9, 9, 9]
    """
    i = L.searchsorted(v)
    return L[-1 if i == len(L) else (0 if i == 0 else (i if v - L[i - 1] > L[i] - v else i - 1))]


def make_features(leaves, ngram_factory):
    """
    Returns a list of dictionaries such that the kth element represents k-gram counts.
    :param iterable leaves: words in a sentence
    :param NGramFactory ngram_factory:
    :return:
    """
    # count kgrams for k=1..n
    max_order = ngram_factory.max_order
    ngram_counts = [defaultdict(float) for _ in range(max_order + 1)]
    reversed_context = deque()
    for w in leaves:
        # left trim the context
        if len(reversed_context) == max_order:
            reversed_context.pop()
        # count ngrams ending in w
        ngram = deque([w])
        # a) process w
        order, nid = ngram_factory.get(ngram)
        ngram_counts[order][nid] += 1
        # b) process longer ngrams ending in w
        for h in reversed_context:
            ngram.appendleft(h)
            order, nid = ngram_factory.get(ngram)
            ngram_counts[order][nid] += 1
        # c) adds w to the context
        reversed_context.appendleft(w)
    return ngram_counts


def dict2array(sparse_features, dimension):
    """
    Converts a dictionary (representing sparse features of order k) into a numpy array
    :param dict sparse_features: feature name => feature value
    :param dimension: the total dimension (for order k features)
    :return:
    """
    dok = np.zeros(dimension)
    for k, v in sparse_features.items():
        dok[k] = v
    return dok


def make_arrays(all_sparse_features, ngram_factory):
    """
    Return a list of DOK matrices.
    :param list[dict] all_sparse_features: a list such that the kth element is a dict of sparse features of order k
    :param NGramFactory ngram_factory:
    :return: list of DOK matrices
    """
    dimensions = [len(ngram_factory.ngrams(k)) for k in range(ngram_factory.max_order + 1)]
    return np.array([dict2array(all_sparse_features[order], dimension) for order, dimension in enumerate(dimensions)])


class BLEU(object):
    """
    Implementations of BLEU
    """

    DEFAULT_MAX_ORDER = 4

    DEFAULT_SMOOTHING = 'ibm'

    @staticmethod
    def ibm_smoothing(cc, tc):
        pn = np.zeros(len(cc), float)
        k = 0
        for i, (c, t) in enumerate(zip(cc, tc)):
            p = c / t
            if p == 0:  # originally we would test whether c == 0, however it may happen that c > 0 (and extremelly small) and c/t is virtually zero
                k += 1
                p = 1.0 / math.pow(2, k)
            pn[i] = p
        return pn

    @staticmethod
    def p1_smoothing(cc, tc):
        """
        Sum 1 to numerator and denorminator for all orders.
        @param cc a vector of exactly n clipped counts (that is, 1-gram counts are in cc[0])
        @param tc a vector of exactly n total counts (that is, 1-gram counts are in cc[0])
        @return generator
        """
        return (cc + np.ones(len(cc), float)) / (tc + np.ones(len(tc), float))

    @staticmethod
    def bleu(r, c, cc, tc, n, smoothing):
        """
        @param r reference length
        @param c candidate length
        @param cc (clipped counts) is a vector of clipped counts such that cc[k] is the count for k-grams
        @param tc (total counts) is a vector of the total ngram counts (for the candidate), tc[k] is the count for k-grams
        @param n max ngram order
        @param smoothing computes smoothed precisions from cc and tc (both adjusted to exactly n positions)
        @return bleu
        """
        bp = 1.0 if c > r else math.exp(1 - float(r) / c)
        return bp * math.exp(1.0 / n * sum(math.log(pn) for pn in smoothing(cc[1:n + 1], tc[1:n + 1])))


class DecodingBLEU(object):
    """
    >>> H = [tuple('the black dog barked at the black cat'.split()), \
            tuple('black dog barks at black cat'.split())]
    >>> dbleu = DecodingBLEU(H, np.array([0.8, 0.2]))
    >>> dbleu.clipped_counts(0, 1)
    array([ 0.,  5.,  2.,  0.,  0.])
    >>> dbleu.clipped_counts(1, 0)
    array([ 0.,  5.,  2.,  0.,  0.])
    >>> dbleu.clipped_counts(0, 0)
    array([ 0.,  8.,  7.,  6.,  5.])
    >>> dbleu.clipped_counts(1, 1)
    array([ 0.,  6.,  5.,  4.,  3.])
    >>> dbleu.totals(0) == dbleu.clipped_counts(0, 0)
    array([ True,  True,  True,  True,  True], dtype=bool)
    >>> BLEU.ibm_smoothing(dbleu.clipped_counts(0, 1)[1:], dbleu.clipped_counts(0, 0)[1:])
    array([ 0.625     ,  0.28571429,  0.5       ,  0.25      ])
    >>> BLEU.p1_smoothing(dbleu.clipped_counts(0, 1)[1:], dbleu.clipped_counts(0, 0)[1:])
    array([ 0.66666667,  0.375     ,  0.14285714,  0.16666667])
    >>> dbleu.bleu(0, 0)
    1.0
    >>> dbleu.bleu(1, 1)
    1.0
    >>> dbleu.bleu(0, 1)
    0.38652758784697266
    >>> dbleu.bleu(1, 0)
    0.323729563941832
    >>> dbleu.posterior_
    array([ 0.8,  0.2])
    >>> dbleu.len_
    array([ 8.,  6.])
    >>> dbleu.expected_length()
    7.6000000000000005
    >>> dbleu.expected_clipped_counts(0)  # doctest: +NORMALIZE_WHITESPACE
    array([ 0. ,  7.4,  6. ,  4.8,  4. ])
    >>> dbleu.expected_clipped_counts(1)  # doctest: +NORMALIZE_WHITESPACE
    array([ 0. ,  5.2,  2.6,  0.8,  0.6])
    >>> dbleu.cobleu(0)
    0.8440024926773596
    >>> dbleu.cobleu(1)
    0.2806512592648872
    """

    def __init__(self, hypotheses, posterior, max_order=4, smoothing='ibm'):
        self.max_order_ = max_order
        self.ngram_factory_ = NGramFactory(max_order)
        self.sparse_features_ = [make_features(y, self.ngram_factory_) for y in hypotheses]

        self.doks_ = np.array([make_arrays(features, self.ngram_factory_) for features in self.sparse_features_])
        self.len_ = np.array([len(y) for y in hypotheses], float)
        # self.posterior_ = hypotheses.copy_posterior()
        self.posterior_ = np.array(posterior)

        # these are computed lazily
        self.cc_ = None
        self.tc_ = None
        self.ulen_ = None
        self.ucc_ = None

        if smoothing == 'p1':
            self.smoothing_ = BLEU.p1_smoothing
        elif smoothing == 'ibm':
            self.smoothing_ = BLEU.ibm_smoothing
        else:
            raise ValueError('Unknown type of smoothing "%s"' % smoothing)

    # TODO: store smoothed counts? (then totals has to change)
    def clipped_counts(self, c, r):
        if self.cc_ is None:
            doks = self.doks_
            m = len(doks)  # number of samples
            n = self.max_order_
            cc = [[None] * m for _ in range(m)]  # clipped counts
            for i in range(m):
                for j in range(i + 1):  # computes just the bottom half (because hypotheses==evidence)
                    cc[i][j] = np.array([np.minimum(doks[i][k], doks[j][k]).sum() for k in range(n + 1)])
            self.cc_ = cc
        return self.cc_[c][r] if r < c else self.cc_[r][c]

    def expected_clipped_counts(self, c):
        if self.ucc_ is None:
            # expected counts
            uc = np.array([self.posterior_.dot(np.array([dok[k] for dok in self.doks_]))
                           for k in range(self.max_order_ + 1)])
            # clip to expected counts
            self.ucc_ = np.array([[np.minimum(dok[k], uc[k]).sum() for k in range(self.max_order_ + 1)]
                                  for dok in self.doks_])
        return self.ucc_[c]

    def expected_length(self):
        if self.ulen_ is None:
            self.ulen_ = self.len_.dot(self.posterior_)
        return self.ulen_

    def totals(self, c):
        if self.tc_ is None:
            self.tc_ = np.array([np.array([counts_k.sum() for counts_k in dok]) for dok in self.doks_])
        return self.tc_[c]

    def bleu(self, c, r):
        """Returns the BLEU whe c is the candidate translation and r is the reference"""
        return BLEU.bleu(r=self.len_[r],
                         c=self.len_[c],
                         cc=self.clipped_counts(c, r),
                         tc=self.totals(c),
                         n=self.max_order_,
                         smoothing=self.smoothing_)

    def cobleu(self, c):
        return BLEU.bleu(r=self.expected_length(),
                         c=self.len_[c],
                         cc=self.expected_clipped_counts(c),
                         tc=self.totals(c),
                         n=self.max_order_,
                         smoothing=self.smoothing_)

    def reset(self):
        self.cc_ = None
        self.tc_ = None
        self.ulen_ = None
        self.ucc_ = None