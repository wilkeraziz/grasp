import numpy as np
cimport numpy as np
from libcpp.vector cimport vector as cppvector
from libcpp.deque cimport deque as cppdeque
from libcpp.string cimport string as cppstring
from libcpp.pair cimport pair as cpppair
from libcpp.map cimport map as cppmap
from cython.operator cimport preincrement as cppinc
from cython.operator cimport dereference as deref
from collections import defaultdict, deque


cdef cpp_make_features(cppvector[cppstring] &leaves,
        cppvector[cppmap[cppdeque[cppstring],float]] &ngram_counts,
        cppvector[float] &totals,
        int max_order=4,
        float posterior=1.0):
    """
    Returns
        1) a list (indexed by ngram length minus 1) of features counts (indexed by reversed ngrams)
        2) a list (indexed by ngram length minus 1) of total counts
    """
    # count kgrams for k=1..n
    ngram_counts.clear()
    ngram_counts.resize(max_order)
    totals.clear()
    totals.resize(max_order)
    cdef cppdeque[cppstring] reversed_context
    cdef cppdeque[cppstring] ngram
    cdef cppvector[cppstring].iterator w_it = leaves.begin()
    cdef cppstring w, h
    while w_it != leaves.end():
        w = deref(w_it)
        # left trim the context
        if reversed_context.size() == max_order:
            reversed_context.pop_back()
        # count ngrams ending in w
        ngram.clear()
        ngram.push_front(w)
        # a) process w
        totals[ngram.size() - 1] += posterior
        ngram_counts[ngram.size() - 1][ngram] += posterior
        # b) process longer ngrams ending in w
        for h in reversed_context:
            ngram.push_front(h)
            totals[ngram.size() - 1] += posterior
            ngram_counts[ngram.size() - 1][ngram] += posterior
        # c) adds w to the context
        reversed_context.push_front(w)
        cppinc(w_it)


cdef cppvector[cppmap[cppdeque[cppstring],float]] cpp_max_features(cppvector[cppvector[cppstring]] &references, int max_order=4):
    """
    Returns
        1) a list (indexed by ngram length minus 1) feature counts (indexed by reversed ngram) from all references.
        we only keep the maximum count per feature
    """
    cdef cppvector[cppmap[cppdeque[cppstring],float]] max_counts
    cdef cppvector[cppmap[cppdeque[cppstring],float]] counts
    cdef cppvector[float] totals  # not really needed
    cdef int i
    max_counts.resize(max_order)
    cdef cppvector[cppvector[cppstring]].iterator it = references.begin()
    while it != references.end():
        cpp_make_features(deref(it), counts, totals, max_order)
        for i in range(max_order):
            update_dmax(max_counts[i], counts[i])
        cppinc(it)
    return max_counts



cdef make_features(leaves,
        cppvector[cppmap[cppdeque[cppstring],float]] &ngram_counts,
        cppvector[float] &totals,
        int max_order=4,
        float posterior=1.0):
    """
    Returns
        1) a list (indexed by ngram length minus 1) of features counts (indexed by reversed ngrams)
        2) a list (indexed by ngram length minus 1) of total counts
    """
    # count kgrams for k=1..n
    ngram_counts.clear()
    ngram_counts.resize(max_order)
    totals.clear()
    totals.resize(max_order)
    cdef cppdeque[cppstring] reversed_context
    cdef cppdeque[cppstring] ngram
    cdef cppstring w, h
    cdef str leaf
    for leaf in leaves:
        w = leaf.encode('utf8')
        # left trim the context
        if reversed_context.size() == max_order:
            reversed_context.pop_back()
        # count ngrams ending in w
        ngram.clear()
        ngram.push_front(w)
        # a) process w
        totals[ngram.size() - 1] += posterior
        ngram_counts[ngram.size() - 1][ngram] += posterior
        # b) process longer ngrams ending in w
        for h in reversed_context:
            ngram.push_front(h)
            totals[ngram.size() - 1] += posterior
            ngram_counts[ngram.size() - 1][ngram] += posterior
        # c) adds w to the context
        reversed_context.push_front(w)


cdef update_dmax(cppmap[cppdeque[cppstring],float] &dmax, cppmap[cppdeque[cppstring],float] &d):
    """In analogy to +=, this computes max="""
    cdef cppmap[cppdeque[cppstring],float].const_iterator it
    cdef cppmap[cppdeque[cppstring],float].iterator r
    it = d.const_begin()
    while it != d.const_end():
        pair = dmax.insert(deref(it))
        if not pair.second:  # already existed
            r = pair.first
            if deref(it).second > deref(r).second:
                deref(r).second = deref(it).second
        cppinc(it)


cdef update_dsum(cppmap[cppdeque[cppstring],float] &dsum, cppmap[cppdeque[cppstring],float] &d, float posterior=1.0):
    """In analogy to +=, this computes max="""
    cdef cppmap[cppdeque[cppstring],float].const_iterator it
    it = d.const_begin()
    while it != d.const_end():
        dsum[deref(it).first] += deref(it).second * posterior
        cppinc(it)


cdef clip_counts(cppmap[cppdeque[cppstring],float] &d, cppmap[cppdeque[cppstring],float] &dmax):
    """In analogy to -=, this implements min="""
    cdef cppmap[cppdeque[cppstring],float].iterator it
    cdef cppmap[cppdeque[cppstring],float].const_iterator r
    it = d.begin()
    while it != d.end():
        r = dmax.const_find(deref(it).first)
        if r == dmax.const_end():
            deref(it).second = 0.0
        elif deref(it).second > deref(r).second:
            deref(it).second = deref(r).second
        cppinc(it)

cdef cppvector[cppmap[cppdeque[cppstring],float]] max_features(references, int max_order=4):
    """
    Returns
        1) a list (indexed by ngram length minus 1) feature counts (indexed by reversed ngram) from all references.
        we only keep the maximum count per feature
    """
    cdef cppvector[cppmap[cppdeque[cppstring],float]] max_counts
    cdef cppvector[cppmap[cppdeque[cppstring],float]] counts
    cdef cppvector[float] totals  # not really needed
    cdef int i
    max_counts.resize(max_order)
    for ref in references:
        make_features(ref, counts, totals, max_order)
        for i in range(max_order):
            update_dmax(max_counts[i], counts[i])
    return max_counts

cdef cppvector[cppmap[cppdeque[cppstring],float]] expected_features(hyps,
        np.float_t[::1] posterior,
        #cppvector[cppvector[cppmap[cppdeque[cppstring],float]]] &counts,
        int max_order=4):
    """
    Returns
        1) a list (indexed by ngram length minus 1) feature counts (indexed by reversed ngram) from all references.
        we only keep the maximum count per feature
    """
    cdef cppvector[cppmap[cppdeque[cppstring],float]] expected_counts
    #cdef cppvector[cppvector[cppmap[cppdeque[cppstring],float]]].iterator c
    cdef cppvector[cppmap[cppdeque[cppstring],float]] counts
    cdef cppvector[float] totals  # not really needed
    cdef int i, k
    expected_counts.resize(max_order)
    #counts.resize(len(hyps))
    for i in range(len(hyps)):
        make_features(hyps[i], counts, totals, max_order, posterior[i])
        for k in range(max_order):
            update_dsum(expected_counts[k], counts[k])  #, posterior[i])
    return expected_counts

cdef float closest(float v, np.float_t[::1] L):
    """
    Returns the element in L (a sorted numpy array) which is closest to v
    """
    cdef int i = np.searchsorted(L, v)
    return L[-1 if i == len(L) else (0 if i == 0 else (i if v - L[i - 1] > L[i] - v else i - 1))]

cdef float sum_counts(cppmap[cppdeque[cppstring],float] &counts):
    cdef cppmap[cppdeque[cppstring],float].const_iterator it
    cdef float total = 0.0
    it = counts.const_begin()
    while it != counts.const_end():
        total += deref(it).second
        cppinc(it)
    return total


cdef bleu(hyp, np.float_t[::1] ref_lenghts, cppvector[cppmap[cppdeque[cppstring],float]] &ref_counts, int max_order, float smoothing=1.0):
    # hyp length
    cdef float h_length = len(hyp)
    # closest ref length
    cdef float r_length = closest(h_length, ref_lenghts)
    # compute the brevity penalty
    cdef float bp = 1.0 if h_length > r_length else np.exp(1 - float(r_length) / h_length)
    # get h counts
    cdef cppvector[cppmap[cppdeque[cppstring],float]] h_counts
    cdef cppvector[float] h_totals
    make_features(hyp, h_counts, h_totals, max_order)
    # clip counts wrt refs and compute precision with +1 smoothing
    cdef np.float_t[::1] cc = np.zeros(max_order, float)
    cdef int i
    for i in range(max_order):
        clip_counts(h_counts[i], ref_counts[i])
        cc[i] = sum_counts(h_counts[i])
    # compute ngram precisions with +1 smoothing
    cdef np.float_t[::1] pn = np.add(cc, smoothing) / np.add(np.array(h_totals), smoothing)
    # return BLEU
    return bp * np.exp(1.0 / max_order * np.sum(np.log(pn)))

cdef cobleu(hyp, float expected_length, cppvector[cppmap[cppdeque[cppstring],float]] &expected_counts, int max_order, float smoothing=1.0):
    # hyp length
    cdef float h_length = len(hyp)
    # compute the brevity penalty
    cdef float bp = 1.0 if h_length > expected_length else np.exp(1 - expected_length / h_length)
    # get h counts
    cdef cppvector[cppmap[cppdeque[cppstring],float]] h_counts
    cdef cppvector[float] h_totals
    make_features(hyp, h_counts, h_totals, max_order)
    # clip counts wrt expected counts and compute precision with +1 smoothing
    cdef np.float_t[::1] cc = np.zeros(max_order, float)
    cdef int k
    for k in range(max_order):
        clip_counts(h_counts[k], expected_counts[k])
        cc[k] = sum_counts(h_counts[k])
    # compute ngram precisions with +1 smoothing
    cdef np.float_t[::1] pn = np.add(cc, smoothing) / np.add(np.array(h_totals), smoothing)
    # return BLEU
    return bp * np.exp(1.0 / max_order * np.sum(np.log(pn)))


cdef cppvector[cppstring] p2c_sequence(list py_sequence):
    cdef size_t length = len(py_sequence)
    cdef cppvector[cppstring] cpp_sequence
    cdef size_t i
    cdef str py_word
    cpp_sequence.resize(length)
    for i in range(length):
        cpp_sequence[i] = py_sequence[i].encode('utf8')
    return cpp_sequence

cdef cppvector[cppvector[cppstring]] p2c_sequences(list py_sequences):
    cdef size_t n_sequences = len(py_sequences)
    cdef cppvector[cppvector[cppstring]] cpp_sequences
    cdef size_t i
    cpp_sequences.resize(n_sequences)
    # convert hypotheses and references to cpp objects
    for i in range(n_sequences):
        cpp_sequences[i] = p2c_sequence(py_sequences[i])
    return cpp_sequences


cdef class TrainingBLEU(object):
    """
    >>> from grasp.loss.fast_bleu import TrainingBLEU
    >>> R = ['the black dog barked at the black cat'.split(), 'the dog barked at the cat'.split()]
    >>> H = ['the black dog barked at a black cat'.split(), 'black dog barks at black cat'.split(), 'the black dog barked at the black cat'.split()]
    >>> tbleu = TrainingBLEU(R, 4, smoothing=1.0)
    >>> tbleu.gain(H[0])
    0.660632848739624
    >>> tbleu.gain(H[1])
    0.27414700388908386
    >>> tbleu.gain(H[2])
    1.0

    """

    cdef cppvector[cppmap[cppdeque[cppstring],float]] _max_counts
    cdef int _max_order
    cdef float _smoothing
    cdef np.float_t[::1] _ref_lengths

    def __init__(self, refs, int max_order=4, float smoothing=1.0):
        """
        :param refs: list of references (each reference is a sequence of words
        """
        #cdef cppvector[cppvector[cppstring]] references = p2c_sequences(refs)
        self._max_order = max_order
        self._smoothing = smoothing
        self._max_counts = max_features(refs, max_order)
        self._ref_lengths = np.array([len(ref) for ref in refs], float)

    cpdef float loss(self, hyp):
        return 1.0 - bleu(hyp, self._ref_lengths, self._max_counts, self._max_order, self._smoothing)

    cpdef float gain(self, hyp):
        return bleu(hyp, self._ref_lengths, self._max_counts, self._max_order, self._smoothing)


cdef class DecodingBLEU(object):
    """
    >>> import numpy as np
    >>> from grasp.loss.fast_bleu import DecodingBLEU
    >>> H = ['the black dog barked at the black cat'.split(), 'black dog barks at black cat'.split()]
    >>> dbleu = DecodingBLEU(H, np.array([0.8, 0.2]))
    >>> dbleu.gain(H[0])
    0.8665595054626465
    >>> dbleu.gain(H[1])
    0.4028468728065491

    """

    cdef cppvector[cppmap[cppdeque[cppstring],float]] _expected_counts
    #cdef cppvector[cppvector[cppmap[cppdeque[cppstring],float]]] _counts
    cdef int _max_order
    cdef float _smoothing
    cdef float _expected_length

    def __init__(self, hyps, posterior, int max_order=4, float smoothing=1.0):
        self._max_order = max_order
        self._smoothing = smoothing
        cdef np.float_t[::1] py = np.array(posterior, float)
        self._expected_counts = expected_features(hyps, py, max_order)
        self._expected_length = np.array([len(hyp) for hyp in hyps], float).dot(py)

    cpdef float loss(self, hyp):
        return 1.0 - cobleu(hyp, self._expected_length, self._expected_counts, self._max_order, self._smoothing)

    cpdef float gain(self, hyp):
        return cobleu(hyp, self._expected_length, self._expected_counts, self._max_order, self._smoothing)


cpdef read_hyps(istream):
    """
    Read a document of hypothesised translations from an input stream.
    :param istream:
    :return: doc:[ hyp:[ word:str ] ]
    """
    return [line.split() for line in istream]


cpdef read_refsets(istream):
    """
    Read a document of reference sets from an input stream
    :param istream:
    :return: doc:[ refset:[ ref:[ word:str ] ] ]
    """
    return [[ref.split() for ref in line.strip().split(' ||| ')] for line in istream]


cdef cpp_read_hyps(istream, cppvector[cppvector[cppstring]] &doc):
    """
    cpp version of read_hyps which fills up cpp containers
    """
    cdef str pystr
    doc.clear()
    for pystr in istream:
        doc.push_back(p2c_sequence(pystr.split()))


cdef cpp_read_refsets(istream, cppvector[cppvector[cppvector[cppstring]]] &doc):
    """
    cpp version of read_refsets which fills up cpp containers
    """
    cdef str refset
    cdef list refs
    cdef size_t i
    doc.clear()
    for refset in istream:
        refs = refset.split(' ||| ')
        doc.push_back([])
        doc.back().resize(len(refs))
        for i in range(len(refs)):
            doc.back()[i] = p2c_sequence(refs[i].split())


cpdef tuple stream_doc_bleu(hyp_stream, refset_stream, int max_order=4, float smoothing=0.0):
    """
    Computes document level BLEU from two streams of text (hypotheses and reference sets).

    In ipython: with a small test set (10 segments)
        %timeit stream_doc_bleu(hstream, rstream)
        100 loops, best of 3: 12.4 ms per loop

    :param hyp_stream: a stream of python strings (each representing a hypothesis)
    :param refset_stream: a stream of python strings (each representing a reference set)
    :param max_order:
    :param smoothing:
    :return:
    """
    cdef cppvector[cppvector[cppstring]] hyps
    cdef cppvector[cppvector[cppvector[cppstring]]] refs
    cpp_read_hyps(hyp_stream, hyps)
    cpp_read_refsets(refset_stream, refs)
    return _doc_bleu(hyps, refs, max_order, smoothing)

cpdef tuple doc_bleu(list hyps, list refsets, int max_order=4, float smoothing=0.0):
    """
    Computes document level BLEU from a list of hypotheses and a list of reference sets.

    In ipython: with a small test set (10 segments)
        %timeit doc_bleu(H, R)
        100 loops, best of 3: 12.3 ms per loop

    :param hyps: a list of hypotheses
    :param refsets: a list of reference sets
    :param max_order:
    :param smoothing:
    :return:
    """
    cdef cppvector[cppvector[cppstring]] cpp_hyps
    cdef cppvector[cppvector[cppvector[cppstring]]] cpp_refsets
    cdef size_t doc_size = len(hyps)
    cdef size_t i
    cpp_hyps.resize(doc_size)
    cpp_refsets.resize(doc_size)
    for i in range(doc_size):
        cpp_hyps[i] = p2c_sequence(hyps[i])
        cpp_refsets[i] = p2c_sequences(refsets[i])
    return _doc_bleu(cpp_hyps, cpp_refsets, max_order, smoothing)


cdef tuple _doc_bleu(cppvector[cppvector[cppstring]] hypotheses,
                     cppvector[cppvector[cppvector[cppstring]]] refsets,
                     int max_order=4, float smoothing=0.0):
    """
    Computes document level BLEU from cpp containers representing a list of hypotheses and a list of reference sets.

    Definitions:
        - a sentence is a list of words
            sentence = [words]
        - a hypothesis is a sentence
        - a reference is a sentence
        - a "reference set" is a list of sentences
        - each hypothesis is paired with a non-empty reference set
        - a document is a list of hypotheses paired with their respective reference sets

    :param H: a list of hypotheses
    :param R: a list of reference sets
    :param max_order: 4 by default
    :param smoothing: stupid add-p smoothing
    :return: bleu score, n-gram precisions, brevity penalty
    """

    cdef size_t doc_size = hypotheses.size()
    cdef size_t i, j, n, n_refs


    # brevity penalty
    cdef np.float_t[::1] ref_lengths
    cdef float hyp_length = 0.0
    cdef float ref_length = 0.0
    for i in range(doc_size):
        # get the length of each reference
        n_refs = refsets[i].size()
        ref_lengths = np.array([refsets[i][j].size() for j in range(n_refs)], float)
        hyp_length += hypotheses[i].size()
        # closest ref length
        ref_length += closest(hypotheses[i].size(), ref_lengths)
    cdef float bp = 1.0 if hyp_length > ref_length else np.exp(1 - float(ref_length) / hyp_length)

    # clipped counts
    cdef np.float_t[::1] cc = np.zeros(max_order, float)
    cdef np.float_t[::1] totals = np.zeros(max_order, float)
    cdef cppvector[cppmap[cppdeque[cppstring],float]] hi_counts
    cdef cppvector[float] hi_totals

    for i in range(doc_size):
        # get ref n-gram features
        ri_counts = cpp_max_features(refsets[i], max_order)
        # get hyp n-gram features
        cpp_make_features(hypotheses[i], hi_counts, hi_totals, max_order)  # make_features resets the containers before updating them
        # for each n-gram order
        for n in range(max_order):
            # clip counts
            clip_counts(hi_counts[n], ri_counts[n])
            # accumulate numerator
            cc[n] += sum_counts(hi_counts[n])
            # accumulate denominator
            totals[n] += hi_totals[n]

    # compute ngram precisions with +1 smoothing
    cdef np.float_t[::1] pn = np.add(cc, smoothing) / np.add(np.array(totals), smoothing)
    return bp * np.exp(1.0 / max_order * np.sum(np.log(pn))), np.array(pn), bp