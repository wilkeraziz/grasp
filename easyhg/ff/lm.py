"""
:Authors: - Wilker Aziz
"""

from collections import defaultdict
from kenlm import LanguageModel as KLM
from .scorer import Stateful
from easyhg.grammar.symbol import Terminal


# TODO: interact with kenlm at a deeper level and save states
# which should make the queries considerably faster!


class LMScorer(Stateful):
    """

    """

    DEFAULT_FNAME = 'LanguageModel'
    DEFAULT_BOS_SYMBOL = '<s>'
    DEFAULT_EOS_SYMBOL = '</s>'

    def __init__(self, uid,
                 name,
                 weights,
                 order,
                 path,
                 bos,
                 eos):
        """
        A language model scorer (KenLM only).

        :param uid: unique id (int)
        :param name: prefix for features
        :param weights: weight vector (two features: logprob and oov count)
        :param order: n-gram order
        :param bos: a Terminal symbol representing the left boundary of the sentence.
        :param eos: a Terminal symbol representing the right boundary of the sentence.
        :param path: path to a kenlm model (ARPA or binary).
        :return:
        """
        super(LMScorer, self).__init__(uid, name, weights)
        self._order = order
        self._bos = bos
        self._eos = eos
        # TODO: organise states in a trie
        self._ngrams = []
        self._ngram2state = defaultdict(None)
        self._initial = self.get_state((self._bos,))
        self._final = self.get_state((self._eos,))
        self._path = path
        self._kenlm = KLM(path)

    def initial(self):
        return self._initial

    def final(self):
        return self._final

    def initial_score(self):
        return 0.0

    def final_score(self, context):
        """

        :param context: a state
        :return:
        """
        prefix = self._ngrams[context]
        bos = False
        if prefix[0] == self._bos:
            prefix = prefix[1:]  # we should not include boundary symbols in the query
            bos = True  # we use flags instead
        query = ' '.join(t.surface for t in prefix)  # a query is a string (TODO: change this)
        scores = list(self._kenlm.full_scores(query, bos=bos, eos=True))  # score all ngrams (TODO: change this)
        logprob, size, oov = scores[-1]
        return logprob * self.weights[0]

    def score(self, word, context):
        """
        :param word: a Terminal
        :param context: a state
        :returns: weight
        """
        ngram = self._ngrams[context] + (word,)
        bos = False
        if ngram[0] == self._bos:  # boundary words are not part of the query
            ngram = ngram[1:]
            bos = True
        query = ' '.join(t.surface for t in ngram)  # a query is a string (TODO: change this)
        scores = list(self._kenlm.full_scores(query, bos=bos, eos=False))  # score all ngrams (TODO: change this)
        logprob, size, oov = scores[-1]
        return logprob * self.weights[0] + oov * self.weights[1]

    def next(self, word, context):
        """
        :param word: a Terminal
        :param context: a state
        :returns: weight
        """

        ngram = self._ngrams[context] + (word,)
        # left-trim if necessary
        if len(ngram) == self._order:
            ngram = ngram[1:]
        to = self.get_state(ngram)
        return to

    def get_state(self, ngram):
        """
        Return the state associated with an ngram.
        :param ngram: a tuple of Terminal symbols
        :returns: a state id (int)
        """
        state = self._ngram2state.get(ngram, None)
        if state is None:
            state = len(self._ngrams)
            self._ngrams.append(ngram)
            self._ngram2state[ngram] = state
        return state
