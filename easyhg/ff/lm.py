"""
A language model scorer using kenlm.

:Authors: - Wilker Aziz
"""

import kenlm as klm
from .scorer import Stateful


class KenLMScorer(Stateful):

    DEFAULT_FNAME = 'LanguageModel'
    DEFAULT_BOS_STRING = '<s>'
    DEFAULT_EOS_STRING = '</s>'

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
        super(KenLMScorer, self).__init__(uid, name, weights)
        self._order = order
        self._bos = bos
        self._eos = eos
        self._path = path
        self._model = klm.Model(path)

        # get the initial state
        self._initial = klm.State()
        self._model.BeginSentenceWrite(self._initial)


    def initial(self):
        return self._initial

    def final(self):
        return None

    def initial_score(self):
        return 0.0

    def final_score(self, context):
        """
        :param context: a state
        :return:
        """
        out_state = klm.State()
        score = self._model.BaseFullScore(context, self._eos.surface, out_state)
        return score.log_prob * self.weights[0] + score.oov * self.weights[1]

    def score(self, word, context):
        """
        :param word: a Terminal
        :param context: a state
        :returns: weight, state
        """
        out_state = klm.State()
        score = self._model.BaseFullScore(context, word.surface, out_state)
        return score.log_prob * self.weights[0] + score.oov * self.weights[1], out_state

    def total_score(self, words):
        """
        :param words: sequence of Terminal objects
        :return: weight
        """
        qa = klm.State()
        qb = klm.State()
        self._model.BeginSentenceWrite(qa)
        log_prob = 0
        oov = 0
        for word in words:
            r = self._model.BaseFullScore(qa, word.surface, qb)
            log_prob += r.log_prob
            oov += int(r.oov)
            qa, qb = qb, qa
        log_prob += self._model.BaseScore(qa, self._eos.surface, qb)
        return log_prob * self.weights[0] + oov * self.weights[1]


