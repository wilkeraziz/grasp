"""
A language model scorer using kenlm.

:Authors: - Wilker Aziz
"""

import numpy as np
import kenlm as klm
from grasp.cfg.projection import get_leaves
from .extractor import Stateful


class KenLMScorer(Stateful):

    DEFAULT_BOS_STRING = '<s>'
    DEFAULT_EOS_STRING = '</s>'

    def __init__(self, uid,
                 name,
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
        super(KenLMScorer, self).__init__(uid, name)
        self._order = order
        self._bos = bos
        self._eos = eos
        self._path = path
        self._model = klm.Model(path)
        self._features = (name, '{0}_OOV'.format(name))

        # get the initial state
        self._initial = klm.State()
        self._model.BeginSentenceWrite(self._initial)

    def weights(self, wmap):  # using dense representation
        wvec = []
        for f in self._features:
            try:
                wvec.append(wmap[f])
            except KeyError:
                raise KeyError('Missing LM feature: %s' % f)
        return np.array(wvec, float)

    def dot(self, fs, ws):  # dense dot
        return fs.dot(ws)

    def initial(self):
        return self._initial

    def final(self):
        return None

    def featurize_initial(self):
        return np.zeros(2)  # log_prob and oov

    def featurize_final(self, context):
        """
        :param context: a state
        :return:
        """
        out_state = klm.State()
        score = self._model.BaseFullScore(context, self._eos.surface, out_state)
        return np.array([score.log_prob, float(score.oov)])

    def featurize(self, word, context):
        """
        :param word: a Terminal
        :param context: a state
        :returns: weight, state
        """
        out_state = klm.State()
        score = self._model.BaseFullScore(context, word.surface, out_state)
        return np.array([score.log_prob, float(score.oov)]), out_state

    def featurize_derivation(self, derivation):
        """
        :param words: sequence of Terminal objects
        :return: weight
        """
        return self.featurize_yield(get_leaves(derivation))

    def featurize_yield(self, projection):
        """
        :param words: sequence of Terminal objects
        :return: weight
        """
        qa = klm.State()
        qb = klm.State()
        self._model.BeginSentenceWrite(qa)
        log_prob = 0.0
        oov = 0.0
        for word in projection:
            r = self._model.BaseFullScore(qa, word.surface, qb)
            log_prob += r.log_prob
            oov += int(r.oov)
            qa, qb = qb, qa
        log_prob += self._model.BaseScore(qa, self._eos.surface, qb)
        return np.array([log_prob, oov])