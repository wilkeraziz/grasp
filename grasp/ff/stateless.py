"""
:Authors: - Wilker Aziz
"""

from collections import defaultdict
import kenlm as klm
import numpy as np
from .extractor import Stateless
from grasp.cfg.symbol import Terminal, Nonterminal


class WordPenalty(Stateless):
    def __init__(self, uid, name, penalty=1.0):
        super(WordPenalty, self).__init__(uid, name)
        self._penalty = penalty

    def __repr__(self):
        return '{0}(uid={1}, name={2}, penalty={3})'.format(WordPenalty.__name__,
                                                            repr(self.id),
                                                            repr(self.name),
                                                            repr(self._penalty))

    def weights(self, wmap):
        try:
            return wmap.get(self.name)
        except KeyError:
            raise KeyError('Missing weight for WordPenalty')

    def featurize(self, edge):
        """
        :param word: a Terminal
        :param context: a state
        :returns: weight
        """
        return sum(self._penalty for sym in filter(lambda s: isinstance(s, Terminal), edge.rhs))

    def dot(self, frepr, wrepr):
        """
        :param frepr: word penalty (a float)
        :param wrepr: weight (a float)
        :return: product
        """
        return frepr * wrepr


class ArityPenalty(Stateless):
    def __init__(self, uid, name, penalty=1.0):
        super(ArityPenalty, self).__init__(uid, name)
        self._penalty = penalty

    def __repr__(self):
        return '{0}(uid={1}, name={2}, penalty={3})'.format(ArityPenalty.__name__,
                                                            repr(self.id),
                                                            repr(self.name),
                                                            repr(self._penalty))

    def weights(self, wmap):  # using a sparse representation
        return defaultdict(None, ((k, v) for k, v in wmap.items() if k.startswith(self.name)))

    def featurize(self, edge):
        """
        :param rule:
        :returns: edge's arity
        """
        arity = sum(1 for _ in filter(lambda s: isinstance(s, Nonterminal), edge.rhs))
        return arity

    def dot(self, frepr, wrepr):
        """

        :param frepr: edge's arity (an integer)
        :param wrepr: weight map
        :return: penalty * wmap[arity]
        """
        arity = frepr
        return self._penalty * wrepr.get('{0}_{1}'.format(self.name, arity), 0)


class StatelessLM(Stateless):

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
        super(StatelessLM, self).__init__(uid, name)
        self._order = order
        self._bos = bos
        self._eos = eos
        self._path = path
        self._model = klm.Model(path)
        self._features = (name, '{0}_OOV'.format(name))

        # get the initial state
        self._initial = klm.State()
        self._model.BeginSentenceWrite(self._initial)

    def __repr__(self):
        return '{0}(uid={1}, name={2}, order={3}, path={4}, bos={5}, eos={6})'.format(StatelessLM.__name__,
                                                                                      repr(self.id),
                                                                                      repr(self.name),
                                                                                      repr(self._order),
                                                                                      repr(self._path),
                                                                                      repr(self._bos),
                                                                                      repr(self._eos))

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

    def featurize(self, edge):
        """
        :param word: a Terminal
        :param context: a state
        :returns: weight, state
        """
        in_state = klm.State()
        out_state = klm.State()
        score = 0
        oov = 0
        for s in edge.rhs:
            if isinstance(s, Nonterminal):  # here we forget the context: this LM is stateless ;)
                self._model.NullContextWrite(in_state)
                continue
            if s == self._bos:  # here we ste the context as <s> and proceed (we never really score BOS)
                self._model.BeginSentenceWrite(in_state)
                continue
            # here we simply score s.surface given in_state
            result = self._model.BaseFullScore(in_state, s.surface, out_state)
            score += result.log_prob
            oov += int(result.oov)
            # and update the state (internally)
            in_state, out_state = out_state, in_state
        return np.array([score, oov], float)