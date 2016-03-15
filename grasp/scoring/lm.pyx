"""
A language model scorer using kenlm.

:Authors: - Wilker Aziz
"""
import kenlm as klm
from grasp.ptypes cimport weight_t
from grasp.cfg.symbol cimport Symbol, Terminal
from grasp.scoring.frepr cimport FRepr, FVec
from grasp.scoring.extractor cimport StatefulFRepr


DEFAULT_BOS_STRING = '<s>'
DEFAULT_EOS_STRING = '</s>'


cdef class StatelessLM(Stateless):

    def __init__(self, int uid,
                 str name,
                 int order,
                 str path,
                 Terminal bos=Terminal(DEFAULT_BOS_STRING),
                 Terminal eos=Terminal(DEFAULT_EOS_STRING)):
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

    cpdef tuple fnames(self, wkeys):
        return self._features

    cpdef FRepr weights(self, dict wmap):  # using dense representation
        cdef list wvec = []
        cdef str f
        for f in self._features:
            try:
                wvec.append(wmap[f])
            except KeyError:
                raise KeyError('Missing stateless LM feature: %s' % f)
        return FVec(wvec)

    cpdef FRepr featurize(self, edge):
        """
        :param word: a Terminal
        :param context: a state
        :returns: weight, state
        """
        in_state = klm.State()
        out_state = klm.State()
        cdef:
            weight_t score = 0
            int oov = 0
            Symbol s

        for s in edge.rhs:
            if not isinstance(s, Terminal):  # here we forget the context: this LM is stateless ;)
                self._model.NullContextWrite(in_state)
                continue
            if s == self._bos:  # here we ste the context as <s> and proceed (we never really score BOS)
                self._model.BeginSentenceWrite(in_state)
                continue
            # here we simply score s.surface given in_state
            result = self._model.BaseFullScore(in_state, s.surface, out_state)
            score += <weight_t>result.log_prob
            oov += int(<bint>result.oov)
            # and update the state (internally)
            in_state, out_state = out_state, in_state
        return FVec([score, oov])

    cpdef FRepr constant(self, weight_t value):
        return FVec([value, value])


cdef class KenLM(Stateful):
    """
    A Language model feature extractor.
    """

    def __init__(self, int uid,
                 str name,
                 int order,
                 str path,
                 Terminal bos=Terminal(DEFAULT_BOS_STRING),
                 Terminal eos=Terminal(DEFAULT_EOS_STRING)):
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
        super(KenLM, self).__init__(uid, name)
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
        return '{0}(uid={1}, name={2}, order={3}, path={4}, bos={5}, eos={6})'.format(KenLM.__name__,
                                                                                      repr(self.id),
                                                                                      repr(self.name),
                                                                                      repr(self._order),
                                                                                      repr(self._path),
                                                                                      repr(self._bos),
                                                                                      repr(self._eos))

    cpdef tuple fnames(self, wkeys):
        return self._features

    cpdef FRepr weights(self, dict wmap):  # using dense representation
        cdef list wvec = []
        cdef str f
        for f in self._features:
            try:
                wvec.append(wmap[f])
            except KeyError:
                raise KeyError('Missing LM feature: %s' % f)
        return FVec(wvec)

    cpdef object initial(self):
        return self._initial

    cpdef object final(self):
        return None

    cpdef FRepr featurize_initial(self):
        return FVec([0, 0])

    cpdef FRepr featurize_final(self, context):
        """
        :param context: a state
        :return:
        """
        out_state = klm.State()
        r = self._model.BaseFullScore(context, self._eos.surface, out_state)
        return FVec([r.log_prob, float(r.oov)])

    cpdef StatefulFRepr featurize(self, word, context):
        """
        :param word: a Terminal
        :param context: a state
        :returns: weight, state
        """
        out_state = klm.State()
        r = self._model.BaseFullScore(context, word.surface, out_state)
        return StatefulFRepr(FVec([r.log_prob, float(r.oov)]), out_state)

    cpdef FRepr featurize_yield(self, derivation_yield):
        """
        :param words: sequence of Terminal objects
        :return: weight
        """
        cdef:
            weight_t log_prob = 0.0
            weight_t oov = 0.0
            Terminal word
            object r
        qa = klm.State()
        qb = klm.State()
        self._model.BeginSentenceWrite(qa)
        for word in derivation_yield:
            r = self._model.BaseFullScore(qa, word.surface, qb)
            log_prob += <weight_t>r.log_prob
            oov += int(r.oov)
            qa, qb = qb, qa
        log_prob += self._model.BaseScore(qa, self._eos.surface, qb)
        return FVec([log_prob, oov])

    cpdef FRepr constant(self, weight_t value):
        return FVec([value, value])