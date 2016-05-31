"""
A language model scorer using kenlm.

:Authors: - Wilker Aziz
"""
import kenlm as klm
from grasp.ptypes cimport weight_t
from grasp.cfg.symbol cimport Symbol, Terminal
from grasp.scoring.frepr cimport FRepr, FVec, FValue
from grasp.scoring.extractor cimport StatefulFRepr
from grasp.recipes import re_key_value
import os

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
        if not os.path.exists(path):
            raise FileNotFoundError('LM file not found: %s' % path)
        self._features = (name, '{0}_OOV'.format(name))
        self._load_model()

    cdef _load_model(self):
        # get the initial state
        self._model = klm.Model(self._path)
        self._initial = klm.State()
        self._model.BeginSentenceWrite(self._initial)

    def __getstate__(self):
        return super(StatelessLM,self).__getstate__(), {'order': self._order,
                                                        'bos': self._bos,
                                                        'eos': self._eos,
                                                        'path': self._path,
                                                        'features': self._features}

    def __setstate__(self, state):
        superstate, d = state
        self._order = d['order']
        self._bos = d['bos']
        self._eos = d['eos']
        self._path = d['path']
        self._features = d['features']
        self._load_model()
        super(StatelessLM,self).__setstate__(superstate)

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

    cpdef tuple features(self):
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

    @classmethod
    def construct(cls, int uid, str name, str cfgstr):
        cdef int order
        cdef str path
        cdef bos = DEFAULT_BOS_STRING
        cdef eos = DEFAULT_EOS_STRING
        cfgstr, value = re_key_value('order', cfgstr, optional=False)
        if value:
            order = int(value)
        cfgstr, value = re_key_value('path', cfgstr, optional=False)
        if value:
            path = value
        cfgstr, value = re_key_value('bos', cfgstr, optional=True)
        if value:
            bos = value
        cfgstr, value = re_key_value('eos', cfgstr, optional=True)
        if value:
            eos = value
        return StatelessLM(uid, name, order, path, Terminal(bos), Terminal(eos))

    @staticmethod
    def help():
        help_msg = ["# A stateless LM is one that forgets the history at the boundary of nonterminals.",
                    "# It is a common heuristic in MT to ignore missing context.",
                    "# Requires: order=int path=str",
                    "# Optional: bos=str eos=str"]
        return '\n'.join(help_msg)

    @staticmethod
    def example():
        return 'StatelessLM order=3 path=trigrams.klm bos=<s> eos=</s>'


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
        if not os.path.exists(path):
            raise FileNotFoundError('LM file not found: %s' % path)
        self._features = (name, '{0}_OOV'.format(name))
        # get the initial state
        self._load_model()

    cdef _load_model(self):
        # get the initial state
        self._model = klm.Model(self._path)
        self._initial = klm.State()
        self._model.BeginSentenceWrite(self._initial)

    def __getstate__(self):
        return super(KenLM,self).__getstate__(), {'order': self._order,
                                                        'bos': self._bos,
                                                        'eos': self._eos,
                                                        'path': self._path,
                                                        'features': self._features}

    def __setstate__(self, state):
        superstate, d = state
        self._order = d['order']
        self._bos = d['bos']
        self._eos = d['eos']
        self._path = d['path']
        self._features = d['features']
        self._load_model()
        super(KenLM,self).__setstate__(superstate)

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

    cpdef tuple features(self):
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

    @classmethod
    def construct(cls, int uid, str name, str cfgstr):
        cdef int order
        cdef str path
        cdef bos = DEFAULT_BOS_STRING
        cdef eos = DEFAULT_EOS_STRING
        cfgstr, value = re_key_value('order', cfgstr, optional=False)
        if value:
            order = int(value)
        cfgstr, value = re_key_value('path', cfgstr, optional=False)
        if value:
            path = value
        cfgstr, value = re_key_value('bos', cfgstr, optional=True)
        if value:
            bos = value
        cfgstr, value = re_key_value('eos', cfgstr, optional=True)
        if value:
            eos = value
        return KenLM(uid, name, order, path, Terminal(bos), Terminal(eos))

    @staticmethod
    def help():
        help_msg = ["# Language model features from kenlm.",
                    "# This produces two scores: LM probability and OOV count.",
                    "# Requires: order=int path=str",
                    "# Optional: bos=str eos=str"]
        return '\n'.join(help_msg)

    @staticmethod
    def example():
        return 'KenLM order=3 path=trigrams.klm bos=<s> eos=</s>'




cdef class ConstantLM(Stateful):

    def __init__(self, int uid,
                 str name,
                 float constant=0.0):
        super(ConstantLM, self).__init__(uid, name)
        self._constant = constant

    def __getstate__(self):
        return super(ConstantLM,self).__getstate__(), {'constant': self._constant}

    def __setstate__(self, state):
        superstate, d = state
        self._constant = d['constant']
        super(ConstantLM,self).__setstate__(superstate)

    def __repr__(self):
        return '{0}(uid={1}, name={2}, constant={3})'.format(ConstantLM.__name__,
                                                             repr(self.id),
                                                             repr(self.name),
                                                             repr(self._constant))

    cpdef tuple fnames(self, wkeys):
        return tuple([self.name])

    cpdef tuple features(self):
        return tuple([self.name])

    cpdef FRepr weights(self, dict wmap):  # using dense representation
        cdef float fvalue = 0.0
        try:
            fvalue = wmap[self.name]
        except KeyError:
            raise KeyError('Missing ConstantLM feature: %s' % self.name)
        return FValue(fvalue)

    cpdef object initial(self):
        return 0

    cpdef object final(self):
        return 0

    cpdef FRepr featurize_initial(self):
        return FValue(self._constant)

    cpdef FRepr featurize_final(self, context):
        """
        :param context: a state
        :return:
        """
        return FValue(0.0)

    cpdef StatefulFRepr featurize(self, word, context):
        return StatefulFRepr(FValue(0.0), 0)

    cpdef FRepr featurize_yield(self, derivation_yield):
        """
        :param words: sequence of Terminal objects
        :return: weight
        """
        return FValue(self._constant)

    cpdef FRepr constant(self, weight_t value):
        return FValue(value)

    @classmethod
    def construct(cls, int uid, str name, str cfgstr):
        cdef float constant = 0.0
        cfgstr, value = re_key_value('constant', cfgstr, optional=True)
        if value:
            constant = float(value)
        return ConstantLM(uid, name, constant)

    @staticmethod
    def help():
        help_msg = ["# A constant for each yield.",
                    "# This is a dummy stateful feature that illustrates the Stateful interface."]
        return '\n'.join(help_msg)

    @staticmethod
    def example():
        return 'ConstantLM name=DummyStateful constant=1.0'