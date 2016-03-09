"""
:Authors: - Wilker Aziz
"""

from grasp.ptypes cimport weight_t, id_t
from grasp.scoring.state cimport StateMapper
from grasp.scoring.extractor cimport FRepr
from grasp.scoring.lookup cimport TableLookup
from grasp.scoring.stateless cimport Stateless
from grasp.scoring.stateful cimport Stateful, StatefulFRepr


cdef class LogLinearModel:

    def __init__(self, wmap, extractors):
        self._wmap = dict(wmap)
        # all scorers sorted by id
        self._extractors = tuple(sorted(extractors, key=lambda scorer: scorer.id))
        # lookup ones
        self._lookup = tuple(filter(lambda s: isinstance(s, TableLookup), self._extractors))
        # stateless ones
        self._stateless = tuple(filter(lambda s: isinstance(s, Stateless), self._extractors))
        # stateful ones
        self._stateful = tuple(filter(lambda s: isinstance(s, Stateful), self._extractors))

        # memorise the a weight representation for each extractor
        self._lookup_weights = tuple(extractor.weights(self._wmap) for extractor in self._lookup)
        self._stateless_weights = tuple(extractor.weights(self._wmap) for extractor in self._stateless)
        self._stateful_weights = tuple(extractor.weights(self._wmap) for extractor in self._stateful)

    property lookup:
        def __get__(self):
            return self._lookup

    property stateless:
        def __get__(self):
            return self._stateless

    property stateful:
        def __get__(self):
            return self._stateful

    cpdef weight_t lookup_score(self, list freprs):
        cdef:
            size_t i = 0
            FRepr frepr
            weight_t total = 0
        for frepr in freprs:
            total += (<TableLookup>self._lookup[i]).dot(frepr, self._lookup_weights[i])
            i += 1
        return total

    cpdef weight_t stateless_score(self, list freprs):
        cdef:
            size_t i = 0
            FRepr frepr
            weight_t total = 0
        for frepr in freprs:
            total += (<Stateless>self._stateless[i]).dot(frepr, self._stateless_weights[i])
            i += 1
        return total

    cpdef weight_t stateful_score(self, list freprs):
        """
        Return the score (a linear combination) associated with a certain vector representation.
        :param fvecs: the features of each scorer
        :param scorers: scorers which produced the features
        :return: dot product
        """
        cdef:
            size_t i = 0
            FRepr frepr
            weight_t total = 0
        for frepr in freprs:
            total += (<Stateful>self._stateful[i]).dot(frepr, self._stateful_weights[i])
            i += 1
        return total


cdef class Scorer: pass


cdef class TableLookupScorer(Scorer):

    def __init__(self, LogLinearModel model):
        self._model = model

    cpdef weight_t score(self, rule):
        """
        Score associated with a transition.
        :param rule:
        :return: weight
        """
        cdef TableLookup extractor
        return self._model.lookup_score([extractor.featurize(rule) for extractor in self._model.lookup])


cdef class StatelessScorer(Scorer):
    """
    Stateful scorers manage states of different nature.
    This class abstracts away these differences. It also abstracts away the number of scorers.
    Basically it maps a number of states (one from each scorer) onto a single integer (much like a state in an FSA).
    """

    def __init__(self, LogLinearModel model):
        """
        :param scorers: sequence of Stateful objects
        :return:
        """
        self._model = model
        self._extractors = tuple(model.stateless)

    def __bool__(self):
        return bool(self._extractors)

    cpdef weight_t score(self, edge):
        """
        Score associated with a transition.
        :param edge: a Rule
        :return: weight
        """
        cdef Stateless extractor
        return self._model.stateless_score([extractor.featurize(edge) for extractor in self._extractors])


cdef class StatefulScorer(Scorer):
    """
    Stateful scorers manage states of different nature.
    This class abstracts away these differences. It also abstracts away the number of scorers.
    Basically it maps a number of states (one from each scorer) onto a single integer (much like a state in an FSA).
    """

    def __init__(self, LogLinearModel model):
        """
        :param scorers: sequence of Stateful objects
        :return:
        """
        self._model = model
        self._extractors = tuple(model.stateful)
        self._mapper = StateMapper()
        self._initial = self._mapper.id(tuple(score.initial() for score in self._extractors))
        self._final = self._mapper.final

    def __bool__(self):
        return bool(self._extractors)

    cpdef id_t initial(self):
        """The initial (integer) state."""
        return self._initial

    cpdef id_t final(self):
        """The final (integer) state."""
        return self._final

    cpdef weight_t initial_score(self):
        """
        Score associated with the initial state.
        :returns: weight
        """
        cdef Stateful extractor
        return self._model.stateful_score([extractor.featurize_initial() for extractor in self._extractors])

    cpdef weight_t final_score(self, context):
        """
        Score associated with a transition to the final state.
        :param context: the origin state.
        :returns: weight
        """
        cdef list fvecs = [None] * len(self._extractors)
        cdef tuple in_states = self._mapper.state(context)
        cdef size_t i = 0
        cdef Stateful extractor
        for extractor in self._extractors:
            fvecs[i] = extractor.featurize_final(context=in_states[i])
            i += 1
        return self._model.stateful_score(fvecs)

    cpdef tuple score(self, word, context):
        """
        Score associated with a transition.
        :param word: the label of the transition.
        :param context: the origin state.
        :return: weight, destination state.
        """
        cdef:
            list fvecs = [None] * len(self._extractors)
            list out_states = [None] * len(self._extractors)
            tuple in_states = self._mapper[context]
            Stateful extractor
            StatefulFRepr r
            size_t i = 0
        #print('Context: {0}'.format(context))
        for extractor in self._extractors:
            #print(' Extractor {0}: {1}'.format(i, extractor))
            x=in_states[i]
            r = extractor.featurize(word, context=x)
            fvecs[i] = r.frepr
            out_states[i] = r.state
            i += 1
        return self._model.stateful_score(fvecs), self._mapper.id(tuple(out_states))

    cpdef weight_t score_derivation(self, derivation):
        cdef Stateful extractor
        return self._model.stateful_score([extractor.featurize_derivation(derivation)
                                           for extractor in self._extractors])
