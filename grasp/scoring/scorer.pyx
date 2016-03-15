"""
:Authors: - Wilker Aziz
"""

from grasp.ptypes cimport weight_t, id_t
from grasp.scoring.state cimport StateMapper
from grasp.scoring.extractor cimport FRepr, Extractor, TableLookup, Stateless, Stateful, StatefulFRepr
from grasp.scoring.model cimport Model


cdef class Scorer:
    """
    Scorers wrap feature extractors.
    They collect feature representations (by calling feature extractors)
        and pack them into score components (a container for feature representation objects).
    They can also apply a model to a score component producing a final real-valued score.
    """

    def __init__(self, Model model):
        self._model = model

    def __bool__(self):
        return bool(self._model.extractors())

    cpdef tuple extractors(self):
        return self._model.extractors()

    cpdef Model model(self):
        return self._model

    cpdef FComponents constant(self, weight_t value):
        cdef Extractor extractor
        return FComponents([extractor.constant(value) for extractor in self.extractors()])

    def __str__(self):
        return '{0} ||| {1}'.format(self.__class__.__name__, str(self._model))


cdef class TableLookupScorer(Scorer):

    def __init__(self, Model model): # TODO: currently use only all extractors of a certain kind or none of them, change this
        super(TableLookupScorer, self).__init__(model)

    cpdef FComponents featurize(self, rule):
        """
        Score associated with a transition.
        :param rule:
        :return: weight
        """
        cdef TableLookup extractor
        return FComponents([extractor.featurize(rule) for extractor in self.extractors()])

    cpdef weight_t score(self, rule):
        """
        Score associated with a transition.
        :param rule:
        :return: weight
        """
        cdef TableLookup extractor
        return self._model.score(self.featurize(rule))

    cpdef tuple featurize_and_score(self, rule):
        cdef FComponents frepr = self.featurize(rule)
        return frepr, self._model.score(frepr)

    cpdef tuple featurize_and_score_derivation(self, tuple rules, Semiring semiring):
        cdef FComponents comp, partial_comp
        cdef weight_t weight, partial_weight
        cdef Rule r
        weight = semiring.one
        comp = self.constant(semiring.one)
        for r in rules:
            partial_comp, partial_weight = self.featurize_and_score(r)
            weight = semiring.times(weight, partial_weight)
            comp = comp.hadamard(partial_comp, semiring.times)
        return comp, weight


cdef class StatelessScorer(Scorer):
    """
    Stateful scorers manage states of different nature.
    This class abstracts away these differences. It also abstracts away the number of scorers.
    Basically it maps a number of states (one from each scorer) onto a single integer (much like a state in an FSA).
    """

    def __init__(self, Model model): # TODO: currently use only all extractors of a certain kind or none of them, change this
        """
        :param scorers: sequence of Stateful objects
        :return:
        """
        super(StatelessScorer, self).__init__(model)

    cpdef FComponents featurize(self, edge):
        """
        Score associated with a transition.
        :param edge: a Rule
        :return: weight
        """
        cdef Stateless extractor
        return FComponents([extractor.featurize(edge) for extractor in self.extractors()])

    cpdef weight_t score(self, edge):
        """
        Score associated with a transition.
        :param edge: a Rule
        :return: weight
        """
        return self._model.score(self.featurize(edge))

    cpdef tuple featurize_and_score(self, edge):
        cdef FComponents frepr = self.featurize(edge)
        return frepr, self._model.score(frepr)

    cpdef tuple featurize_and_score_derivation(self, tuple edges, Semiring semiring):
        cdef FComponents comp, partial_comp
        cdef weight_t weight, partial_weight
        cdef Rule r
        weight = semiring.one
        comp = self.constant(semiring.one)
        for r in edges:
            partial_comp, partial_weight = self.featurize_and_score(r)
            weight = semiring.times(weight, partial_weight)
            comp = comp.hadamard(partial_comp, semiring.times)
        return comp, weight


cdef class StatefulScorer(Scorer):
    """
    Stateful scorers manage states of different nature.
    This class abstracts away these differences. It also abstracts away the number of scorers.
    Basically it maps a number of states (one from each scorer) onto a single integer (much like a state in an FSA).
    """

    def __init__(self, Model model): # TODO: currently use only all extractors of a certain kind or none of them, change this
        """
        :param scorers: sequence of Stateful objects
        :return:
        """
        super(StatefulScorer, self).__init__(model)
        self._mapper = StateMapper()
        self._initial = self._mapper.id(tuple(score.initial() for score in self.extractors()))
        self._final = self._mapper.final

    cpdef id_t initial(self):
        """The initial (integer) state."""
        return self._initial

    cpdef id_t final(self):
        """The final (integer) state."""
        return self._final

    cpdef FComponents featurize_initial(self):
        cdef Stateful extractor
        return FComponents([extractor.featurize_initial() for extractor in self.extractors()])

    cpdef weight_t initial_score(self):
        """
        Score associated with the initial state.
        :returns: weight
        """
        return self._model.score(self.featurize_initial())

    cpdef FComponents featurize_final(self, context):
        cdef list freprs = [None] * len(self.extractors())
        cdef tuple in_states = self._mapper.state(context)
        cdef size_t i = 0
        cdef Stateful extractor
        for extractor in self.extractors():
            freprs[i] = extractor.featurize_final(context=in_states[i])
            i += 1
        return FComponents(freprs)

    cpdef weight_t final_score(self, context):
        """
        Score associated with a transition to the final state.
        :param context: the origin state.
        :returns: weight
        """
        return self._model.score(self.featurize_final(context))

    cpdef tuple featurize(self, word, context):
        """
        Score associated with a transition.
        :param word: the label of the transition.
        :param context: the origin state.
        :return: weight, destination state.
        """
        cdef:
            list freprs = [None] * len(self.extractors())
            list out_states = [None] * len(self.extractors())
            tuple in_states = self._mapper[context]
            Stateful extractor
            StatefulFRepr r
            size_t i = 0
        #print('Context: {0}'.format(context))
        for extractor in self.extractors():
            #print(' Extractor {0}: {1}'.format(i, extractor))
            x = in_states[i]
            r = extractor.featurize(word, context=x)
            freprs[i] = r.frepr
            out_states[i] = r.state
            i += 1
        return FComponents(freprs), self._mapper.id(tuple(out_states))

    cpdef tuple score(self, word, context):
        cdef:
            FComponents comp
            object state
        comp, state = self.featurize(word, context)
        return self._model.score(comp), state

    cpdef tuple featurize_and_score(self, word, context):
        cdef:
            FComponents comp
            object state
        comp, state = self.featurize(word, context)
        return comp, self._model.score(comp), state

    cpdef FComponents featurize_yield(self, derivation_yield):
        cdef Stateful extractor
        return FComponents([extractor.featurize_yield(derivation_yield) for extractor in self.extractors()])

    cpdef weight_t score_yield(self, derivation_yield):
        return self._model.score(self.featurize_yield(derivation_yield))

    cpdef tuple featurize_and_score_yield(self, derivation_yield):
        """
        Featurize a derivation yield and score it.
        :param derivation_yield:
        :return: component, score
        """
        cdef:
            FComponents comp = self.featurize_yield(derivation_yield)
        return comp, self._model.score(comp)