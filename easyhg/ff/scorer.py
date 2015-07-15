"""
:Authors: - Wilker Aziz
"""
from .state import StateMapper


class TableLookupScorer(object):

    def __init__(self, model):
        self._model = model

    def score(self, rule):
        """
        Score associated with a transition.
        :param rule:
        :return: weight
        """
        fvecs = [scorer.featurize(rule) for scorer in self._model.lookup]
        return self._model.lookup_score(fvecs)


class StatelessScorer(object):
    """
    Stateful scorers manage states of different nature.
    This class abstracts away these differences. It also abstracts away the number of scorers.
    Basically it maps a number of states (one from each scorer) onto a single integer (much like a state in an FSA).
    """

    def __init__(self, model):
        """
        :param scorers: sequence of Stateful objects
        :return:
        """
        self._model = model

    def score(self, edge):
        """
        Score associated with a transition.
        :param edge:
        :return: weight
        """
        freprs = [scorer.featurize(edge) for scorer in self._model.stateless]
        return self._model.stateless_score(freprs)


class StatefulScorer(object):
    """
    Stateful scorers manage states of different nature.
    This class abstracts away these differences. It also abstracts away the number of scorers.
    Basically it maps a number of states (one from each scorer) onto a single integer (much like a state in an FSA).
    """

    def __init__(self, model):
        """
        :param scorers: sequence of Stateful objects
        :return:
        """
        self._scorers = model.stateful
        self._mapper = StateMapper()
        self._initial = self._mapper.to_int(tuple(score.initial() for score in self._scorers))
        self._final = self._mapper.final
        self._model = model

    def initial(self):
        """The initial (integer) state."""
        return self._initial

    def final(self):
        """The final (integer) state."""
        return self._final

    def initial_score(self):
        """
        Score associated with the initial state.
        :returns: weight
        """
        fvecs = [None] * len(self._scorers)
        for i, scorer in enumerate(self._scorers):
            fvecs[i] = scorer.featurize_initial()
        return self._model.stateful_score(fvecs)

    def final_score(self, context):
        """
        Score associated with a transition to the final state.
        :param context: the origin state.
        :returns: weight
        """
        fvecs = [None] * len(self._scorers)
        in_states = self._mapper[context]
        for i, scorer in enumerate(self._scorers):
            fvecs[i] = scorer.featurize_final(context=in_states[i])
        return self._model.stateful_score(fvecs)

    def score(self, word, context):
        """
        Score associated with a transition.
        :param word: the label of the transition.
        :param context: the origin state.
        :return: weight, destination state.
        """
        fvecs = [None] * len(self._scorers)
        out_states = [None] * len(self._scorers)
        in_states = self._mapper[context]
        for i, scorer in enumerate(self._scorers):
            fvec, out_state = scorer.featurize(word, context=in_states[i])
            fvecs[i] = fvec
            out_states[i] = out_state

        return self._model.stateful_score(fvecs), self._mapper.to_int(tuple(out_states))

    def score_derivation(self, derivation):
        fvecs = [None] * len(self._scorers)
        for i, scorer in enumerate(self._scorers):
            fvecs[i] = scorer.featurize_derivation(derivation)
        return self._model.stateful_score(fvecs)


def apply_scorers(d, stateless, stateful, semiring, do_nothing):
    w = semiring.one
    # stateless scorer goes edge by edge
    # TODO: stateless.total_score(derivation)
    for edge in filter(lambda e: e.lhs not in do_nothing, d):
        w = semiring.times(w, stateless.score(edge))
    w = semiring.times(w, stateful.score_derivation(d))
    return w