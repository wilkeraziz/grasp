"""
This module contains definitions for scorers.

Highlights:
    - when implementing a stateful scorer you should inherit from Stateful
    - when using your scorer with programs such as Earley and Nederhof, remember to wrap it using StatefulScorerWrapper
        this will basically abstract away implementation details such as the nature of states.

:Authors: - Wilker Aziz
"""

import numpy as np
from .state import StateMapper


class Scorer(object):

    def __init__(self, uid, name, weights):
        self._uid = uid
        self._name = name
        self._weights = np.array(weights, float)

    @property
    def id(self):
        return self._uid

    @property
    def name(self):
        return self._name

    @property
    def weights(self):
        return self._weights


class Stateless(Scorer):

    def __init__(self, uid, name, weights):
        super(Stateless, self).__init__(uid, name, weights)

    def score(self, rule):
        raise NotImplementedError('I have not been implemented!')


class Stateful(Scorer):
    """
    Basic interface for stateful scorers.
    """

    def __init__(self, uid, name, weights):
        super(Stateful, self).__init__(uid, name, weights)

    def initial(self):
        """
        Return the initial state.
        :return:
        """
        raise NotImplementedError('I have not been implemented!')

    def final(self):
        """
        Return the final state.
        :return:
        """
        raise NotImplementedError('I have not been implemented!')

    def initial_score(self):
        """
        Score associated with the initial state.
        :return:
        """
        raise NotImplementedError('I have not been implemented!')

    def final_score(self, context):
        """
        Score associated with a transition to the final state.
        :param context: a state
        :return: weight
        """
        raise NotImplementedError('I have not been implemented!')

    def score(self, word, context):
        """
        Return the score and the next state.
        :param word: a Terminal
        :param context: a state
        :returns: weight, state
        """
        raise NotImplementedError('I have not been implemented!')


class StatefulScorerWrapper(object):
    """
    Stateful scorers manage states of different nature.
    This class abstracts away these differences. It also abstracts away the number of scorers.
    Basically it maps a number of states (one from each scorer) onto a single integer (much like a state in an FSA).
    """

    def __init__(self, scorers):
        """
        :param scorers: sequence of Stateful objects
        :return:
        """
        self._scorers = tuple(scorers)
        self._mapper = StateMapper()
        self._initial = self._mapper.to_int(tuple(score.initial() for score in self._scorers))
        self._final = self._mapper.final

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
        weights = np.zeros(len(self._scorers))
        for i, scorer in enumerate(self._scorers):
            weights[i] = scorer.initial_score()
        return weights.sum()

    def final_score(self, context):
        """
        Score associated with a transition to the final state.
        :param context: the origin state.
        :returns: weight
        """
        weights = np.zeros(len(self._scorers))
        in_states = self._mapper[context]
        for i, scorer in enumerate(self._scorers):
            weights[i] = scorer.final_score(context=in_states[i])
        return weights.sum()

    def score(self, word, context):
        """
        Score associated with a transition.
        :param word: the label of the transition.
        :param context: the origin state.
        :return: weight, destination state.
        """
        weights = np.zeros(len(self._scorers))
        out_states = [None] * len(self._scorers)
        in_states = self._mapper[context]
        for i, scorer in enumerate(self._scorers):
            weight, out_state = scorer.score(word, context=in_states[i])
            weights[i] = weight
            out_states[i] = out_state
        return weights.sum(), self._mapper.to_int(tuple(out_states))