"""
:Authors: - Wilker Aziz
"""

import numpy as np


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
        :param word: a Terminal
        :param context: a state
        :returns: weight
        """
        raise NotImplementedError('I have not been implemented!')

    def next(self, word, context):
        """
        Return the next state.
        :param word: a Terminal
        :param context: a state
        :returns: state
        """
        raise NotImplementedError('I have not been implemented!')

    def get_state(self, key):
        """
        Return the state associated with a certain key.
        """
        raise NotImplementedError('I have not been implemented!')