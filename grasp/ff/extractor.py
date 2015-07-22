"""
This module contains definitions for scorers.

Highlights:
    - when implementing a stateful scorer you should inherit from Stateful
    - when using your scorer with programs such as Earley and Nederhof, remember to wrap it using StatefulScorerWrapper
        this will basically abstract away implementation details such as the nature of states.

:Authors: - Wilker Aziz
"""


class Extractor(object):

    def __init__(self, uid, name):
        self._uid = uid
        self._name = name

    @property
    def id(self):
        return self._uid

    @property
    def name(self):
        return self._name

    def weights(self, wmap):
        return NotImplementedError('I have not been implemented!')

    def dot(self, frepr, wrepr):
        return NotImplementedError('I have not been implemented!')


class TableLookup(Extractor):

    def __init__(self, uid, name):
        super(TableLookup, self).__init__(uid, name)

    def featurize(self, rule):
        raise NotImplementedError('I have not been implemented!')


class Stateless(Extractor):

    def __init__(self, uid, name):
        super(Stateless, self).__init__(uid, name)

    def featurize(self, edge):
        raise NotImplementedError('I have not been implemented!')


class Stateful(Extractor):
    """
    Basic interface for stateful scorers.
    """

    def __init__(self, uid, name):
        super(Stateful, self).__init__(uid, name)

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

    def featurize_initial(self):
        """
        Score associated with the initial state.
        :return:
        """
        raise NotImplementedError('I have not been implemented!')

    def featurize_final(self, context):
        """
        Score associated with a transition to the final state.
        :param context: a state
        :return: feature representation
        """
        raise NotImplementedError('I have not been implemented!')

    def featurize(self, word, context):  # TODO: pass edge and position of the dot instead of word
        """
        Return the score and the next state.
        :param word: a Terminal
        :param context: a state
        :returns: feature representation, state
        """
        raise NotImplementedError('I have not been implemented!')

    def featurize_derivation(self, derivation):
        """
        Featurize a derivation (as a sequence of edges).
        :param derivation:
        :return:
        """
        raise NotImplementedError('I have not been implemented!')