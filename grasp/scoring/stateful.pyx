"""
:Authors: - Wilker Aziz
"""


cdef class StatefulFRepr:
    """
    Wraper for a feature representation object (FRepr) and a State object to be used by stateful extractors.
    """

    def __cinit__(self, FRepr frepr, object state):
        self.frepr = frepr
        self.state = state

    def __getitem__(self, int i):
        if i == 0:
            return self.frepr
        elif i == 1:
            return self.state
        else:
            raise IndexError('StatefulReturn can only be indexed by 0 or 1')

    def __iter__(self):
        return iter([self.frepr, self.state])

    def __str__(self):
        return '(%s, %s)' % (self.frepr, self.state)

    def __repr__(self):
        return '(%r, %r)' % (self.frepr, self.state)


cdef class Stateful(Extractor):
    """
    Basic interface for stateful scorers.
    """

    def __init__(self, int uid, str name):
        super(Stateful, self).__init__(uid, name)

    cpdef initial(self):
        """
        Return the initial state.
        :return:
        """
        pass

    cpdef final(self):
        """
        Return the final state.
        :return:
        """
        pass

    cpdef FRepr featurize_initial(self):
        """
        Score associated with the initial state.
        :return:
        """
        pass

    cpdef FRepr featurize_final(self, context):
        """
        Score associated with a transition to the final state.
        :param context: a state
        :return: feature representation
        """
        pass

    cpdef StatefulFRepr featurize(self, word, context):  # TODO: pass edge and position of the dot instead of word
        """
        Return the score and the next state.
        :param word: a Terminal
        :param context: a state
        :returns: feature representation, state
        """
        pass

    cpdef FRepr featurize_derivation(self, derivation):
        """
        Featurize a derivation (as a sequence of edges).
        :param derivation:
        :return:
        """
        pass
