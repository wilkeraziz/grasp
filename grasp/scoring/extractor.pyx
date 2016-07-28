"""
This module contains definitions for feature extractors.

:Authors: - Wilker Aziz
"""
from grasp.ptypes cimport weight_t


cdef class Extractor:
    """
    An Extractor is capable of featurizing rules/edges/strings/derivations.
    """

    def __init__(self, int uid, str name):
        self._uid = uid
        self._name = name

    property id:
        def __get__(self):
            return self._uid

    property name:
        def __get__(self):
            return self._name

    def __getstate__(self):
        return {'uid': self._uid, 'name': self._name}

    def __setstate__(self, d):
        self._uid = d['uid']
        self._name = d['name']

    cpdef tuple fnames(self, wkeys):
        """
        Return an order list of feature names associated with this extractor.
        Some extractors (e.g. dense ones) will ignore whatever argument passed to this method.
        Others (e.g. sparse ones) will consult wkeys for features matching a certain prefix.
        """
        raise NotImplementedError('I do not know how to convert a dict to an order list of features')

    cpdef tuple features(self):
        raise NotImplementedError('I do not know which features I can create')

    cpdef FRepr weights(self, dict wmap):
        raise NotImplementedError('I do not know how to convert a dict to a weight representation.')

    cpdef weight_t dot(self, FRepr frepr, FRepr wrepr):
        return frepr.dot(wrepr)

    cpdef FRepr constant(self, weight_t value):
        """
        Return a constant feature representation (this is useful to get a vector/map of zeros/ones
        of appropriate size.
        """
        raise NotImplementedError('I do not know which type of FRepr to create.')

    cpdef str cfg(self):
        return NotImplementedError('I have not been configured.')

    @classmethod
    def construct(cls, int uid, str name, str cfgstr):
        raise NotImplementedError('I do not know how to construct an extractor from a string')

    @staticmethod
    def help():
        return "My developer didn't care to leave a help message."

    @staticmethod
    def example():
        return "My developer didn't care to leave an example."


cdef class TableLookup(Extractor):
    """
    This is the simplest feature extractor.
    It applies to rules directly.
    """

    def __init__(self, int uid, str name):
        super(TableLookup, self).__init__(uid, name)

    cpdef FRepr featurize(self, rule):
        raise NotImplementedError('I do not know how to featurize a rule')


cdef class Stateless(Extractor):
    """
    Stateless extractor score edges individually and do not return state information.
    """

    def __init__(self, int uid, str name):
        super(Stateless, self).__init__(uid, name)

    cpdef FRepr featurize(self, edge):
        raise NotImplementedError('I do not know how to featurize an edge')


cdef class StatefulFRepr:
    """
    Wraper for a feature representation object (FRepr) and a State object to be used by stateful extractors.
    """

    def __cinit__(self, FRepr frepr, object state):
        self.frepr = frepr
        self.state = state

    def __getstate__(self):
        return {'frepr': self.frepr, 'state': self.state}

    def __setstate__(self, d):
        self.frepr = d['frepr']
        self.state = d['state']

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
    Stateful extractors are those that score edges in context and update state information.
    """

    def __init__(self, int uid, str name):
        super(Stateful, self).__init__(uid, name)

    cpdef object initial(self):
        """
        Return the initial state.
        :return:
        """
        raise NotImplementedError('I do not know what an initial state looks like.')

    cpdef object final(self):
        """
        Return the final state.
        :return:
        """
        raise NotImplementedError('I do not know what a final state looks like.')

    cpdef FRepr featurize_initial(self):
        """
        Score associated with the initial state.
        :return:
        """
        raise NotImplementedError('I cannot featurize initial states.')

    cpdef FRepr featurize_final(self, context):
        """
        Score associated with a transition to the final state.
        :param context: a state
        :return: feature representation
        """
        raise NotImplementedError('I cannot featurize final states.')

    cpdef StatefulFRepr featurize(self, word, context):  # TODO: pass edge and position of the dot instead of word
        """
        Return the score and the next state.
        :param word: a Terminal
        :param context: a state
        :returns: feature representation, state
        """
        raise NotImplementedError('I cannot featurize words in context.')

    cpdef FRepr featurize_yield(self, derivation_yield):
        """
        Featurize a derivation (as a sequence of edges).
        :param derivation:
        :return:
        """
        raise NotImplementedError('I cannot featurize the yield of a derivation.')


