"""
This module contains definitions for scorers.

Highlights:
    - when implementing a stateful scorer you should inherit from Stateful
    - when using your scorer with programs such as Earley and Nederhof, remember to wrap it using StatefulScorerWrapper
        this will basically abstract away implementation details such as the nature of states.

:Authors: - Wilker Aziz
"""
cimport numpy as np
import numpy as np
from grasp.ptypes cimport weight_t
import grasp.ptypes as ptypes


cdef class FRepr:

    cpdef weight_t dot(self, FRepr w): pass


cdef class FValue(FRepr):

    def __init__(self, weight_t value):
        self.value = value

    cpdef weight_t dot(self, FRepr w) except *:
        return self.value * (<FValue?>w).value

    def __len__(self):
        return 1

    def __iter__(self):
        return iter([self.value])

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return repr(self.value)


cdef class FVec(FRepr):

    def __init__(self, iterable):
        self.vec = np.array(iterable, dtype=ptypes.weight)

    cpdef weight_t dot(self, FRepr w) except *:
        return np.dot(self.vec, (<FVec?>w).vec)

    def __len__(self):
        return len(self.vec)

    def __iter__(self):
        return iter(self.vec)

    def __str__(self):
        return ' '.join(str(x) for x in self.vec)

    def __repr__(self):
        return repr(self.vec)


cdef class FMap(FRepr):

    def __init__(self, iterable):
        self.map = dict(iterable)

    cpdef weight_t dot(self, FRepr w) except *:
        cdef weight_t v
        cdef dict wmap = (<FMap?>w).map
        if len(self.map) < len(wmap):
            return np.sum([v * wmap.get(k, 0.0) for k, v in self.map.items()])
        else:
            return np.sum([v * self.map.get(k, 0.0) for k, v in wmap.items()])

    def __len__(self):
        return len(self.map)

    def __iter__(self):
        return iter(self.map.items())

    def __str__(self):
        return ' '.join('{0}={1}'.format(k, v) for k, v in self.map.items())

    def __repr__(self):
        return repr(self.map)


cdef class Extractor:

    def __init__(self, int uid, str name):
        self._uid = uid
        self._name = name

    property id:
        def __get__(self):
            return self._uid

    property name:
        def __get__(self):
            return self._name

    cpdef FRepr weights(self, dict wmap): pass

    cpdef weight_t dot(self, FRepr frepr, FRepr wrepr):
        return frepr.dot(wrepr)
