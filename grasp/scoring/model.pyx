from grasp.scoring.extractor cimport Extractor, TableLookup, Stateless, Stateful
from grasp.ptypes cimport weight_t

import itertools


cdef class Model:

    def __init__(self, dict wmap, extractors):
        cdef Extractor extractor
        self._extractors = tuple(extractors)
        self._wmap = dict(wmap)
        self._weights = FComponents([extractor.weights(self._wmap) for extractor in self._extractors])

    def __getstate__(self):
        return {'extractors': self._extractors, 'wmap': self._wmap}

    def __setstate__(self, state):
        self._extractors = tuple(state['extractors'])
        self._wmap = dict(state['wmap'])
        self._weights = FComponents([extractor.weights(self._wmap) for extractor in self._extractors])

    cpdef tuple extractors(self):
        return self._extractors

    cpdef weight_t score(self, FComponents freprs) except *:
        return freprs.dot(self._weights)

    cpdef tuple fnames(self, wkey=None):
        cdef Extractor extractor
        cdef list names = []
        if wkey is None:
            wkey = self._wmap.keys()
        for extractor in self._extractors:
            names.extend(extractor.fnames(wkey))
        return tuple(names)

    cpdef FComponents weights(self):
        return self._weights

    cpdef FComponents constant(self, weight_t value):
        cdef Extractor extractor
        return FComponents([extractor.constant(value) for extractor in self._extractors])

    def __str__(self):
        return '\n'.join('{0} ||| {1}'.format(repr(ex), w) for ex, w in zip(self._extractors, self._weights))


cdef class DummyModel(Model):

    def __init__(self):
        super(DummyModel, self).__init__(dict(), [])


cdef class ModelContainer(Model):

    def __init__(self, wmap, lookup_extractors, stateless_extractors, stateful_extractors):
        super(ModelContainer, self).__init__(wmap,
                                             itertools.chain(lookup_extractors,
                                                             stateless_extractors,
                                                             stateful_extractors))

        self.lookup = Model(wmap, lookup_extractors)
        self.stateless = Model(wmap, stateless_extractors)
        self.stateful = Model(wmap, stateful_extractors)
        self.dummy = DummyModel()
        self._weights = FComponents([self.lookup.weights(), self.stateless.weights(), self.stateful.weights()])

    def __getstate__(self):
        return super(ModelContainer, self).__getstate__(), {'lookup': self.lookup,
                                                            'stateless': self.stateless,
                                                            'stateful': self.stateful,
                                                            'dummy': self.dummy}

    def __setstate__(self, state):
        superstate, selfstate = state
        self.lookup = selfstate['lookup']
        self.stateless = selfstate['stateless']
        self.stateful = selfstate['stateful']
        self.dummy = selfstate['dummy']
        super(ModelContainer,self).__setstate__(superstate)
        self._weights = FComponents([self.lookup.weights(), self.stateless.weights(), self.stateful.weights()])

    def __str__(self):
        return '# Lookup\n{0}\n# Stateless\n{1}\n# Stateful\n{2}'.format(str(self.lookup), str(self.stateless), str(self.stateful))

    cpdef FComponents constant(self, weight_t value):
        cdef Extractor extractor
        return FComponents([self.lookup.constant(value), self.stateless.constant(value), self.stateful.constant(value)])

    cpdef itercomponents(self):
        return iter([self.lookup.extractors(), self.stateless.extractors(), self.stateful.extractors()])