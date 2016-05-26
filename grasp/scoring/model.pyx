from grasp.scoring.extractor cimport TableLookup, Stateless, Stateful
from grasp.ptypes cimport weight_t

import itertools


cdef class Model:

    def __init__(self, dict wmap, extractors):
        cdef Extractor extractor
        cdef size_t i
        self._extractors = tuple(extractors)
        self._wmap = dict(wmap)
        self._weights = FComponents([extractor.weights(self._wmap) for extractor in self._extractors])
        self._name_to_position = {extractor.name: i for i, extractor in enumerate(self._extractors)}

    def __bool__(self):
        return bool(self._extractors)

    def __len__(self):
        return len(self._extractors)

    def __getstate__(self):
        return {'extractors': self._extractors, 'wmap': self._wmap}

    cpdef Extractor get_extractor(self, name):
        cdef int i = self._name_to_position.get(name, -1)
        if i >= 0:
            return self._extractors[i]
        else:
            return None

    cpdef int get_position(self, name):
        return self._name_to_position.get(name, -1)

    property wmap:
        def __get__(self):
            return self._wmap

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

    def __init__(self, wmap, extractors):

        cdef:
            list lookup_extractors = []
            list stateless_extractors = []
            list stateful_extractors = []
            Extractor extractor
        # separate by type of extractor
        for extractor in extractors:
            if isinstance(extractor, TableLookup):
                lookup_extractors.append(extractor)
            elif isinstance(extractor, Stateless):
                stateless_extractors.append(extractor)
            elif isinstance(extractor, Stateful):
                stateful_extractors.append(extractor)
            else:  # ignoring extractor that does not implement adequate interfaces
                pass  # TODO: warn user

        # create a model with all extractors sorted as (lookup, stateless, stateful)
        super(ModelContainer, self).__init__(wmap,
                                             itertools.chain(lookup_extractors,
                                                             stateless_extractors,
                                                             stateful_extractors))

        # create a model for each type of extractor
        self.lookup = Model(wmap, lookup_extractors)
        self.stateless = Model(wmap, stateless_extractors)
        self.stateful = Model(wmap, stateful_extractors)
        self.dummy = DummyModel()
        #self._weights = FComponents([self.lookup.weights(), self.stateless.weights(), self.stateful.weights()])

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
        #self._weights = FComponents([self.lookup.weights(), self.stateless.weights(), self.stateful.weights()])

    def __str__(self):
        return '# Lookup\n{0}\n# Stateless\n{1}\n# Stateful\n{2}'.format(str(self.lookup), str(self.stateless), str(self.stateful))

    cpdef FComponents constant(self, weight_t value):
        cdef Extractor extractor
        # TODO: should I skip empty models? then I have to be consistent elsewhere (e.g. weights(self, wmap))
        #return FComponents([model.constant(value) for model in [self.lookup, self.stateless, self.stateful] if model])
        return FComponents([self.lookup.constant(value), self.stateless.constant(value), self.stateful.constant(value)])

    cpdef itercomponents(self):
        return iter([self.lookup.extractors(), self.stateless.extractors(), self.stateful.extractors()])


cdef class ModelView(ModelContainer):

    def __init__(self, wmap, extractors, local_names=set(), nonlocal_names=set()):
        """
        A model container where we can change the notion of locality and nonlocality.
        By default Stateful components are nonlocal, all other components are local.

        :param wmap: weight map
        :param extractors: all extractors
        :param local_names: overwrites default behaviour specifying local names
        :param nonlocal_names: overwrites default behaviou specifying nonlocal names
        """
        super(ModelView, self).__init__(wmap, extractors)
        # separate in Views
        cdef:
            list local_extractors = []
            list nonlocal_extractors = []
            Extractor extractor
        for extractor in extractors:
            if extractor.name in local_names:  # user wants this to be local
                local_extractors.append(extractor)
            elif extractor.name in nonlocal_names:  # user wants this to be nonlocal
                nonlocal_extractors.append(extractor)
            else:  # user accepts default behaviour
                if isinstance(extractor, TableLookup):  # this is local
                    local_extractors.append(extractor)
                elif isinstance(extractor, Stateless):  # this is local
                    local_extractors.append(extractor)
                else:  # Stateful extractors are nonlocal by default
                    nonlocal_extractors.append(extractor)

        self._local = ModelContainer(wmap, local_extractors)
        self._nonlocal = ModelContainer(wmap, nonlocal_extractors)

    def __getstate__(self):
        return super(ModelView, self).__getstate__(), {'local': self._local, 'nonlocal': self._nonlocal}

    def __setstate__(self, state):
        superstate, selfstate = state
        self._local = selfstate['local']
        self._nonlocal = selfstate['nonlocal']
        super(ModelView,self).__setstate__(superstate)

    def __str__(self):
        return '[LOCAL]\n{0}\n\n[NONLOCAL]\n{1}\n'.format(self._local, self._nonlocal)

    cpdef FComponents merge(self, FComponents local_comps, FComponents nonlocal_comps):
        cdef dict name_to_comp = {}
        for extractor, comps in zip(self._local.extractors(), local_comps):  # this is the local order
            name_to_comp[extractor.name] = comps
        for extractor, comps in zip(self._nonlocal.extractors(), nonlocal_comps):  # this is the nonlocal order
            name_to_comp[extractor.name] = comps
        cdef:
            tuple extractors = self.extractors()
            list ordered = [None] * len(extractors)
            size_t i
        for i in range(len(extractors)):  # this the correct order!
            ordered[i] = name_to_comp[extractors[i].name]
        return FComponents(ordered)

    cpdef ModelContainer local_model(self):
        return self._local

    cpdef ModelContainer nonlocal_model(self):
        return self._nonlocal