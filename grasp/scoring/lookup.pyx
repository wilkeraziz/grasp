"""
Table lookup extractors.

:Authors: - Wilker Aziz
"""
from grasp.scoring.extractor cimport FVec


CDEC_DEFAULT = 'Glue PassThrough EgivenFCoherent SampleCountF CountEF MaxLexFgivenE MaxLexEgivenF IsSingletonF IsSingletonFE'.split()


cdef class TableLookup(Extractor):

    def __init__(self, int uid, str name):
        super(TableLookup, self).__init__(uid, name)

    cpdef FRepr featurize(self, rule): pass


cdef class RuleTable(TableLookup):

    def __init__(self, int uid, str name, list fnames=CDEC_DEFAULT):
        super(RuleTable, self).__init__(uid, name)
        self._fnames = tuple(fnames)

    def __repr__(self):
        return '{0}(uid={1}, name={2}, fnames={3})'.format(RuleTable.__name__,
                                                           repr(self.id),
                                                           repr(self.name),
                                                           repr(self._fnames))

    cpdef FRepr weights(self, dict wmap):  # using a dense representation
        cdef list wvec = []
        for f in self._fnames:
            try:
                wvec.append(wmap[f])
            except KeyError:
                raise KeyError('Missing RuleTable feature: %s' % f)
        return FVec(wvec)

    cpdef FRepr featurize(self, rule):  # using a dense representation
        """
        :param rule: an SCFGProduction
        :return:
        """
        return FVec([rule.fvalue(fname, 0) for fname in self._fnames])