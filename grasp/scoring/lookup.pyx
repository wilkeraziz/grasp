"""
Table lookup extractors.

:Authors: - Wilker Aziz
"""
from grasp.scoring.frepr cimport FRepr, FVec
from grasp.ptypes cimport weight_t


CDEC_DEFAULT = 'Glue PassThrough EgivenFCoherent SampleCountF CountEF MaxLexFgivenE MaxLexEgivenF IsSingletonF IsSingletonFE'.split()


cdef class RuleTable(TableLookup):
    """
    A rule table much like those in cdec and Moses.
    Features are stored in an FVec.
    """

    def __init__(self, int uid, str name, list fnames=CDEC_DEFAULT):
        super(RuleTable, self).__init__(uid, name)
        self._fnames = tuple(fnames)

    def __repr__(self):
        return '{0}(uid={1}, name={2}, fnames={3})'.format(RuleTable.__name__,
                                                           repr(self.id),
                                                           repr(self.name),
                                                           repr(self._fnames))

    def __getstate__(self):
        return super(RuleTable,self).__getstate__(), {'fnames': self._fnames}

    def __setstate__(self, state):
        superstate, d = state
        self._fnames = tuple(d['fnames'])
        super(RuleTable,self).__setstate__(superstate)

    cpdef tuple fnames(self, wkeys):
        return self._fnames

    cpdef FRepr weights(self, dict wmap):  # using a dense representation
        cdef list wvec = []
        for f in self._fnames:
            try:
                wvec.append(wmap[f])
            except KeyError:
                raise KeyError('Missing RuleTable feature: %s' % f)
        return FVec(wvec)

    cpdef FRepr featurize(self, rule):
        """
        :param rule: an SCFGProduction
        :return:
        """
        return FVec([rule.fvalue(fname, 0) for fname in self._fnames])

    cpdef FRepr constant(self, weight_t value):
        return FVec([value] * len(self._fnames))