"""
Table lookup extractors.

:Authors: - Wilker Aziz
"""
from grasp.scoring.frepr cimport FRepr, FVec
from grasp.ptypes cimport weight_t
from grasp.scoring.extractor cimport Extractor
from grasp.recipes import re_key_value
import numpy as np

CDEC_DEFAULT = 'Glue PassThrough IsSingletonF IsSingletonFE EgivenFCoherent SampleCountF CountEF MaxLexFgivenE MaxLexEgivenF'.split()


cdef class RuleTable(TableLookup):
    """
    A rule table much like those in cdec and Moses.
    Features are stored in an FVec.
    """

    def __init__(self, int uid, str name, list fnames=CDEC_DEFAULT, default=0.0):
        super(RuleTable, self).__init__(uid, name)
        self._fnames = tuple(fnames)
        self._default = default

    def __repr__(self):
        return '{0}(uid={1}, name={2}, fnames={3}, default={4})'.format(RuleTable.__name__,
                                                                        repr(self.id),
                                                                        repr(self.name),
                                                                        repr(self._fnames),
                                                                        repr(self._default))

    def __getstate__(self):
        return super(RuleTable,self).__getstate__(), {'fnames': self._fnames, 'default': self._default}

    def __setstate__(self, state):
        superstate, selfstate = state
        self._fnames = tuple(selfstate['fnames'])
        self._default = selfstate['default']
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
        return FVec([rule.fvalue(fname, self._default) for fname in self._fnames])

    cpdef FRepr constant(self, weight_t value):
        return FVec([value] * len(self._fnames))

    @classmethod
    def construct(cls, int uid, str name, str cfgstr):
        cdef list fnames = CDEC_DEFAULT
        cdef weight_t default = 0.0
        cfgstr, value = re_key_value('names', cfgstr, optional=True)
        if value:
            fnames = [fname for fname in value.split(',') if fname]
        cfgstr, value = re_key_value('default', cfgstr, optional=True)
        if value:
            default = float(value)
        return RuleTable(uid, name, fnames, default)

    @staticmethod
    def help():
        help_msg = ["# Read dense features from a grammar file.",
                    "# By default this extractor is instantiated as follows:",
                    "RuleTable fnames={0} default=0.0".format(','.join(CDEC_DEFAULT))]
        return '\n'.join(help_msg)

    @staticmethod
    def example():
        return 'RuleTable fnames=A,B,C,D default=0.0'

cdef class LogTransformedRuleTable(RuleTable):
    """
    A rule table much like those in cdec and Moses.
    Features are stored in an FVec.
    """

    def __init__(self, int uid, str name, list fnames, weight_t default=1.0):
        super(LogTransformedRuleTable, self).__init__(uid, name, fnames, default)

    def __repr__(self):
        return '{0}(uid={1}, name={2}, fnames={3}, default={4})'.format(LogTransformedRuleTable.__name__,
                                                                        repr(self.id),
                                                                        repr(self.name),
                                                                        repr(self._fnames),
                                                                        repr(self._default))

    cpdef FRepr featurize(self, rule):
        """
        :param rule: an SCFGProduction
        :return:
        """
        return FVec([np.log(rule.fvalue(fname, self._default)) for fname in self._fnames])

    @classmethod
    def construct(cls, int uid, str name, str cfgstr):
        cdef list fnames = []
        cdef weight_t default = 1.0
        cfgstr, value = re_key_value('names', cfgstr, optional=False)
        if value:
            fnames = [fname for fname in value.split(',') if fname]
        cfgstr, value = re_key_value('default', cfgstr, optional=True)
        if value:
            default = float(value)
        return LogTransformedRuleTable(uid, name, fnames, default)

    @staticmethod
    def help():
        return "# Read dense features from a grammar file and apply a log transform."

    @staticmethod
    def example():
        return 'LogTransformedRuleTable fnames=A,B,C,D default=1.0'