"""
Table lookup extractors.

:Authors: - Wilker Aziz
"""
from grasp.scoring.frepr cimport FRepr, FVec, FValue
from grasp.ptypes cimport weight_t
from grasp.scoring.extractor cimport Extractor
from grasp.recipes import re_key_value
import numpy as np

CDEC_DEFAULT = 'Glue PassThrough IsSingletonF IsSingletonFE EgivenFCoherent SampleCountF CountEF MaxLexFgivenE MaxLexEgivenF'.split()


cdef class NamedFeature(TableLookup):

    def __init__(self, int uid, str name, str fkey='', default=0.0):
        super(NamedFeature, self).__init__(uid, name)
        self._default = default
        if not fkey:
            self._fkey = name
        else:
            self._fkey = fkey

    def __repr__(self):
        return '{0}(uid={1}, name={2}, fkey={3}, default={4})'.format(NamedFeature.__name__,
                                                                       repr(self.id),
                                                                       repr(self.name),
                                                                       repr(self._fkey),
                                                                       repr(self._default))

    cpdef str cfg(self):
        return '%s name=%s key=%s default=%f' % (NamedFeature.__name__, self.name, self._fkey, self._default)

    def __getstate__(self):
        return super(NamedFeature,self).__getstate__(), {'default': self._default, 'fkey': self._fkey}

    def __setstate__(self, state):
        superstate, selfstate = state
        self._default = selfstate['default']
        self._fkey = selfstate['fkey']
        super(NamedFeature,self).__setstate__(superstate)

    cpdef tuple fnames(self, wkeys):
        return tuple([self.name])

    cpdef tuple features(self):
        return tuple([self.name])

    cpdef FRepr weights(self, dict wmap):
        try:
            return FValue(wmap[self.name])
        except KeyError:
            raise KeyError('Missing weight for %r' % self)

    cpdef FRepr featurize(self, rule):
        """
        :param word: a Terminal
        :param context: a state
        :returns: weight
        """
        return FValue(rule.fvalue(self._fkey, self._default))

    cpdef FRepr constant(self, weight_t value):
        return FValue(value)

    @classmethod
    def construct(cls, int uid, str name, str cfgstr):
        cdef weight_t default = 0.0
        cdef str fkey = name
        cfgstr, value = re_key_value('key', cfgstr, optional=True)
        if value:
            fkey = value
        cfgstr, value = re_key_value('default', cfgstr, optional=True)
        if value:
            default = float(value)
        return NamedFeature(uid, name, fkey, default)

    @staticmethod
    def help():
        return "# A named feature from the rule's feature map. Use default=float to set a default value"

    @staticmethod
    def example():
        return 'NamedFeature name=GlueFeature key=Glue default=0.0'


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

    cpdef str cfg(self):
        return '%s name=%s default=%f names=%s' % (RuleTable.__name__, self.name, self._default, ','.join(self._fnames))

    def __getstate__(self):
        return super(RuleTable,self).__getstate__(), {'fnames': self._fnames, 'default': self._default}

    def __setstate__(self, state):
        superstate, selfstate = state
        self._fnames = tuple(selfstate['fnames'])
        self._default = selfstate['default']
        super(RuleTable,self).__setstate__(superstate)

    cpdef tuple fnames(self, wkeys):
        return self._fnames

    cpdef tuple features(self):
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

    cpdef str cfg(self):
        return '%s name=%s default=%f names=%s' % (LogTransformedRuleTable.__name__, self.name, self._default,
                                                   ','.join(self._fnames))

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