from grasp.scoring.frepr cimport FRepr, FVec, FCSR
from sklearn.feature_extraction import FeatureHasher
from grasp.recipes import re_key_value
from scipy.sparse import csr_matrix


cdef class RuleIndicator(Stateless):
    """
    This counts the number of rules of each arity.
    It returns an FMap if configured to deal with unbounded arity, otherwise, FVec.
    """

    def __init__(self, int uid, str name, int n_features=10000):
        super(RuleIndicator, self).__init__(uid, name)
        self._n_features = n_features
        # TODO: use non_negative=True ???
        self._hasher = FeatureHasher(n_features=n_features, input_type='string', non_negative=True)

    def __getstate__(self):
        return super(RuleIndicator,self).__getstate__(), {'n_features': self._n_features,
                                                          'hasher': self._hasher}

    def __setstate__(self, state):
        superstate, d = state
        self._n_features = d['n_features']
        self._hasher = d['hasher']
        super(RuleIndicator,self).__setstate__(superstate)

    def __repr__(self):
        return '{0}(uid={1}, name={2}, n_features={3}'.format(RuleIndicator.__name__,
                                                                           repr(self.id),
                                                                           repr(self.name),
                                                                           repr(self._n_features))

    cpdef str cfg(self):
        return '%s name=%s n-features=%d' % (RuleIndicator.__name__, self.name, self._n_features)

    cpdef tuple fnames(self, wkeys):
        return tuple(['{0}_{1}'.format(self.name, suffix) for suffix in range(self._n_features)])

    cpdef tuple features(self):
        return tuple(['{0}_{1}'.format(self.name, suffix) for suffix in range(self._n_features)])

    cpdef FRepr weights(self, dict wmap):
        cdef weight_t default = wmap.get(self.name, 0.0)
        return FCSR(csr_matrix([wmap.get('{0}_{1}'.format(self.name, i), default) for i in range(self._n_features)]))

    cpdef FRepr featurize(self, edge):
        """
        :param rule:
        :returns: edge's arity
        """
        if edge.fvalue('GoalRule') != 0:
            # GoalRules are added after each scoring pass,
            # as the number of scoring passes may vary, we better not score such "dummy" rules
            return FCSR.construct([], [], self._n_features)
        csr = self._hasher.transform([[edge.srule.tostr(fmap=False)]])
        return FCSR(csr)

    cpdef FRepr constant(self, weight_t value):
        return FCSR.construct([], [], self._n_features)

    @classmethod
    def construct(cls, int uid, str name, str cfgstr):
        cdef int n_features
        cfgstr, value = re_key_value('n-features', cfgstr, optional=False)
        n_features = int(value)
        return RuleIndicator(uid, name, n_features)

    @staticmethod
    def help():
        help_msg = ["# Indicator features based on rules.",
                    "# We make use of the hash-trick for efficiency, thus one must specify n-features."]
        return '\n'.join(help_msg)

    @staticmethod
    def example():
        return 'RuleIndicator n-features=10000'