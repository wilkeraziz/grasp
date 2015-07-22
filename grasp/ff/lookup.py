"""
:Authors: - Wilker Aziz
"""

import numpy as np
from .extractor import TableLookup


class RuleTable(TableLookup):

    CDEC_DEFAULT = 'Glue PassThrough EgivenFCoherent SampleCountF CountEF MaxLexFgivenE MaxLexEgivenF IsSingletonF IsSingletonFE'.split()

    def __init__(self, uid, name, fnames=CDEC_DEFAULT):
        super(RuleTable, self).__init__(uid, name)
        self._fnames = tuple(fnames)

    def __repr__(self):
        return '{0}(uid={1}, name={2}, fnames={3})'.format(RuleTable.__name__,
                                                           repr(self.id),
                                                           repr(self.name),
                                                           repr(self._fnames))

    def weights(self, wmap):  # using a dense representation
        wvec = []
        for f in self._fnames:
            try:
                wvec.append(wmap[f])
            except KeyError:
                raise KeyError('Missing RuleTable feature: %s' % f)
        return np.array(wvec, float)

    def featurize(self, rule):  # using a dense representation
        """
        :param rule: an SCFGProduction
        :return:
        """
        fmap = rule.fmap
        return np.array([fmap.get(fname, 0) for fname in self._fnames], float)

    def dot(self, fs, ws):  # dense dot
        return fs.dot(ws)



