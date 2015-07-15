"""
:Authors: - Wilker Aziz
"""

import numpy as np
import itertools
from tabulate import tabulate
from .recipes import smart_wopen


class EmptyReport(object):

    def init(self, **kwargs):
        pass

    def report(self, i, **kwargs):
        pass

    def save(self):
        pass


class IterationReport(object):

    def __init__(self, prefix):
        self._prefix = prefix
        self._ids = []
        self._iterations = []
        self._init = {}

    def init(self, **kwargs):  # TODO: save it
        self._init = kwargs

    def report(self, i, **kwargs):
        self._ids.append(i)
        self._iterations.append(kwargs)

    def save(self):
        if not self._iterations:
            return

        header = sorted(self._iterations[0].keys())
        cid = {h: i for i, h in enumerate(header)}
        data = []
        for record in self._iterations:
            row = [0] * len(header)
            for k, v in sorted(record.items()):
                row[cid[k]] = v
            data.append(row)
        data = np.array(data)
        with smart_wopen('{0}.complete.gz'.format(self._prefix)) as fo:
            print(tabulate(np.column_stack([self._ids, data]), ['i'] + header), file=fo)

        with smart_wopen('{0}.summary.txt'.format(self._prefix)) as fo:
            table = [list(itertools.chain(['what'], header))]
            names = ['sum', 'mean', 'std', 'min', 'max']
            for n, d in zip(names, [data.sum(0), data.mean(0), data.std(0), data.min(0), data.max(0)]):
                table.append(list(itertools.chain([n], d)))
            print(tabulate(table, headers="firstrow"), file=fo)

