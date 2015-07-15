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

    def add_iteration(self, **kwargs):
        pass

    def save(self):
        pass


class DefaultReport(object):

    def __init__(self, prefix):
        self._prefix = prefix
        self._iterations = []
        self._init = {}

    def init(self, **kwargs):
        self._init = kwargs

    def add_iteration(self, **kwargs):
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
            print(tabulate(data, header), file=fo)

        with smart_wopen('{0}.summary.txt'.format(self._prefix)) as fo:

            n = len(self._iterations)

            total = data.sum(0)
            mean = data.mean(0)
            std = data.mean(0)
            norm = total / n

            table = [list(itertools.chain(['what'], header))]
            names = ['sum', 'mean', 'std', 'norm']
            for n, d in zip(names, [total, mean, std, norm]):
                table.append(list(itertools.chain([n], d)))
            print(tabulate(table, headers="firstrow"), file=fo)

