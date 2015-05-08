"""
@author wilkeraziz
"""

from collections import defaultdict
from itertools import chain, groupby


class CFG(object):

    def __init__(self, rules=[]):
        self._rules_by_lhs = defaultdict(set,
                {lhs: set(group) 
                    for lhs, group in groupby(sorted(rules, key=lambda r: r.lhs), key=lambda r: r.lhs)}
                )

    def add(self, rule):
        self._rules_by_lhs[rule.lhs].add(rule)

    def __getitem__(self, lhs):
        return self._rules_by_lhs.get(lhs, frozenset())

    def __iter__(self):
        return chain(*self._rules_by_lhs.itervalues())
    
    def iteritems(self):
        return self._rules_by_lhs.iteritems()
    
    def __str__(self):
        lines = []
        for lhs, rules in self.iteritems():
            for rule in rules:
                lines.append(str(rule))
        return '\n'.join(lines)


class FrozenCFG(object):

    def __init__(self, rules):
        self._rules_by_lhs = defaultdict(None,
                {lhs: frozenset(group) 
                    for lhs, group in groupby(sorted(rules, key=lambda r: r.lhs), key=lambda r: r.lhs)}
                )

    def __getitem__(self, lhs):
        return self._rules_by_lhs.get(lhs, frozenset())

    def __iter__(self):
        return chain(*self._rules_by_lhs.itervalues())

    def iteritems(self):
        return self._rules_by_lhs.iteritems()
    
    def __str__(self):
        lines = []
        for lhs, rules in self.iteritems():
            for rule in rules:
                lines.append(str(rule))
        return '\n'.join(lines)

