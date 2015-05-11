"""
@author wilkeraziz
"""

from weakref import WeakValueDictionary

class Item(object):

    _items = WeakValueDictionary()

    def __new__(cls, rule, dot, inner=[]):
        """The symbols in lhs and rhs must be hashable"""
        skeleton = (rule, dot, tuple(inner))
        obj = Item2._items.get(skeleton, None)
        if not obj:
            obj = object.__new__(cls)
            Item2._items[skeleton] = obj
            obj._skeleton = skeleton
            obj._start = inner[0] if inner else dot
            obj._next = rule.rhs[len(inner)] if len(inner) < len(rule.rhs) else None
            obj._status = 0  # 0) created, 1) active, 2) passive, other) discarded
        return obj
    
    @property
    def rule(self):
        return self._skeleton[0]

    @property
    def dot(self):
        return self._skeleton[1]

    @property
    def inner(self):
        return self._skeleton[2]

    @property
    def start(self):
        return self._start

    @property
    def next(self):
        return self._next

    def is_active(self):
        return self._active

    def make_passive(self):
        return self._active = False

    def nextsymbols(self):
        return tuple(self.rule.rhs[len(self.inner):])

    def is_complete(self):
        return len(self.inner) == len(self.rule.rhs)

    def __str__(self):
        return '%s ||| %s ||| %d' % (str(self.rule), self.inner, self.dot)

