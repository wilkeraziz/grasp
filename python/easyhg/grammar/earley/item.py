"""
@author wilkeraziz
"""

from weakref import WeakValueDictionary
from collections import defaultdict, deque

class Item(object):
    
    def __init__(self, sid, rule, dot, inner):
        """
        A state in the intersection procedure.
        @param sid: item's id
        @param rule: a cfg rule
        @param dot: an fsa state
        @param inner: fsa states intersected thus far
        """
        self.uid_ = sid
        self.rule_ = rule
        self.dot_ = dot
        self.inner_ = tuple(inner)
        self.start_ = inner[0] if inner else dot
        n = len(inner)
        self.next_ = rule.rhs_[n] if n < len(rule.rhs_) else None

    def __eq__(self, other):
        return self.uid_ == other.uid_

    def __ne__(self, other):
        return self.uid_ != other.uid_

    def __hash__(self):
        return hash(self.uid_)

    def __str__(self):
        return '%d) %s %s %d' % (self.uid_, str(self.rule_), self.inner_, self.dot_)

    @property
    def uid(self):
        return self.uid_

    @property
    def rule(self):
        return self.rule_

    @property
    def start(self):
        return self.start_

    @property
    def dot(self):
        return self.dot_

    @property
    def inner(self):
        return self.inner_

    @property
    def next(self):
        return self.next_

    def nextsymbols(self):
        return tuple(self.rule_.rhs[len(self.inner_):])

    def is_complete(self):
        return len(self.inner_) == len(self.rule_.rhs)


class ItemFactory(object):

    def __init__(self):
        self._uid_by_key = defaultdict(None)
        self._items = deque()

    def get_item(self, rule, dot, inner=[]):
        key = (rule, dot, tuple(inner))
        uid = self._uid_by_key.get(key, None)
        if uid is None:
            uid = len(self._items)
            self._items.append(Item(uid, rule, dot, inner))
            self._uid_by_key[key] = uid
        return self._items[uid]

    def __getitem__(self, uid):
        return self._items[uid]

class Item2(object):

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
            obj._active = True
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

