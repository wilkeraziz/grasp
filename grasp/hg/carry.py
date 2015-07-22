"""
:Authors: - Wilker Aziz
"""

from collections import defaultdict
from weakref import WeakValueDictionary


class Carry(object):
    """
    An immutable carry.
    """

    _inventory = WeakValueDictionary()  # TODO: make it threadlocal

    def __new__(cls, **kwargs):
        """The key-value pairs must be hashable."""
        state = defaultdict(None, kwargs)
        key = frozenset(state.items())
        obj = Carry._inventory.get(key, None)
        if not obj:
            obj = object.__new__(cls)
            Carry._inventory[key] = obj
            obj._state = state
        return obj

    def get(self, key):
        return self._state(key, None)

    def __str__(self):
        return ' '.join('{0}={1}'.format(k, v) for k, v in self._state.items())