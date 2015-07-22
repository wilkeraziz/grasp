"""
Any hashable object is a potential *state*, thus we do not have a State class.
By this definition, a tuple of states is itself a state (because tuples are also hashable objects).

A StateMapper maps states (possibly tuples of states) to integers.

:Authors: - Wilker Aziz
"""
from collections import defaultdict


class StateMapper(object):
    """
    Map states (hashable objects) to integers.
    This allows one to use arbitrarily complex state objects in programs such as Earley and Nederhof.

    Note:
        - 0 is reserved to be a final state

    """

    def __init__(self):
        self._state2int = defaultdict(None, {None: 0})
        self._int2state = [None]

    @property
    def final(self):
        """State reserved as final."""
        return 0

    def __len__(self):
        """number of states mapped"""
        return len(self._int2state)

    def to_int(self, state):
        """Return the integer associated with the given state (any hashable object)."""
        uid = self._state2int.get(state, None)
        if uid is None:
            uid = len(self._int2state)
            self._int2state.append(state)
            self._state2int[state] = uid
        return uid

    def __getitem__(self, item):
        """Return the state (a hashable object) associated with a given integer."""
        return self._int2state[item]
