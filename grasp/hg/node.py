"""
:Authors: - Wilker Aziz
"""


from .carry import Carry


class Node(object):

    def __init__(self, uid, carry):
        self._uid = uid
        self._carry = carry

    @property
    def id(self):
        return self._uid

    @property
    def carry(self):
        return self._carry

    def __eq__(self, other):
        return self._carry == other._carry

    def __hash__(self):
        return hash(self._carry)

    def __str__(self):
        return '{0} ||| {1}'.format(self._uid, self._carry)