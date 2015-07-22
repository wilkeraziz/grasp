"""
:Authors: - Wilker Aziz
"""


class Edge(object):

    def __init__(self, uid, head, tail, weight, rule):
        self._uid = uid
        self._head = head
        self._tail = tuple(tail)
        self._weight = weight
        self._rule = rule

    def __eq__(self, other):
        return self._head == other._head and self._tail == other._tail and self._weight == other._weight

    def __hash__(self):
        return hash((self._head, self._tail, self._weight))

    @property
    def id(self):
        return self._uid

    @property
    def head(self):
        return self._head

    @property
    def tail(self):
        return self._tail

    @property
    def weight(self):
        return self._weight

    @property
    def rule(self):
        return self._rule

    def __str__(self):
        return '{0} ||| head={1} tail=({2}) weight={3}'.format(self._uid, self._head.id, ' '.join(str(u.id) for u in self._tail), self._weight)

