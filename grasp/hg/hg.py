"""
:Authors: - Wilker Aziz
"""

from collections import deque, defaultdict
from .carry import Carry
from .node import Node
from .edge import Edge


class Hypergraph(object):

    def __init__(self):
        self._nodes = deque()
        self._edges = deque()

    def add_node(self, carry):
        """
        Add a node based on a certain carry (does not check for duplicates).
        :param carry:
        :return: node id
        """
        nid = len(self._nodes)
        self._nodes.append(Node(nid, carry))
        return nid

    def add_edge(self, head, tail, weight, rule):
        """
        Add an edge (does not check node ids).
        :param head: id of the head node
        :param tail:  sequence of ids of tail nodes
        :param weight: edge weight
        :param rule: a rule application
        :return: edge id
        """
        eid = len(self._edges)
        self._edges.append(Edge(eid, self._nodes[head], tuple(self._nodes[n] for n in tail), weight, rule))
        return eid

    def write(self, ostream):
        print('# nodes={0}'.format(len(self._nodes), file=ostream))
        for node in self._nodes:
            print(node, file=ostream)
        print('# edges={0}'.format(len(self._edges), file=ostream))
        for edge in self._edges:
            print(edge, file=ostream)