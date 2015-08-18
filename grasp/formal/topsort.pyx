"""

TODO: a common interface between AcyclicTopSortTable and RobustTopSortTable.

:Authors: - Wilker Aziz
"""


import itertools
import numpy as np
from libcpp.queue cimport queue
from libcpp.stack cimport stack
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from cython.operator cimport dereference as deref, preincrement as inc
from grasp.ptypes cimport id_t
from grasp.formal.hg cimport Hypergraph


cdef np.int_t[::1] acyclic_topsort(Hypergraph hg):

    cdef:
        vector[bint] booked
        vector[unordered_set[id_t]] deps
        vector[unordered_set[id_t]] fs
        queue[id_t] agenda
        np.int_t[::1] levels = np.zeros(hg.n_nodes(), dtype=np.int)
        id_t n, e, p, q

    # pre-allocate stuff
    booked.resize(hg.n_nodes())
    deps.resize(hg.n_nodes())
    fs.resize(hg.n_nodes())

    # dependencies and forward star
    for p in range(hg.n_nodes()):
        deps[p] = hg.iterdeps(p)
        fs[p] = hg.iterfs(p)
        if deps[p].empty():
            agenda.push(p)
            booked[p] = True
        if hg.is_terminal(p):  # terminals get level 1
            levels[p] = 1

    cdef unordered_set[id_t].iterator it
    # exhausts the agenda
    while not agenda.empty():
        q = agenda.front()
        agenda.pop()
        it = fs[q].begin()
        while it != fs[q].end():
            e = deref(it)
            p = hg.head(e)
            deps[p].erase(q)
            if levels[p] < levels[q] + 1:
                levels[p] = levels[q] + 1
            if deps[p].empty() and not booked[p]:
                agenda.push(p)
                booked[p] = True
            inc(it)

    return levels


cdef strong_connect(id_t v,
                      vector[tarjan_node_t]& nodes,
                      id_t* index,
                      stack[id_t]& agenda,
                      vector[unordered_set[id_t]]& deps,
                      vector[vector[id_t]]& order):
    nodes[v].index = index[0]
    nodes[v].low = index[0]
    index[0] += 1
    agenda.push(v)
    nodes[v].stacked = True

    cdef unordered_set[id_t].iterator it = deps[v].begin()
    cdef id_t w
    while it != deps[v].end():
        w = deref(it)
        if nodes[w].index < 0:  # undefined
            strong_connect(w, nodes, index, agenda, deps, order)
            nodes[v].low = np.min([nodes[v].low, nodes[w].low])
        elif nodes[w].stacked:
            nodes[v].low = np.min([nodes[v].low, nodes[w].index])
        inc(it)

    if nodes[v].low == nodes[v].index:
        order.push_back(vector[id_t]())
        while not agenda.empty():
            w = agenda.top()
            agenda.pop()
            nodes[w].stacked = False
            order.back().push_back(w)
            if w == v:
                break


cdef void c_tarjan(Hypergraph hg, vector[vector[id_t]]& order):

    cdef:
        vector[tarjan_node_t] nodes
        id_t index = 0
        id_t n
        stack[id_t] agenda
        vector[unordered_set[id_t]] deps
        #vector[vector[id_t]] order

    deps.resize(hg.n_nodes())
    nodes.resize(hg.n_nodes())

    for n in range(hg.n_nodes()):
        deps[n] = hg.iterdeps(n)
        nodes[n].index = -1
        nodes[n].low = -1
        nodes[n].stacked = False

    for n in range(hg.n_nodes()):
        if nodes[n].index < 0:  # undefined
            strong_connect(n, nodes, &index, agenda, deps, order)


cpdef list tarjan(Hypergraph hg):
    cdef vector[vector[id_t]] order
    c_tarjan(hg, order)
    return order


cdef np.int_t[::1] robust_topsort(Hypergraph hg, vector[vector[id_t]]& components):

    # map nodes to components
    cdef:
        size_t c
        vector[size_t] node_to_component
        vector[id_t].iterator it
    node_to_component.resize(hg.n_nodes())
    for c in range(components.size()):
        it = components[c].begin()
        while it != components[c].end():
            node_to_component[deref(it)] = c
            inc(it)

    # topsort components
    cdef:
        vector[bint] booked
        vector[unordered_set[id_t]] comp_indeps  # nodes in incoming edges
        vector[unordered_set[id_t]] comp_outdeps  # nodes in outgoing edges
        queue[id_t] agenda
        np.int_t[::1] levels = np.zeros(components.size(), dtype=np.int)
        id_t p_c, q_c

    booked.resize(components.size())
    comp_indeps.resize(components.size())
    comp_outdeps.resize(components.size())

    # find the dependencies between strongly connected components
    cdef id_t p, q
    for p in range(hg.n_nodes()):
        p_c = node_to_component[p]
        for q in hg.iterdeps(p):
            q_c = node_to_component[q]
            if p_c != q_c:  # in finding levels we do not add self dependencies
                comp_indeps[p_c].insert(q_c)
                comp_outdeps[q_c].insert(p_c)

    # find the source and terminal components
    for c in range(components.size()):
        if comp_indeps[c].empty():
            agenda.push(c)
            booked[c] = True
        if components[c].size() == 1 and hg.is_terminal(components[c][0]):  # Terminal components get level 1
            levels[c] = 1

    # exhausts the agenda finding the level of each component
    cdef unordered_set[id_t].iterator dit
    while not agenda.empty():
        q = agenda.front()
        agenda.pop()
        dit = comp_outdeps[q].begin()
        while dit != comp_outdeps[q].end():
            p = deref(dit)
            comp_indeps[p].erase(q)
            if levels[p] < levels[q] + 1:
                levels[p] = levels[q] + 1
            if comp_indeps[p].empty() and not booked[p]:
                agenda.push(p)
                booked[p] = True
            inc(dit)

    return levels


cdef class TopSortTable: pass


cdef class AcyclicTopSortTable(TopSortTable):

    def __init__(self, Hypergraph hg):
        self._hg = hg
        cdef size_t c, i
        cdef id_t n
        cdef np.int_t[::1] tmp_levels
        self._levels = acyclic_topsort(hg)
        cdef int max_level = np.max(self._levels)
        self._tsort = [[] for _ in range(max_level + 1)]
        cdef int l
        for n in range(hg.n_nodes()):
            l = self._levels[n]
            (<list>self._tsort[l]).append(n)

    cpdef size_t n_levels(self):
        return len(self._tsort)

    cpdef size_t n_top(self):
        return len(self._tsort[-1])

    cpdef size_t level(self, id_t node):
        return self._levels[node]

    cpdef id_t root(self) except -1:
        """
        Return the start/root/goal symbol/node of the grammar/forest/hypergraph
        :return: node
        """
        if self.n_top() > 1:  # more than one root
            raise ValueError('I expected a single root instead of %d' % self.n_top())
        return self._tsort[-1][0]

    cpdef itertop(self):
        """Iterate over the top buckets of the grammar/forest/hypergraph"""
        return iter(self._tsort[-1])

    cpdef iterlevels(self, bint reverse=False, size_t skip=0):
        """
        Iterate level by level (a level is a set of buckets sharing a ranking).
        In Goodman (1999), a bucket which is not a singleton is called a "loopy bucket".

        :param reverse: bottom-up if False, top-down if True
        :param skip: skip a number of levels
        :return: iterator over levels
        """
        # bottom-up vs top-down
        if not reverse:
            iterator = iter(self._tsort)
        else:
            iterator = reversed(self._tsort)
        # skipping n levels
        for n in range(skip):
            next(iterator)
        return iterator

    cpdef iternodes(self, bint reverse=False, size_t skip=0):
        """
        Iterate bucket by bucket (a bucket is a set of strongly connected nodes).
        In Goodman (1999), a bucket which is not a singleton is called a "loopy bucket".

        :param reverse: bottom-up if False, top-down if True
        :param skip: skip a number of levels
        :return: iterator over buckets
        """
        iterator = self.iterlevels(reverse, skip)
        return itertools.chain(*iterator)

    def __iter__(self):
        return self.iternodes(reverse=False, skip=0)

    def __str__(self):
        cdef id_t i, n
        cdef list level, lines = []
        for i, level in enumerate(self.iterlevels(skip=0)):
            lines.append('level=%d' % i)
            for n in level:
                lines.append('  %d %r' % (n, self._hg.label(n)))
        return '\n'.join(lines)

    def pp(self):
        cdef id_t i, n
        cdef list level, lines = []
        for i, level in enumerate(self.iterlevels(skip=0)):
            lines.append('level=%d' % i)
            for n in level:
                lines.append('  %r' % (self._hg.label(n)))
        return '\n'.join(lines)


cdef class RobustTopSortTable(TopSortTable):

    def __init__(self, Hypergraph hg):
        self._hg = hg

        c_tarjan(hg, self._components)
        cdef:
            np.int_t[::1] comp_levels = robust_topsort(hg, self._components)
            int max_level = np.max(comp_levels)
            size_t c, l
        self._tsort = [[] for _ in range(max_level + 1)]
        for c in range(self._components.size()):
            l = comp_levels[c]
            (<list>self._tsort[l]).append(self._components[c])

        # map node to levels
        self._levels = np.zeros(hg.n_nodes(), dtype=np.int)
        cdef vector[id_t].iterator it
        for c in range(self._components.size()):
            it = self._components[c].begin()
            while it != self._components[c].end():
                n = deref(it)
                self._levels[n] = comp_levels[c]
                inc(it)

    cpdef size_t n_levels(self):
        return len(self._tsort)

    cpdef size_t n_top(self):
        return np.sum([len(b) for b in self._tsort[-1]])

    cpdef size_t level(self, id_t node):
        return self._levels[node]

    cpdef id_t root(self) except -1:
        """
        Return the start/root/goal symbol/node of the grammar/forest/hypergraph
        :return: node
        """

        toplevel = self._tsort[-1]  # top-most set of buckets
        if len(toplevel) > 1:  # more than one bucket
            raise ValueError('I expected a single bucket instead of %d\n%s')
        topbucket = toplevel[0]
        if len(topbucket) > 1:  # sometimes this is a loopy bucket (more than one node)
            raise ValueError('I expected a single start symbol instead of %d' % len(topbucket))
        return topbucket[0]

    cpdef itertopbuckets(self):
        """Iterate over the top buckets of the grammar/forest/hypergraph"""
        return iter(self._tsort[-1])

    cdef itertopnodes(self):
        return itertools.chain(*self.itertopbuckets())

    cpdef iterlevels(self, bint reverse=False, size_t skip=0):
        """
        Iterate level by level (a level is a set of buckets sharing a ranking).
        In Goodman (1999), a bucket which is not a singleton is called a "loopy bucket".

        :param reverse: bottom-up if False, top-down if True
        :param skip: skip a number of levels
        :return: iterator over levels
        """
        # bottom-up vs top-down
        if not reverse:
            iterator = iter(self._tsort)
        else:
            iterator = reversed(self._tsort)
        # skipping n levels
        for n in range(skip):
            next(iterator)
        return iterator

    cpdef iterbuckets(self, bint reverse=False, size_t skip=0):
        """
        Iterate bucket by bucket (a bucket is a set of strongly connected nodes).
        In Goodman (1999), a bucket which is not a singleton is called a "loopy bucket".

        :param reverse: bottom-up if False, top-down if True
        :param skip: skip a number of levels
        :return: iterator over buckets
        """
        iterator = self.iterlevels(reverse, skip)
        return itertools.chain(*iterator)

    cpdef iternodes(self, bint reverse=False, size_t skip=0):
        iterator = self.iterlevels(reverse, skip)
        return itertools.chain(*itertools.chain(*iterator))

    def __iter__(self):
        return self.iternodes(reverse=False, skip=0)

    cpdef bint is_loopy(self, list bucket):
        return (len(bucket) > 1) or (len(bucket) == 1 and self._hg.self_depends(bucket[0]))

    def __str__(self):
        cdef list level, bucket, lines = []
        cdef id_t i, j, n
        for i, level in enumerate(self.iterlevels(skip=0)):
            lines.append('level=%d' % i)
            for j, bucket in enumerate(level):
                lines.append(' bucket=%d loopy=%s' % (j, self.is_loopy(bucket)))
                for n in bucket:
                    lines.append('  %d %r' % (n, self._hg.label(n)))
        return '\n'.join(lines)

    def pp(self):
        cdef list level, bucket, lines = []
        cdef id_t i, j, n
        for i, level in enumerate(self.iterlevels(skip=0)):
            lines.append('level=%d' % i)
            for j, bucket in enumerate(level):
                if not self.is_loopy(bucket):
                    lines.append(' acyclic: %r' % self._hg.label(bucket[0]))
                else:
                    lines.append(' loopy:')
                    for n in bucket:
                        lines.append('  %r' % (self._hg.label(n)))
        return '\n'.join(lines)


class LazyTopSortTable:
    """
    A simple container for a TopSortTable object which gets lazily computed.
    """

    def __init__(self, Hypergraph forest, bint acyclic=False):
        """
        :param forest: the forest whose nodes will be sorted
        :param acyclic: whether the forest contains cycles
        """
        self._forest = forest
        self._acyclic = acyclic
        self._tsort = None

    def __call__(self):
        return self.do()

    def do(self):
        """Return a TopSortTable (acyclic or robust depending on the case) building it if necessary."""
        if self._tsort is None:
            if self._acyclic:
                self._tsort = AcyclicTopSortTable(self._forest)
            else:
                self._tsort = RobustTopSortTable(self._forest)
        return self._tsort