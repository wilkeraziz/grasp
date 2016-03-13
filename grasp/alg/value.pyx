"""
This implements the value recursion for numerical semirings.

    V(v) = \bigoplus_{e \in BS(v)} \omega(e) \bigotimes_{u \in tail(e)} V(u)

We also have an implementation which is robust to the presence of cycles.

:Authors: - Wilker Aziz
"""

from grasp.ptypes cimport weight_t, id_t
from grasp.formal.hg cimport Hypergraph
from grasp.formal.topsort cimport AcyclicTopSortTable, RobustTopSortTable

import numpy as np
cimport numpy as np


cdef class ValueFunction:
    """
    A value function assigns a value to an edge.
    """

    def __init__(self):
        pass

    def __call__(self, id_t e):
        return self.value(e)

    cpdef weight_t value(self, id_t e): pass

    cpdef weight_t reduce(self, Operator op, iterable):
        cdef id_t e
        return op.reduce([self.value(e) for e in iterable])


cdef class ConstantFunction(ValueFunction):

    def __init__(self, weight_t constant):
        self.constant = constant

    cpdef weight_t value(self, id_t e):
        return self.constant

    def __repr__(self):
        return '%s(%r)' % (ConstantFunction.__name__, self.constant)


cdef class CascadeValueFunction(ValueFunction):

    def __init__(self, Operator op, functions):
        self.functions = tuple(functions)
        self.op = op

    cpdef weight_t value(self, id_t e):
        cdef ValueFunction func
        return self.op.reduce([func.value(e) for func in self.functions])

    def __repr__(self):
        return '%s(%r, %r)' % (CascadeValueFunction.__name__, self.op, self.functions)


cdef class LookupFunction(ValueFunction):
    """
    A value function that consists in plain simple table lookup.
    """

    def __init__(self, weight_t[::1] table):
        self.table = table

    cpdef weight_t value(self, id_t e):
        return self.table[e]


cdef class EdgeWeight(ValueFunction):
    """
    A value function which reproduces the edge's weight in a hypergraph.
    """

    def __init__(self, Hypergraph hg):
        self.hg = hg

    cpdef weight_t value(self, id_t e):
        return self.hg.weight(e)


cdef class ScaledEdgeWeight(ValueFunction):
    """
    A value function which reproduces the edge's weight in a hypergraph.
    """

    def __init__(self, Hypergraph hg, weight_t scalar):
        self.hg = hg
        self.scalar = scalar

    cpdef weight_t value(self, id_t e):
        return self.hg.weight(e) * self.scalar


cdef class ThresholdValueFunction(ValueFunction):
    """
    Applies a threshold to an edge's value.
    """

    def __init__(self, ValueFunction f, Semiring input_semiring, weight_t threshold, Semiring output_semiring):
        self.f = f
        self.input_semiring = input_semiring
        self.threshold = threshold
        self.output_semiring = output_semiring

    cpdef weight_t value(self, id_t e):
        """
        Returns 1 (in the output semiring) if the edge's value is greater than the threshold (both in the input semiring),
        otherwise it returns 0 (in the output semiring).
        """
        return self.output_semiring.one if self.input_semiring.gt(self.f.value(e), self.threshold) else self.output_semiring.zero


cdef class BinaryEdgeWeight(ValueFunction):

    def __init__(self, Hypergraph hg, Semiring input_semiring, Semiring output_semiring):
        self.hg = hg
        self.input_semiring = input_semiring
        self.output_semiring = output_semiring

    cpdef weight_t value(self, id_t e):
        if self.input_semiring.gt(self.hg.weight(e), self.input_semiring.zero):
            return self.output_semiring.one
        else:
            return self.output_semiring.zero


cpdef weight_t derivation_value(Hypergraph forest, tuple edges, Semiring semiring, ValueFunction omega=None):
    cdef id_t e
    if omega is None:
        return semiring.times.reduce([forest.weight(e) for e in edges])
    else:
        return semiring.times.reduce([omega.value(e) for e in edges])


cdef weight_t node_value(Hypergraph forest,
                            ValueFunction omega,
                            Semiring semiring,
                            weight_t[::1] values,
                            id_t parent):

    if forest.is_source(parent):
        return semiring.one if forest.is_terminal(parent) else semiring.zero

    cdef:
        weight_t[::1] partials = semiring.zeros(forest.n_incoming(parent))
        size_t i = 0
        id_t e, child
    for e in forest.iterbs(parent):
        partials[i] = semiring.times(omega.value(e),
                                     semiring.times.reduce([values[child] for child in forest.tail(e)]))
        i += 1
    return semiring.plus.reduce(partials)


cpdef weight_t[::1] acyclic_value_recursion(Hypergraph forest,
                                            AcyclicTopSortTable tsort,
                                            Semiring semiring,
                                            ValueFunction omega=None):
    """
    Returns items' values in a given semiring.
    This is a bottom-up pass through the forest which runs in O(|forest|).
    :param forest: an acyclic hypergraph-like object.
    :param tsort: a TopSortTable object.
    :param semiring: must define zero, one, sum and times.
    :param omega: a function that weighs edges/rules (defaults to the edge's weight).
        You might want to use this to, for instance, convert between semirings.
    :return:
    """

    if omega is None:
        omega = EdgeWeight(forest)

    cdef weight_t[::1] values = semiring.zeros(forest.n_nodes())

    cdef:
        size_t i
        id_t parent, child, e
        weight_t[::1] partials

    # we go bottom-up
    for parent in tsort.iternodes(reverse=False, skip=1):  # skipping useless nodes in level 0
        values[parent] = node_value(forest, omega, semiring, values, parent)

    return values


cpdef weight_t[::1] acyclic_reversed_value_recursion(Hypergraph forest,
                                            AcyclicTopSortTable tsort,
                                            Semiring semiring,
                                            weight_t[::1] values,
                                            ValueFunction omega=None):
    """
    Returns items' reversed values in a given semiring.
    This is a top-down pass through the forest which runs in O(|forest|).
    :param forest: an acyclic hypergraph-like object.
    :param tsort: a TopSortTable object.
    :param semiring: must define zero, one, sum and times.
    :param omega: a function that weighs edges/rules (defaults to the edge's weight).
        You might want to use this to, for instance, convert between semirings.
    :return: reversed values
    """

    if omega is None:
        omega = EdgeWeight(forest)

    cdef:
        size_t i
        id_t parent, child, sister, e
        weight_t partial
        weight_t[::1] reversed_values = semiring.zeros(forest.n_nodes())

    reversed_values[tsort.root()] = semiring.one

    # we go top-down
    for parent in tsort.iternodes(reverse=True):
        for e in forest.iterbs(parent):
            for child in forest.tail(e):
                # start from parent's outside and the edge's weight
                partial = semiring.times(omega.value(e), reversed_values[parent])
                for sister in forest.tail(e):
                    if child != sister:
                        # incorporate inside of sister: note that every sister participates in the tail, thus semiring.times
                        partial = semiring.times(partial, values[sister])
                # update child's outside: note that edges contribute with alternative paths, thus we use semiring.plus
                reversed_values[child] = semiring.plus(reversed_values[child], partial)

    return reversed_values


cpdef weight_t[::1] robust_value_recursion(Hypergraph forest,
                                           RobustTopSortTable tsort,
                                           Semiring semiring,
                                           ValueFunction omega=None):
    """
    Returns items' values in a given semiring.
    This is a bottom-up pass through the forest which runs in O(|forest|).
    :param forest: a hypergraph-like object.
    :param tsort: a TopSortTable object.
    :param semiring: must define zero, one, sum and times.
    :param omega: a function that weighs edges/rules (defaults to the edge's weight).
        You might want to use this to, for instance, convert between semirings.
    :return:
    """

    if omega is None:
        omega = EdgeWeight(forest)

    cdef weight_t[::1] values = semiring.zeros(forest.n_nodes())

    cdef:
        size_t i
        id_t parent
        list bucket
        weight_t[::1] supremum

    # we go bottom-up
    for bucket in tsort.iterbuckets(reverse=False, skip=1):  # we skip useless nonterminals at level 0
        if not tsort.is_loopy(bucket):  # for sure a singleton
            parent = bucket[0]
            values[parent] = node_value(forest, omega, semiring, values, parent)
        else:
            supremum = approximate_supremum(forest, omega, semiring, values, bucket)
            for i in range(len(bucket)):
                parent = bucket[i]
                values[parent] = supremum[i]
    return values


cdef weight_t[::1] approximate_supremum(Hypergraph forest,
                                        ValueFunction omega,
                                        Semiring semiring,
                                        weight_t[::1] values,
                                        list bucket):
    # TODO: solve supremum using linear algebra

    cdef:
        weight_t[::1] supremum = semiring.zeros(len(bucket))
        weight_t[::1] supremum_g = semiring.zeros(len(bucket))
        weight_t[::1] partials
        set bucket_set = set(bucket)

    cdef:
        id_t parent, e, child
        size_t i, j
        weight_t w

    while True:
        supremum_g[...] = semiring.zero
        for i in range(len(bucket)):
            parent = bucket[i]

            if forest.is_source(parent):
                supremum_g[i] = semiring.one if forest.is_terminal(parent) else semiring.zero
                continue

            partials = semiring.zeros(forest.n_incoming(parent))
            j = 0
            for e in forest.iterbs(parent):
                w = semiring.times.reduce([supremum[child] if child in bucket_set else values[child]
                                           for child in forest.tail(e)])
                partials[j] = semiring.times(omega.value(e), w)
                j += 1
            supremum_g[i] = semiring.plus.reduce(partials)

        if np.all(np.equal(supremum, supremum_g)):
            # if the values of all items have remained unchanged,
            # they have all converged to their supremum values
            break
        else:
            supremum[...] = supremum_g[...]

    return supremum


cpdef weight_t[::1] compute_edge_values(Hypergraph forest,
                                        Semiring semiring,
                                        weight_t[::1] node_values,
                                        ValueFunction omega=None,
                                        bint normalise=False):
    cdef:
        weight_t[::1] edge_values = semiring.zeros(forest.n_edges())
        id_t e, child
        weight_t tail_value, w

    if omega is None:
        omega = EdgeWeight(forest)

    if normalise:
        for e in range(forest.n_edges()):
            tail_value = semiring.times.reduce([node_values[child] for child in forest.tail(e)])
            w = semiring.times(omega.value(e), tail_value)
            edge_values[e] = semiring.divide(w, node_values[forest.head(e)])
    else:
        for e in range(forest.n_edges()):
            tail_value = semiring.times.reduce([node_values[child] for child in forest.tail(e)])
            w = semiring.times(omega.value(e), tail_value)
            edge_values[e] = w

    return edge_values


cpdef weight_t[::1] compute_edge_expectation(Hypergraph forest,
                                        Semiring semiring,
                                        weight_t[::1] node_values,
                                        weight_t[::1] node_reversed_values,
                                        ValueFunction omega=None,
                                        bint normalise=False):
    cdef:
        weight_t[::1] edge_expec = semiring.zeros(forest.n_edges())
        id_t e, child
        weight_t w

    if omega is None:
        omega = EdgeWeight(forest)

    if normalise:
        for e in range(forest.n_edges()):
            # we start with the weight outside the parent node
            # and incorporate the edges weight
            w = semiring.times(node_reversed_values[forest.head(e)], omega.value(e))
            # we then incorporate the inside weight of each child node
            for child in forest.tail(e):
                w = semiring.times(w, node_values[child])

            # and finally normalise the edge's weight by its head inside
            edge_expec[e] = semiring.divide(w, node_values[forest.head(e)])
    else:
        for e in range(forest.n_edges()):
            # we start with the weight outside the parent node
            # and incorporate the edges weight
            w = semiring.times(node_reversed_values[forest.head(e)], omega.value(e))
            # we then incorporate the inside weight of each child node
            for child in forest.tail(e):
                w = semiring.times(w, node_values[child])
            edge_expec[e] = w

    return edge_expec


cdef class EdgeValues(ValueFunction):
    """
    Compute the value associated with edges given node values and rule values.
    """

    def __init__(self, Hypergraph forest,
                 Semiring semiring,
                 weight_t[::1] node_values,
                 ValueFunction omega=None,
                 normalise=False):
        """
        :param forest: a hypergraph
        :param semiring: a semiring
        :param node_values: the values associated with nodes (in the given semiring)
        :param omega: a weight function over edges
        :param normalise: whether to normalise an edge's value by its head node's value.
        """

        self._forest = forest
        self._semiring = semiring
        self._node_values = node_values
        self._normalise = normalise

        if omega is None:
            self._omega = EdgeWeight(forest)
        else:
            self._omega = omega

        self._edge_values = compute_edge_values(self._forest,
                                                self._semiring,
                                                self._node_values,
                                                self._omega,
                                                self._normalise)

    cpdef weight_t value(self, id_t e):
        return self._edge_values[e]


cdef class LazyEdgeValues(ValueFunction):
    """
    In some cases, such as in slice sampling, we are unlikely to visit every edge.
    Thus lazily computing edge values might be appropriate.
    """

    def __init__(self, Hypergraph forest,
                 Semiring semiring,
                 weight_t[::1] node_values,
                 ValueFunction omega=None,
                 normalise=False):
        """
        :param forest: a hypergraph
        :param semiring: a semiring
        :param node_values: the values associated with nodes (in the given semiring)
        :param omega: a weight function over edges
        :param normalise: whether to normalise an edge's value by its head node's value.
        """

        self._forest = forest
        self._semiring = semiring
        self._node_values = node_values
        self._edge_values = [None] * forest.n_edges()

        if omega is None:
            self._omega = EdgeWeight(forest)
        else:
            self._omega = omega

        self._normalise = normalise

    cdef weight_t _unnormalised(self, id_t e):
        cdef id_t child
        cdef weight_t tail_value = self._semiring.times.reduce([self._node_values[child]
                                                                for child in self._forest.tail(e)])
        return self._semiring.times(self._omega.value(e), tail_value)


    cdef _normalised(self, id_t e):
        return self._semiring.divide(self._unnormalised(e),
                                     self._node_values[self._forest.head(e)])

    def __getitem__(self, id_t e):
        return self.value(e)

    cpdef weight_t value(self, id_t e):
        if self._edge_values[e] is None:
            self._edge_values[e] = self._normalised(e) if self._normalise else self._unnormalised(e)
        return self._edge_values[e]

