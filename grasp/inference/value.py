"""
This implements the value recursion for numerical semirings.

    V(v) = \bigoplus_{e \in BS(v)} \omega(e) \bigotimes_{u \in tail(e)} V(u)

We also have an implementation which is robust to the presence of cycles.

:Authors: - Wilker Aziz
"""

from collections import defaultdict
from functools import reduce
from grasp.semiring import SumTimes


def derivation_value(derivation, semiring=SumTimes, Z=None, omega=lambda e: e.weight):
    """
    Compute the total weight of a derivation (as a sequence of edges) under a semiring.

    :param derivation: sequence of edges
    :param semiring: the given semiring (requires divide if Z is given)
    :param Z: the normalisation constant (in the given semiring)
    :param omega: a function over edges
    :return:
    """

    if not derivation:
        return semiring.one  # or zero?
    if Z is None:
        return semiring.times.reduce([omega(e) for e in derivation])
    else:
        return semiring.divide(semiring.times.reduce([omega(e) for e in derivation]), Z)


def acyclic_value_recursion(forest,
                            topsorted,
                            semiring,
                            omega=lambda e: e.weight,
                            infinity=1):
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
    I = defaultdict(None)
    # we go bottom-up
    for parent in topsorted:  # the inside of a node
        incoming = forest.get(parent, set())
        if not incoming:  # a terminal node
            I[parent] = semiring.one if forest.is_terminal(parent) else semiring.zero
            continue
        # the inside of a nonterminal node is a sum over all of its incoming edges (rewrites)
        # for each rewriting rule, we get the product of the RHS nodes' insides times the rule weight
        # partials = (reduce(semiring.times, (I[child] for child in rule.rhs), omega(rule)) for rule in incoming)
        partials = [semiring.times(omega(rule),
                                   semiring.times.reduce(list(I[child] for child in rule.rhs)))
                    for rule in incoming]
        I[parent] = semiring.plus.reduce(partials)
        #I[parent] = reduce(semiring.plus, partials, semiring.zero)
    return I


def robust_value_recursion(forest,
                           tsort,
                           semiring,
                           omega=lambda e: e.weight,
                           infinity=20):
    """
    Returns items' values in a given semiring.
    This is a bottom-up pass through the forest which runs in O(|forest|).
    :param forest: a hypergraph-like object.
    :param tsort: a TopSortTable object.
    :param semiring: must define zero, one, sum and times.
    :param omega: a function that weighs edges/rules (defaults to the edge's weight).
        You might want to use this to, for instance, convert between semirings.
    :param infinity: the maximum number of generations in supremum computations.
    :return:
    """
    I = defaultdict(lambda: semiring.one)
    # we go bottom-up
    for bucket in tsort.iterbuckets(skip=1):  # we skip the terminals
        if len(bucket) == 1:  # non-loopy
            parent = next(iter(bucket))
            incoming = forest.get(parent, set())
            if not incoming:  # a terminal node
                I[parent] = semiring.one if forest.is_terminal(parent) else semiring.zero
                continue
            # the inside of a nonterminal node is a sum over all of its incoming edges (rewrites)
            # for each rewriting rule, we get the product of the RHS nodes' insides times the rule weight
            partials = (reduce(semiring.times, (I[child] for child in rule.rhs), omega(rule)) for rule in incoming)
            I[parent] = reduce(semiring.plus, partials, semiring.zero)
        else:
            V = defaultdict(lambda: semiring.zero)  # this will hold partial inside values for loopy nodes
            for g in range(infinity):  # we iterate to "infinity"
                _V = defaultdict(lambda: semiring.zero)  # this is the current generation
                for parent in bucket:
                    incoming = forest.get(parent, set())
                    if not incoming:
                        value = semiring.one if forest.is_terminal(parent) else semiring.zero
                    else:
                        partials = (reduce(semiring.times,
                                           (V[child] if child in bucket else I[child] for child in rule.rhs),
                                           omega(rule))
                                    for rule in incoming)
                        value = reduce(semiring.plus, partials, semiring.zero)
                    _V[parent] = value
                if V == _V:
                    # if the values of all items have remained unchanged,
                    # they have all converged to their supremum values
                    break
                else:
                    V = _V
            for node, value in V.items():
                I[node] = value
    return I


def compute_edge_values(forest, semiring, node_values, omega=lambda e: e.weight, normalise=False):
    """
    Return the normalised inside weights of the edges in a forest.
    Normalisation happens with respect to an edge's head inside weight.
    @param node_values: inside of nodes
    @param semiring: requires times and divide
    @param omega: a function that weighs edges/rules (serves as a bypass)
    """
    if normalise:
        return defaultdict(None, ((edge, semiring.divide(reduce(semiring.times,
                                                                (node_values[s] for s in edge.rhs),
                                                                omega(edge)),
                                                         node_values[edge.lhs]))
                                  for edge in forest))
    else:
        return defaultdict(None, ((edge, reduce(semiring.times,
                                                (node_values[s] for s in edge.rhs),
                                                omega(edge)))
                                  for edge in forest))


class LazyEdgeValues(object):
    """
    In some cases, such as in slice sampling, we are unlikely to visit every edge.
    Thus lazily computing edge values might be appropriate.
    """

    def __init__(self, semiring,
                 node_values,
                 edge_values={},
                 omega=lambda e: e.weight,
                 normalise=False):
        """
        :param semiring: a semiring
        :param node_values: the values associated with nodes (in the given semiring)
        :param edge_values: the values associated with edges (in the given semiring)
        :param omega: a weight function over edges
        :param normalise: whether to normalise an edge's value by its head node's value.
        """
        self._semiring = semiring
        self._node_values = node_values
        self._edge_values = defaultdict(None, edge_values)
        self._omega = omega
        if normalise:
            self._compute = self._normalised
        else:
            self._compute = self._unnormalised

    def _normalised(self, edge):
        tail_value = self._semiring.times.reduce(list(self._node_values[s] for s in edge.rhs))
        edge_value = self._semiring.times(self._omega(edge), tail_value)
        return self._semiring.divide(edge_value, self._node_values[edge.lhs])

    def _unnormalised(self, edge):
        tail_value = self._semiring.times.reduce(list(self._node_values[s] for s in edge.rhs))
        return self._semiring.times(self._omega(edge), tail_value)

    def __getitem__(self, edge):
        w = self._edge_values.get(edge, None)
        if w is None:
            w = self._compute(edge)
            self._edge_values[edge] = w
        return w