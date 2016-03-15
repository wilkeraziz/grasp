"""
Find the best path in a forest using the value recursion with the max-times semiring.

:Authors: - Wilker Aziz
"""

import numpy as np
from libcpp.queue cimport queue
from libcpp.vector cimport vector
import grasp.ptypes as ptypes
import grasp.semiring as semiring
from grasp.formal.topsort cimport AcyclicTopSortTable, RobustTopSortTable
from grasp.formal.wfunc cimport HypergraphLookupFunction, TableLookupFunction, ThresholdFunction
from grasp.alg.value cimport LazyEdgeValues, acyclic_value_recursion, robust_value_recursion, compute_edge_values
from grasp.ptypes cimport weight_t, id_t


cpdef tuple sample(Hypergraph forest,
                   id_t root,
                   Semiring semiring,
                   WeightFunction omega):
    """
    Samples (in a generalised sense) a derivation from the forest.
    The outcome depends on the semiring. For instance, with semiring.viterbi we get the 1-best,
    with semiring.inside we get an independent sample.

    :param forest: a hypergraph
    :param root: its root
    :param semiring: a numerical semiring
    :param omega: the value (in the given semiring) of edges in the forest
    :return: a sequence of edges
    """

    cdef:
        vector[id_t] derivation
        id_t parent, child, e
        size_t i
        queue[id_t] Q
    Q.push(root)
    while not Q.empty():
        parent = Q.front()
        Q.pop()
        if forest.is_source(parent):
            continue
        i = semiring.plus.choice(np.array([omega.value(e) for e in forest.iterbs(parent)],
                                          dtype=ptypes.weight))
        e = forest.bs_i(parent, i)
        derivation.push_back(e)
        for child in forest.tail(e):
            Q.push(child)
    return tuple(derivation)


cpdef list batch_sample(Hypergraph forest,
                        TopSortTable tsort,
                        Semiring semiring,
                        size_t size,
                        WeightFunction omega=None,
                        weight_t[::1] node_values=None,
                        weight_t[::1] edge_values=None):
    """
    1-best derivation in the Viterbi semiring.

    :param forest: a hypergraph
    :param tsort:  topsort table
    :param semiring: a numerical semiring
    :param size: the number of samples
    :param omega: a function over edges
    :param node_values: values (in the Viterbi semiring) associated with nodes (optional).
    :param edge_values: values (in the Viterbi semiring) associated with edges (optional).
    :return: list of derivations (each one is a tuple of edges)
    """

    if omega is None:
        omega = HypergraphLookupFunction(forest)

    if node_values is None:
        if isinstance(tsort, AcyclicTopSortTable):
            node_values = acyclic_value_recursion(forest,
                                                  <AcyclicTopSortTable>tsort,
                                                  semiring,
                                                  omega)
        else:
            node_values = robust_value_recursion(forest,
                                                 <RobustTopSortTable>tsort,
                                                 semiring,
                                                 omega)


    cdef WeightFunction e_omega
    if edge_values is None:
        e_omega = LazyEdgeValues(forest,
                                 semiring,
                                 node_values,
                                 omega,
                                 normalise=not semiring.idempotent)
    else:
        e_omega = TableLookupFunction(edge_values)

    return [sample(forest, tsort.root(), semiring, e_omega) for _ in range(size)]


cpdef tuple viterbi_derivation(Hypergraph forest,
                               TopSortTable tsort,
                               WeightFunction omega=None,
                               weight_t[::1] node_values=None,
                               weight_t[::1] edge_values=None):
    return batch_sample(forest, tsort, semiring.viterbi, 1, omega, node_values, edge_values)[0]


cpdef tuple sample_derivation(Hypergraph forest,
                              TopSortTable tsort,
                              WeightFunction omega=None,
                              weight_t[::1] node_values=None,
                              weight_t[::1] edge_values=None):
    return batch_sample(forest, tsort, semiring.inside, 1, omega, node_values, edge_values)[0]


cpdef list sample_derivations(Hypergraph forest,
                              TopSortTable tsort,
                              size_t size,
                              WeightFunction omega=None,
                              weight_t[::1] node_values=None,
                              weight_t[::1] edge_values=None):
    return batch_sample(forest, tsort, semiring.inside, size, omega, node_values, edge_values)


cdef class DerivationCounter:

    def __init__(self,
                 Hypergraph forest,
                 TopSortTable tsort):
        self._forest = forest
        self._tsort = tsort
        self._omega = HypergraphLookupFunction(forest)
        self._root = self._tsort.root()
        self._counts_computed = False

    cdef void do(self):
        cdef WeightFunction omega
        if not self._counts_computed:

            # this emulates the Counting semiring
            omega = ThresholdFunction(HypergraphLookupFunction(self._forest),
                                      semiring.inside)

            if isinstance(self._tsort, AcyclicTopSortTable):
                self._count_values = acyclic_value_recursion(self._forest,
                                                            <AcyclicTopSortTable>self._tsort,
                                                            semiring.inside,
                                                            omega)
            else:
                # TODO: in the Counting semiring this will not converge if cycles exists, do something about it
                self._count_values = robust_value_recursion(self._forest,
                                                           <RobustTopSortTable>self._tsort,
                                                           semiring.inside,
                                                           omega)
            self._counts_computed = True

    cpdef id_t count(self, id_t node):
        self.do()
        return semiring.inside.as_real(self._count_values[node])

    cpdef id_t n_derivations(self):
        self.do()
        return semiring.inside.as_real(self._count_values[self._root])


cdef class AncestralSampler:

    def __init__(self,
                 Hypergraph forest,
                 TopSortTable tsort,
                 WeightFunction omega=None):
        self._forest = forest
        self._tsort = tsort

        if omega is None:
            omega = HypergraphLookupFunction(forest)
        self._omega = omega


        if isinstance(tsort, AcyclicTopSortTable):
            self._node_values = acyclic_value_recursion(forest,
                                                        <AcyclicTopSortTable>tsort,
                                                        semiring.inside,
                                                        omega)
        else:
            self._node_values = robust_value_recursion(forest,
                                                       <RobustTopSortTable>tsort,
                                                       semiring.inside,
                                                       omega)

        self._edge_values = compute_edge_values(self._forest,
                                                semiring.inside,
                                                self._node_values,
                                                self._omega,
                                                normalise=not semiring.inside.idempotent)

        self._root = self._tsort.root()
        self._counter = DerivationCounter(self._forest, tsort)

    property Z:
        def __get__(self):
            """Return the partition function."""
            return self._node_values[self._root]

    cpdef list sample(self, size_t n):
        """Draw samples from the inverted CDF."""
        return sample_derivations(self._forest,
                                  self._tsort,
                                  n,
                                  self._omega,
                                  self._node_values,
                                  self._edge_values)


    cpdef weight_t prob(self, tuple d):
        cdef weight_t w = self._omega.reduce(semiring.inside.times, d)
        return semiring.inside.as_real(semiring.inside.divide(w, self.Z))

    cpdef int n_derivations(self):
        return self._counter.n_derivations()