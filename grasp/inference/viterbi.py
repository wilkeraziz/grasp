"""
Find the best path in a forest using the value recursion with the max-times semiring.

:Authors: - Wilker Aziz
"""

from grasp.semiring import MaxTimes
from .value import robust_value_recursion as compute_values
from .sample import sample_one


def viterbi_derivation(forest, tsort, omega=lambda e: e.weight, generations=10, node_values=None, edge_values=None):
    """
    Viterbi derivation using the MaxTimes semiring.

    :param forest: a CFG
    :param tsort:  a TopSortTable
    :param omega: a function over edges
    :param generations:  the maximum number of generations in a value recursion
    :param node_values: values (in the MaxTimes semiring) associated with nodes (optional).
    :param edge_values: values (in the MaxTimes semiring) associated with edges (optional).
    :return: derivation (tuple of edges) and its value
    """

    root = tsort.root()
    if node_values is None:
        node_values = compute_values(forest, tsort, semiring=MaxTimes, omega=omega, infinity=generations)
    d = sample_one(forest, root,
                   semiring=MaxTimes,
                   node_values=node_values,
                   edge_values=edge_values,
                   omega=omega)
    return d