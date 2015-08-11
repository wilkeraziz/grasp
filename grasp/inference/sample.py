"""
This module implements the Inside-Outside algorithm (Baker, 1970) and inference agorithms related to those quantities.

It implements 
    i) PCFG sampling (Chappelier and Rajman, 2000)
    ii) the Viterbi algorithm

@author wilkeraziz
"""

from collections import deque
from .value import LazyEdgeValues
import logging


def sample_one(forest, root, semiring, node_values, edge_values=None, omega=lambda e: e.weight):
    """
    Sample (in a general sense which depends on the semiring) one derivation.

    :param forest: a CFG
    :param root: the forest's start symbol
    :param semiring: the given semiring
    :param node_values: values (in the given semiring) associated with nodes
    :param edge_values: values (in the given semiring) associated with edges
    :param omega: a function over edges
    :return: a derivation
    """
    if edge_values is None:
        edge_values = LazyEdgeValues(semiring, node_values, omega=omega, normalise=not semiring.IDEMPOTENT)

    derivation = []
    Q = deque([root])
    while Q:
        parent = Q.popleft()
        incoming = list(forest.get(parent, set()))
        if not incoming:  # terminal node
            continue
        try:
            edge = semiring.choice(incoming, [edge_values[e] for e in incoming])
        except ValueError as ex:
            logging.error('It seems that the normalised inside weights of incoming edges to %s do not sum to 1' % parent)
            raise ex
        derivation.append(edge)
        Q.extend(edge.rhs)
    return tuple(derivation)


def sample_k(forest, root, semiring, node_values, edge_values=None, omega=lambda e: e.weight, n_samples=1):
    """
    Return a generator for k sample derivations.
    Equivalent to k calls to `sample_one`.

    Note that this code will *not* produce k-best derivations if you use the max-times (or another idempotent) semiring.
    Instead, it will return k times the best derivation.
    For k-best derivations see the `kbest` module.

    :param forest: a CFG
    :param root: the forest's start symbol
    :param semiring: the given semiring
    :param node_values: values (in the given semiring) associated with nodes
    :param edge_values: values (in the given semiring) associated with edges (optional)
    :param omega: a function over edges
    :param n_samples: number of samples
    :return: a generator
    """
    if edge_values is None:
        edge_values = LazyEdgeValues(semiring, node_values, omega=omega, normalise=not semiring.IDEMPOTENT)

    if n_samples < 0:
        while True:
            yield sample_one(forest, root, semiring, node_values, edge_values, omega)
    else:
        for i in range(n_samples):
            yield sample_one(forest, root, semiring, node_values, edge_values, omega)