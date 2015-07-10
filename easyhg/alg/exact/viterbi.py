"""
Find the best path in a forest using the Inside recursion with the max-times semiring.

:Authors: - Wilker Aziz
"""

from easyhg.grammar.semiring import MaxTimes
from .inference import robust_inside, optimise


def viterbi(forest, tsort, omega=lambda e: e.weight, generations=10, semiring=MaxTimes):
    """
    Viterbi derivation using the MaxTimes semiring.

    :param state: ParserState
    :return: a Result object containing the Viterbi derivation
    """
    root = tsort.root()
    inside_nodes = robust_inside(forest, tsort, semiring, omega=omega, infinity=generations)
    d = optimise(forest, root, semiring, Iv=inside_nodes)
    return d, inside_nodes[root]