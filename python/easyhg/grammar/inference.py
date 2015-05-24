"""
This module implements the Inside-Outside algorithm (Baker, 1970) and inference agorithms related to those quantities.

It implements 
    i) PCFG sampling (Chappelier and Rajman, 2000)
    ii) the Viterbi algorithm

@author wilkeraziz
"""

import numpy as np
from itertools import ifilter
from collections import defaultdict, deque


def inside(forest, topsorted, semiring, omega=lambda e: e.weight):
    """
    Returns inside weights in a given semiring.
    This is a bottom-up pass through the forest, thus runs in O(|forest|).
    @param omega: a function that weighs edges/rules (serves as a bypass)
    """
    I = defaultdict(None)
    # we go bottom-up
    for parent in topsorted:  # the inside of a node
        incoming = forest.get(parent, None)
        if incoming is None:  # a terminal node
            I[parent] = semiring.one
            continue
        # the inside of a nonterminal node is a sum over all of its incoming edges (rewrites)
        # for each rewriting rule, we get the product of the RHS nodes' insides times the rule weight
        partials = (reduce(semiring.times, (I[child] for child in rule.rhs), omega(rule)) for rule in forest.get(parent, set()))
        I[parent] = reduce(semiring.plus, partials, semiring.zero)
    return I


def outside(forest, topsorted, semiring, I=None, omega=lambda e: e.weight):
    """
    Returns otuside weights in a given semiring (it also computes inside weights, unless given).
    This is a top-down pass through the forest, thus runs in O(|forest|).
    @param omega: a function that weighs edges/rules (serves as a bypass)
    """
    topsorted = list(topsorted)
    if I is None:
        I = inside(forest, topsorted, semiring, omega)
    O = defaultdict(lambda : semiring.zero)
    O[topsorted[-1]] = semiring.one  # this assumes a single root node
    # we go top-down
    for parent in reversed(topsorted):
        for rule in forest.get(parent, set()):  # for each of its rewrites
            for child in rule.rhs:  # the outside of a node 
                # includes the outside of that node's parent times the weight of the rule (1)
                # times the product of its siblings' inside weights (2)
                partial = reduce(semiring.times, 
                        (I[sibling] for sibling in rule.rhs if sibling != child),         # (2)
                        semiring.times(omega(rule), O[parent])                            # (1)
                        )
                # we accumulated that quantity over every possible rule whose RHS contain that node
                O[child] = semiring.plus(O[child], partial)
    return O


def normalised_edge_inside(forest, I, semiring, omega=lambda e: e.weight):
    """
    Return the normalised inside weights of the edges in a forest.
    Normalisation happens with respect to an edge's head inside weight.
    @param I: inside of nodes
    @param semiring: requires times and divide
    @param omega: a function that weighs edges/rules (serves as a bypass)
    """
    return defaultdict(None, ((edge, semiring.divide(reduce(semiring.times, (I[s] for s in edge.rhs), omega(edge)), I[edge.lhs])) for edge in forest))


class LazyEdgeInside(object):
    """
    In some cases, such as in slice sampling, we are unlikely to visit every edge, thus lazily computing the inside of edges might be appropriate.
    """

    def __init__(self, semiring, Iv, Ie={}, omega=lambda e: e.weight):
        """
        @param semiring: defining times and divide (if normalisation is required)
        @param Iv: inside for nodes
        @param Ie: inside for edges (may be initialised)
        @param omega: a function that weighs edges/rules (serves as a bypass)
        """
        self._semiring = semiring
        self._Iv = Iv
        self._Ie = defaultdict(None, Ie)
        self._omega = omega

    def __getitem__(self, edge):
        w = self._Ie.get(edge, None)
        if w is None:
            w = self._semiring.divide(reduce(self._semiring.times, (self._Iv[s] for s in edge.rhs), self._omega(edge)), self._Iv[edge.lhs])
            self._Ie[edge] = w
        return w


def sample(forest, root, semiring, Iv, Ie=None, N=1, omega=lambda e: e.weight):
    """
    Returns a generator for random samples

    @param forest
    @param root: where to start sampling from
    @param semiring: plus is necessary 
    @param Iv: inside for nodes
    @param Ie: if provided, should be normalised wrt to head nodes
    @param N: number of samples (use a negative number to sample for ever)
    @param omega: a function that weighs edges/rules (serves as a bypass)
    """
    if Ie is None:
        Ie = LazyEdgeInside(semiring, Iv, omega=omega)  # we require normalisation

    def sample_edge(edges):
        edges = list(edges)
        i = np.random.choice(len(edges), p=[semiring.as_real(Ie[e]) for e in edges])
        return edges[i]
    
    def sample_derivation():
        derivation = []
        Q = deque([root])
        while Q:
            parent = Q.popleft()
            incoming = forest.get(parent, None)
            if incoming is None:  # terminal node
                continue  
            edge = sample_edge(incoming)
            derivation.append(edge)
            Q.extend(edge.rhs)
        return tuple(derivation) #, semiring.divide(reduce(semiring.times, (e.weight for e in derivation), semiring.one), Iv[root])

    if N < 0:
        while True:
            yield sample_derivation()
    else:
        for i in range(N):
            yield sample_derivation()


def optimise(forest, root, semiring, Iv, Ie=None, omega=lambda e: e.weight, maximisation=True):
    """
    Returns a generator for random samples

    @param forest
    @param root: where to start sampling from
    @param semiring: plus and times are necessary 
    @param Iv: inside for nodes
    @param Ie: if provided, should be normalised wrt to head nodes
    @param omega: a function that weighs edges/rules (serves as a bypass)
    @param maximisation: whether we solve a maximisation or a minimisation problem
    """
    if Ie is None:
        Ie = LazyEdgeInside(semiring, Iv, omega=omega)  # we require normalisation

    choice = max if maximisation else min

    derivation = []
    Q = deque([root])
    while Q:
        parent = Q.popleft()
        incoming = forest.get(parent, None)
        if incoming is None:  # terminal node
            continue  
        edge = choice(incoming, key=lambda e: Ie[e])
        derivation.append(edge)
        Q.extend(edge.rhs)
    return tuple(derivation) #, Iv[root]


def total_weight(derivation, semiring, Z=None):
    """
    Compute the total weight of a derivation (as a sequence of edges) under a semiring
    @params derivation: sequence of edges
    @params semiring: requires `one` and `times` (and may require `divide`, see Z below)
    @params Z: inside of the root node, if provided, the total weight will be normalised 
    """
    if Z is None:
        return reduce(semiring.times, (e.weight for e in derivation), semiring.one)
    else:
        return semiring.divide(reduce(semiring.times, (e.weight for e in derivation), semiring.one), Z)
