"""
This module extracts k-best derivations from a B-hypergraph.
The algorithm is due to Huang and Chiang (2005).


    @inproceedings{Huang+2005:kbest,
        Address = {Stroudsburg, PA, USA},
        Author = {Huang, Liang and Chiang, David},
        Booktitle = {Proceedings of the Ninth International Workshop on Parsing Technology},
        Pages = {53--64},
        Publisher = {Association for Computational Linguistics},
        Series = {Parsing '05},
        Title = {Better K-best Parsing},
        Year = {2005}
        }


:Authors: - Wilker Aziz
"""

import heapq
from collections import deque, defaultdict
from grasp.cfg.projection import ItemDerivationYield
from grasp.semiring import MaxTimes


class Derivation(object):
    """
    A derivation is identified with and edge and a collection of backpointers (for the nodes in its tail).
    This class also stores the total weight of the derivation,
    and possibly its projection (when that information is relevant).
    """

    def __init__(self, edge, J, weight):
        """
        @params edge
        @params J: backpointers
        @param weight: total weight (this is typically different from edge.weight)
        """
        self._edge = edge
        self._J = list(J)
        self._weight = weight
        self._projection = []

    @property
    def edge(self):
        return self._edge

    @property
    def J(self):
        return self._J
    
    @property
    def weight(self):
        return self._weight

    @property
    def projection(self):
        return self._projection

    def __lt__(self, other):  # TODO: check why this is necessary!
        return self.weight > other.weight


class NodeDerivationState(object):
    """
    This class manages the k-best derivations for a certain node.
    This corresponds to \mathbf D(v) in the paper, where v is a node.
    """

    def __init__(self, uniqueness=False):
        """
        @param uniqueness: whether or not we care about uniqueness
        """
        self._candidates = []  # this is the priority queue used to enumerate derivations in best-first order
        self._derivations = deque()  #  these are the sorted derivations 
        self._queueing = set()  # represent derivations which are queueing (or have already left the queue)
        self._projections = None if not uniqueness else set()  # can be used to avoid redundant yields ("hack" typically used in MT to increase diversity)

    @property
    def D(self):
        return self._derivations

    def empty(self):
        """whether the state is empty (i.e. no derivations have been enumerated, nor there are derivations queueing)"""
        return len(self._candidates) + len(self._derivations) == 0

    def __contains__(self, e_J):
        """
        whether a certain derivation (signature) has been seen
        @param e: edge
        @param J: backpointers
        """
        (e, J) = e_J
        return (e, tuple(J)) in self._queueing

    def has_candidates(self):
        return len(self._candidates) > 0

    def make_heap(self, candidates, k):
        """creates a heap of candidates containing up to k elements"""
        if len(candidates) > k:
            self._candidates = heapq.nsmallest(k, candidates)
        else:
            self._candidates = list(candidates)
        heapq.heapify(self._candidates)

    def push(self, w, d):
        """pushes to the heap, note that w must obey the heapq logic for weights, i.e. smallest first"""
        heapq.heappush(self._candidates, (w, d))
        self._queueing.add((d.edge, tuple(d.J)))

    def pop(self):
        """pop the cheapest derivation in the heap (with heapq this means the first item)"""
        (w, d) = heapq.heappop(self._candidates)
        return d

    def is_unique(self, projection):
        """uniqueness test for the given projection"""
        if self._projections is None:  # we do not care about uniqueness
            return True
        else:
            n = len(self._projections)
            self._projections.add(tuple(projection))
            return len(self._projections) > n  # if the projection is unique, the set will be bigger


class KBest(object):
    """
    This implements the complete algorithm.
    """

    def __init__(self, forest, root, k, semiring=MaxTimes, traversal=ItemDerivationYield.string, uniqueness=False):
        """
        @param forest: a hypergraph-like object
        @param root: goal node
        @param k: number of derivations (1-based)
        @param semiring: a maximisation-like semiring (requires `times` and `heapify`)
        @param traversal: a projection algorithm
        @param uniqueness: whether or not we care about uniqueness
        """
        self._forest = forest
        self._root = root
        self._k = k
        self._semiring = semiring
        self._traversal = traversal
        self._uniqueness = uniqueness
        
        # node derivation states
        # one state variable for each node
        self._nds = defaultdict(None, ((v, NodeDerivationState(uniqueness)) for v in forest.itersymbols()))

        # I use the following lambda functions to keep the terminology compatible with that of hypergraphs
        self.weight = lambda r: r.weight  # abstracts the weight of an edge (a rule's weight)
        self.head = lambda r: r.lhs  # abstracts the head of an edge (LHS of rule)
        self.tail = lambda r: r.rhs  # abstracts the tail of an edge (RHS of rule)
        self.arity = lambda r : len(r.rhs)
        self.BS = lambda u: self._forest.get(u, frozenset())

    def create_derivation(self, edge, J):
        """returns a the derivation associated with an edge and backpointers J"""
        w = self.weight(edge)  # the weight of the derivation include that of the edge
        tail = self.tail(edge) 
        for i, u in enumerate(tail):  # for each child node
            if not self.BS(u):  # a source node has no incoming edges
                continue
            ant = self.lazy_kth(u, J[i])  # recursively (and lazily) solve the problem for the child (getting an antecedent)
            if ant is None:
                return None
            w = self._semiring.times(w, ant.weight)  # incorporate the antecedent's weight
        return Derivation(edge, J, w)

    def get_node_derivation_state(self, v):
        """
        In the paper this is called `GetCandidates`, it returns the state associated with node v populating its priority queue.
        Here we enumerate all derivations incoming to v and populate a priority queue.
        @param v: node
        @returns D(v) i.e. NodeDerivationState 
        """
        state = self._nds[v]
        if not state.empty(): 
            return state
        # when there are no derivations or candidates, we must populate the priority queue using the incoming edges
        incoming = self.BS(v)
        C = [None] * len(incoming)
        for i, e in enumerate(incoming):
            d = self.create_derivation(e, [0] * self.arity(e))
            if d is None:
                raise ValueError('Expected a derivation, got None')
            C[i] = (self._semiring.heapify(d.weight), d)  # we "heapify" the weight (this is extremely important! forget that and you might get worst-first)
        # if we don't care about uniqueness, we can keep up to k candidates in the heap
        # otherwise we will need to maintain all candidates as some of them might be redundant in terms of their projections
        effective_k = min(self._k, len(C)) if not self._uniqueness else len(C)
        state.make_heap(C, effective_k)
        return state

    def lazy_next(self, d, state):
        """
        Recursively enumerates the next best derivation for d in D(v).
        """
        tail = self.tail(d.edge)
        for i in range(len(d.J)):
            J = list(d.J)
            J[i] += 1
            ant = self.lazy_kth(tail[i], J[i])
            if ant is None:
                continue
            if (d.edge, J) not in state:  # must avoid repeating computation
                new_d = self.create_derivation(d.edge, J)
                if new_d is not None:
                    state.push(self._semiring.heapify(new_d.weight), new_d)  # never forget to "heapify" the weight!
   
    def lazy_kth(self, v, k):
        """
        Enumerates the k-th best from v.

        @param v: node 
        @param k: kth element (0-based)
        @returns Derivation
        """
        state = self.get_node_derivation_state(v)
        D = state.D
        add_next = True
        while len(D) <= k:
            if add_next and len(D) > 0:  # at least 1 derivation has been enumerated in best-first order
                self.lazy_next(D[-1], state)  # extend/update the heap by enumerating successors of the last derivation
            add_next = False
            while not add_next and state.has_candidates():  # recursively process candidates from the priority queue
                d = state.pop()
                ants = [None] * self.arity(d.edge)
                for i, u in enumerate(self.tail(d.edge)):
                    if not self.BS(u):
                        continue
                    kth = self.lazy_kth(u, d.J[i])
                    if kth is None:
                        raise ValueError('Expected a state, got None')
                    ants[i] = kth.projection
                if not self._uniqueness:  # small optimisation: if we don't care about uniqueness, we do not need to compute a projection
                    D.append(d)
                    add_next = True
                else:  # otherwise
                    self._traversal(self.head(d.edge), self.tail(d.edge), ants, d.projection)  # we compute a projection
                    if state.is_unique(d.projection):  # and test it for uniqueness
                        D.append(d)
                        add_next = True
                    else:
                        self.lazy_next(d, state)
            if not add_next:
                break
        return D[k] if k < len(D) else None

    def do(self):
        """enumerate k-best derivations from the goal node"""
        self.lazy_kth(self._root, self._k - 1)
        return self

    def iterderivations(self):
        """returns an iterator for up to k-best derivations"""
        for k in range(self._k):
            Q = deque([(self._root, k)])
            edges = []
            weights = []
            while Q:
                v, i = Q.popleft()
                state = self._nds[v]
                if not state.D:
                    continue
                if i >= len(state.D):  # user asked for more than one can compute
                    return
                d = state.D[i]
                weights.append(d.weight)
                edges.append(d.edge)
                Q.extend(zip(self.tail(d.edge), d.J))
            yield tuple(edges)  #, weights[0]  # the total weight