"""
This is a standard implementation of topological sorting.
Dependencies are encoded through a dictionary and a partial ordering generator is returned.

:Authors: - Wilker Aziz
"""

from collections import deque, defaultdict, Counter


def deps_assert(deps):
    for syms in deps.values():
        for sym in syms:
            if sym not in deps:
                return False
    return True


def topological_sort(deps):
    """
    A topological sorting algorithm for acyclic graphs.
    Unlike other topsort algorithms, here we return ordered sets where elements within a set have the same level.

    For cyclic graphs see `robust_topological_sort'.

    :param deps: a dictionary representing dependencies between nodes.
        Every node must be associated with a set of dependencies (even if empty).

    :return: sorted groups (a group is a frozenset of nodes)

    >>> deps1 = {'S':{'X','Y','Z'}, 'X':{'Y'}, 'Z':{'Y'}, 'Y':{}}
    >>> topological_sort(deps1) == [{'Y'}, {'Z', 'X'}, {'S'}]
    True
    """
    #assert deps_assert(deps), 'Expected all items to have dependencies, but found some that didnt.'

    # count the number of nodes depending on each node
    count = Counter()
    for v, tail in deps.items():
        count.update(tail)

    group = deps.keys() - count.keys()  # nodes with zero dependencies

    order = deque()
    while group:
        order.append(frozenset(group))
        group = set()
        # decrements dependencies
        for v in order[-1]:
            for u in deps[v]:
                n = count.get(u)
                if n == 1:  # last dependency
                    del count[u]
                    group.add(u)
                else:
                    count[u] -= 1

    if count:
        raise ValueError('%d cyclic or incomplete dependencies were encountered: %s' % (len(count), count))

    order.reverse()

    return order


class Node:
    """
    Helper data structure used by Tarjan's algorithm.
    """

    def __init__(self, label: 'a node identifier'):
        self.label = label
        self.index = None
        self.low = None
        self.stacked = False


def tarjan(deps):
    """
    Tarjan's strongly connected components algorithm.

    A strongly connected component is a set of nodes defining a connected subgraph, that is,
    every node in the set is reachable from every other node in the set.

    :param deps: a dictionary representing the dependencies between nodes.
        Every node must be associated with a set of dependencies (even if empty).

    :return: strongly connected components (list of sets)

    To use with CFGs it is recommended to ignore terminals
    >>> deps1 = {'S':{'S','X'}, 'X':{'Y'}, 'Y':{'Z'}, 'Z':{'X'}}
    >>> tarjan(deps1) == [{'X', 'Y', 'Z'}, {'S'}]
    True

    This returns a partial ordering, thus sibling strongly connected components (sets/buckets) are returned in
     arbitrary order.
    >>> deps1 = {'S':{'S','X', 'A'}, 'X':{'Y', 'B'}, 'Y':{'Z'}, 'Z':{'X'}, 'A':{'B'}, 'B':{}}
    >>> order = tarjan(deps1)
    >>> order.index({'B'}) < order.index({'A'}) < order.index({'S'})
    True
    >>> order.index({'X', 'Y', 'Z'}) < order.index({'S'})
    True
    >>> order.index({'B'}) < order.index({'X', 'Y', 'Z'})
    True

    One can also consider the terminals, but in this case, the empty dependencies of terminals must be explicitly given.
    >>> deps2 = {'TOP':{'BOS', 'S', 'EOS'}, 'S':{'S', 'X'}, 'X':{'a', 'b', 'c', 'Y'}, 'Y':{'Z', 'y'}, 'Z':{'X', 'z'}, 'a':{}, 'b':{}, 'c':{}, 'y':{}, 'z':{}, 'BOS':{}, 'EOS':{}}
    >>> order = tarjan(deps2)
    >>> order.index({'X', 'Y', 'Z'}) < order.index({'S'}) < order.index({'TOP'})
    True
    """

    n2i = defaultdict(None, ((v, i) for i, v in enumerate(deps.keys())))
    E = defaultdict(set, ((n2i[h], frozenset(n2i[s] for s in tail)) for h, tail in deps.items()))
    V = [Node(v) for v in deps.keys()]
    stack = deque()
    index = 0
    order = deque()

    def strongconnect(v: 'a node index'):
        nonlocal index
        V[v].index = index
        V[v].low = index
        index += 1
        stack.append(v)
        V[v].stacked = True

        for u in E.get(v, set()):
            if V[u].index is None:
                strongconnect(u)
                V[v].low = min(V[v].low, V[u].low)
            elif V[u].stacked:
                V[v].low = min(V[v].low, V[u].index)

        if V[v].low == V[v].index:
            bucket = set()
            u = -1
            while u != v:
                u = stack.pop()
                V[u].stacked = False
                bucket.add(V[u].label)
            order.append(frozenset(bucket))

    for v in range(len(V)):
        if V[v].index is None:
            strongconnect(v)

    return order


def robust_topological_sort(deps):
    """
    A topological sorting algorithm which is robust enough to handle cyclic graphs.
    First, we bucket nodes into strongly connected components (we use Tarjan's linear algorithm for that).
    Then, we topologically sort these buckets grouping sibling buckets into sets.

    :param deps: a dictionary representing the dependencies between nodes
    :return: groups of buckets (a bucket is a strongly connected component) sorted bottom-up

    >>> deps1 = {'S':{'S','X', 'A'}, 'X':{'Y', 'B'}, 'Y':{'Z'}, 'Z':{'X'}, 'A':{'B'}, 'B':{}}
    >>> expected = [frozenset({frozenset({'B'})}), frozenset({frozenset({'A'}), frozenset({'Y', 'X', 'Z'})}), frozenset({frozenset({'S'})})]
    >>> order = robust_topological_sort(deps1)
    >>> order == expected
    True
    """

    # correspondences between nodes and buckets (strongly connected components)
    n2c = defaultdict(None)
    components = tarjan(deps)
    for i, component in enumerate(components):
        for v in component:
            n2c[v] = i

    # find the dependencies between strongly connected components
    cdeps = defaultdict(set)
    for head, tail in deps.items():
        hc = n2c[head]
        for t in tail:
            tc = n2c[t]
            if hc != tc:
                cdeps[hc].add(tc)

    # topsort buckets and translate bucket ids back into nodes
    return deque(frozenset(components[c] for c in group) for group in topological_sort(cdeps))