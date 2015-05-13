"""
@author wilkeraziz
"""
from collections import defaultdict
from itertools import chain


def topsort(dependencies, independent=None):
    """
    Finds a partial ordering of the given objects.
    You may specify which objects are known to be independent or let the algorithm find that out for you.

    >>> # in this case we specify the independent objects
    >>> D = {'S': {'S','X'}, 'X': {'X', 'Y', 'a', 'b', 'c'}, 'Y':{'.'}}
    >>> I = {'a', 'b', 'c', '.'}
    >>> expected = [set(['a', 'b', 'c', '.']), set(['Y']), set(['X']), set(['S'])]
    >>> list(topsort(D, I)) == expected
    True
    >>> # in this case we add all independent items as items with no dependencies
    >>> D = {'S': {'S','X'}, 'X': {'X', 'Y', 'a', 'b', 'c'}, 'Y':{'.'}, 'a':{}, 'b':{}, 'c':{}, '.':{}}
    >>> expected = [set(['a', 'b', 'c', '.']), set(['Y']), set(['X']), set(['S'])]
    >>> list(topsort(D)) == expected
    True
    >>> # in this case we add some independent items as items with no dependencies
    >>> D = {'S': {'S','X'}, 'X': {'X', 'Y', 'a', 'b', 'c'}, 'Y':{'.'}, 'a':{}, 'b':{}}
    >>> expected = [set(['a', 'b', 'c', '.']), set(['Y']), set(['X']), set(['S'])]
    >>> list(topsort(D)) == expected
    True
    """
    dependencies = {k: set(deps) for k, deps in dependencies.iteritems()}

    if independent:  # independent items are given
        ordered = set(independent) 
    else:  # we must find the independent items
        # values with no dependencies
        ordered = set(chain(*dependencies.itervalues())) - set(dependencies.iterkeys())
        # keys with no dependencies
        ordered.update(k for k, deps in dependencies.iteritems() if len(deps) == 0)
    
    # ignore self dependencies
    [deps.discard(k) for k, deps in dependencies.iteritems()]
   
    # sort
    while ordered:
        yield ordered
        # remove items that have already been ordered
        for o in ordered:
            dependencies.pop(o, None)
        # update dependencies
        dependencies = {k: (deps - ordered) for k, deps in dependencies.iteritems()}
        # check which items had their dependencies all sorted
        ordered = set(k for k, deps in dependencies.iteritems() if len(deps) == 0)  # items with no dependencies
    
    if dependencies:
        raise ValueError('Cyclic dependencies were encountered: %s' % dependencies)


def topsort_cfg(cfg):
    """
    Finds a partial ordering of the symbols in a CFG.
    The algorithm will throw an exception if a cycle is detected (except for self dependencies of the kind 'S -> S X').

    >>> from symbol import Terminal, Nonterminal
    >>> from ply_cfg import read_grammar
    >>> cfg = read_grammar(_EXAMPLE_GRAMAR_)
    >>> order = list(topsort_cfg(cfg))
    >>> len(order)
    3
    >>> expected = [set([Terminal('a'), Terminal('b'), Terminal('c'), Terminal('d'), Terminal('e')]), set([Nonterminal('X'), Nonterminal('Y')]), set([Nonterminal('S')])]
    >>> order == expected
    True
    """
    # make dependencies
    D = defaultdict(set)  
    for v in cfg.nonterminals:
        deps = D[v]
        for r in cfg.iterrules(v):
            deps.update(r.rhs)
    
    return topsort(D, cfg.terminals)


_EXAMPLE_GRAMAR_ = ["[S] ||| [X] ||| 1.0", 
        "[X] ||| 'a' [X] 'c' [X] 'e' ||| 1.0",
        "[X] ||| 'b' ||| 1.0", 
        "[X] ||| 'd' ||| 1.0", 
        "[Y] ||| 'a' ||| 1.0", 
        "[S] ||| [S] [X] ||| 1.0",
        "[S] ||| [Y] [X] ||| 1.0",
        ]


