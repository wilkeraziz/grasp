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




_EXAMPLE_GRAMAR_ = ["[S] ||| [X] ||| 1.0", 
        "[X] ||| 'a' [X] 'c' [X] 'e' ||| 1.0",
        "[X] ||| 'b' ||| 1.0", 
        "[X] ||| 'd' ||| 1.0", 
        "[Y] ||| 'a' ||| 1.0", 
        "[S] ||| [S] [X] ||| 1.0",
        "[S] ||| [Y] [X] ||| 1.0",
        ]


