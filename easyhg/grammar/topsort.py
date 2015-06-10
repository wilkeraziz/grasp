"""
This is a standard implementation of topological sorting.
Dependencies are encoded through a dictionary and a partial ordering generator is returned.

:Authors: - Wilker Aziz
"""

from collections import defaultdict
from itertools import chain


def topsort(dependencies, independent=None):
    """Find a partial ordering of the given objects.
    
    :param dependencies:
        dictionary o dependencies (object -> set of dependencies).
    :param independent: 
        independent objects.
    :returns:
        a generator which produces groups of objects in bottom-up order

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
    dependencies = {k: set(deps) for k, deps in dependencies.items()}

    if independent:  # independent items are given
        ordered = set(independent) 
    else:  # we must find the independent items
        # values with no dependencies
        ordered = set(chain(*iter(dependencies.values()))) - set(dependencies.keys())
        # keys with no dependencies
        ordered.update(k for k, deps in dependencies.items() if len(deps) == 0)
    
    # ignore self dependencies
    [deps.discard(k) for k, deps in dependencies.items()]
   
    # sort
    while ordered:
        yield ordered
        # remove items that have already been ordered
        for o in ordered:
            dependencies.pop(o, None)
        # update dependencies
        dependencies = {k: (deps - ordered) for k, deps in dependencies.items()}
        # check which items had their dependencies all sorted
        ordered = set(k for k, deps in dependencies.items() if len(deps) == 0)  # items with no dependencies
    
    if dependencies:
        raise ValueError('Cyclic or incomplete dependencies were encountered: %s' % dependencies)




_EXAMPLE_GRAMAR_ = ["[S] ||| [X] ||| 1.0", 
        "[X] ||| 'a' [X] 'c' [X] 'e' ||| 1.0",
        "[X] ||| 'b' ||| 1.0", 
        "[X] ||| 'd' ||| 1.0", 
        "[Y] ||| 'a' ||| 1.0", 
        "[S] ||| [S] [X] ||| 1.0",
        "[S] ||| [Y] [X] ||| 1.0",
        ]


