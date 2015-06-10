"""
@author wilkeraziz
"""

from collections import deque

def count_derivations(wcfg, root):
    """
    This code does not check for infinite recursions, thus you should use it only with (finite) forests.
    """
    
    def recursion(derivation, projection, Q, wcfg, counts):
        if Q:
            sym = Q.popleft()
            if isinstance(sym, Terminal):
                recursion(derivation, [sym] + projection, Q, wcfg, counts)
            else:
                for rule in wcfg[sym]:
                    copy_Q = deque(Q)
                    copy_Q.extendleft(rule.rhs)
                    recursion(derivation + [rule], projection, copy_Q, wcfg, counts)
        else:
            counts['d'][tuple(derivation)] += 1
            counts['p'][tuple(projection)] += 1

    counts = {'d': defaultdict(int), 'p': defaultdict(int)}
    recursion([], [], deque([root]), wcfg, counts)
    return counts


