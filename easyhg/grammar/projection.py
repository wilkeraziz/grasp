from .symbol import Terminal
from collections import defaultdict

_LB_ = '('
_RB_ = ')'


def string(head, tail, ants, Y):
    """a `flat` projection is nothing but the string yield of a derivation"""
    for i, u in enumerate(tail):
        if isinstance(u, Terminal):
            Y.append(u)
        else:
            Y.extend(ants[i])


def bracketing(head, tail, ants, Y):
    """a bracketed projection is nothing but the string yield of a derivation marked with phrase boundaries"""
    Y.append(_LB_)
    for i, u in enumerate(tail):
        if isinstance(u, Terminal):
            Y.append(u)
        else:
            Y.extend(ants[i])
    Y.append(_RB_)


def tree(head, tail, ants, Y):
    """a labelled projection is the labelled bracketing (a tree)"""
    Y.append('{0}{1}'.format(_LB_, head.label))
    for i, u in enumerate(tail):
        if isinstance(u, Terminal):
            Y.append(u)
        else:
            Y.extend(ants[i])
    Y.append(_RB_)


def get_leaves(derivation):
    d = defaultdict(None, ((r.lhs, r) for r in derivation))
    projection = []

    def visit(sym):
        for child in d[sym].rhs:
            if child not in d:
                projection.append(child)
            else:
                visit(child)

    visit(derivation[0].lhs)

    return tuple(projection)
