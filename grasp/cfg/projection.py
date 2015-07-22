from . import Terminal
from collections import defaultdict
from .utils import inlinetree, make_nltk_tree

_LB_ = '('
_RB_ = ')'


def get_leaves(derivation):
    """Return the yield of a derivation as a sequence of Terminal symbols."""
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


class ItemDerivationYield:

    @staticmethod
    def string(head, tail, ants, Y):
        """a `flat` projection is nothing but the string yield of a derivation"""
        for i, u in enumerate(tail):
            if isinstance(u, Terminal):
                Y.append(u)
            else:
                Y.extend(ants[i])

    @staticmethod
    def bracketing(head, tail, ants, Y):
        """a bracketed projection is nothing but the string yield of a derivation marked with phrase boundaries"""
        Y.append(_LB_)
        for i, u in enumerate(tail):
            if isinstance(u, Terminal):
                Y.append(u)
            else:
                Y.extend(ants[i])
        Y.append(_RB_)

    @staticmethod
    def tree(head, tail, ants, Y):
        """a labelled projection is the labelled bracketing (a tree)"""
        Y.append('{0}{1}'.format(_LB_, head.label))
        for i, u in enumerate(tail):
            if isinstance(u, Terminal):
                Y.append(u)
            else:
                Y.extend(ants[i])
        Y.append(_RB_)


class DerivationYield:

    @staticmethod
    def derivation(derivation):
        return inlinetree(make_nltk_tree(derivation))

    @staticmethod
    def tree(derivation):
        return inlinetree(make_nltk_tree(derivation, nt2str=lambda s: s.base.underlying_str()))

    @staticmethod
    def string(derivation):
        return ' '.join(t.surface for t in get_leaves(derivation))