from . import Terminal, Nonterminal
from collections import defaultdict, deque
from .utils import inlinetree
from nltk.tree import Tree

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


def robust_get_leaves(derivation):
    """
    Recursively constructs an nlt Tree from a list of rules.
    :param derivation: a sequence of edges
    :param skip: how many levels should be ignored
    """

    hg = [[] for _ in derivation]
    j = 1
    for i, r in enumerate(derivation):
        for s in r.rhs:
            if isinstance(s, Terminal):
                hg[i].append(-1)
            else:
                hg[i].append(j)
                j += 1
        hg.append(hg)

    leaves = []

    def traverse(head):

        tail = hg[head]
        rhs = []
        for n, sym in zip(tail, derivation[head].rhs):
            if n == -1:
                leaves.append(sym)
            else:
                traverse(n)

    traverse(0)

    return leaves


def robust_make_nltk_tree(derivation, skip=0,
                   t2str=lambda s: str(s),
                   nt2str=lambda s: str(s)):
    """
    Recursively constructs an nlt Tree from a list of rules.
    :param derivation: a sequence of edges
    :param skip: how many levels should be ignored
    """

    def replrb(sym):
        return sym.replace('(', '-LRB-').replace(')', '-RRB-')

    derivation = derivation[skip:]
    hg = [[] for _ in derivation]
    j = 1
    for i, r in enumerate(derivation):
        for s in r.rhs:
            if isinstance(s, Terminal):
                hg[i].append(-1)
            else:
                hg[i].append(j)
                j += 1
        hg.append(hg)

    def make_tree(head):

        tail = hg[head]
        rhs = []
        for n, sym in zip(tail, derivation[head].rhs):
            if n == -1:
                rhs.append(t2str(sym))
            else:
                rhs.append(make_tree(n))
        return Tree(replrb(nt2str(derivation[head].lhs)), rhs)

    return make_tree(0)

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
        return inlinetree(robust_make_nltk_tree(derivation))

    @staticmethod
    def tree(derivation):
        return inlinetree(robust_make_nltk_tree(derivation, nt2str=lambda s: str(s.base)))

    @staticmethod
    def string(derivation):
        return ' '.join(t.surface for t in robust_get_leaves(derivation))