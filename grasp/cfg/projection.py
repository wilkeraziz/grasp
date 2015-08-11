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

    Q = deque(derivation)

    linear = []

    def linearise():
        rule = Q.popleft()
        for child in rule.rhs:
            if isinstance(child, Terminal):
                linear.append(child)
            else:
                linearise()
    linearise()

    return linear


def robust_make_nltk_tree(derivation, skip=0,
                   t2str=lambda s: s.underlying_str(),
                   nt2str=lambda s: s.underlying_str()):
    """
    Recursively constructs an nlt Tree from a list of rules.
    :param derivation: a sequence of edges
    :param skip: how many levels should be ignored
    """

    def replrb(sym):
        return sym.replace('(', '-LRB-').replace(')', '-RRB-')

    Q = deque(derivation[skip:])

    def make_tree(sym):
        if isinstance(sym, Terminal):
            return t2str(sym)
        rule = Q.popleft()
        rhs = []
        for child in rule.rhs:
            rhs.append(make_tree(child))
        return Tree(replrb(nt2str(rule.lhs)), rhs)

    return make_tree(derivation[skip].lhs)


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
        return inlinetree(robust_make_nltk_tree(derivation, nt2str=lambda s: s.base.underlying_str()))

    @staticmethod
    def string(derivation):
        return ' '.join(t.surface for t in robust_get_leaves(derivation))