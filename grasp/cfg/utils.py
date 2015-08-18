from nltk.tree import Tree
from collections import defaultdict, deque
import re


def make_nltk_tree(derivation, skip=0,
                   t2str=lambda s: s.underlying_str(),
                   nt2str=lambda s: s.underlying_str()):
    """
    Recursively constructs an nlt Tree from a list of rules.
    :param derivation: a sequence of edges
    :param skip: how many levels should be ignored
    """

    def replrb(sym):
        return sym.replace('(', '-LRB-').replace(')', '-RRB-')

    d = defaultdict(None, ((r.lhs, r) for r in derivation[skip:]))

    def make_tree(sym):
        r = d[sym]
        return Tree(replrb(nt2str(r.lhs)),
                    (replrb(t2str(child)) if child not in d else make_tree(child) for child in r.rhs))

    return make_tree(derivation[skip].lhs)


def inlinetree(t):
    s = str(t).replace('\n', '')
    return re.sub(' +', ' ', s)