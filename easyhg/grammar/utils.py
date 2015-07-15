from nltk.tree import Tree
from collections import defaultdict
import re


def make_nltk_tree(derivation, skip=0):
    """
    Recursively constructs an nlt Tree from a list of rules.
    @param top: index to the top rule (0 and -1 are the most common values)
    """

    def replrb(sym):
        return sym.replace('(', '-LRB-').replace(')', '-RRB-')

    d = defaultdict(None, ((r.lhs, r) for r in derivation[skip:]))

    def make_tree(sym):
        r = d[sym]
        return Tree(replrb(r.lhs.underlying_str()), (replrb(child.underlying_str()) if child not in d else make_tree(child) for child in r.rhs))

    return make_tree(derivation[skip].lhs)


def inlinetree(t):
    s = str(t).replace('\n', '')
    return re.sub(' +', ' ', s)