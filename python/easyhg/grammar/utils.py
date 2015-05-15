from nltk.tree import Tree
from collections import defaultdict

def make_nltk_tree(derivation, top=0):
    """
    Recursively constructs an nlt Tree from a list of rules.
    @param top: index to the top rule (0 and -1 are the most common values)
    """ 
    d = defaultdict(None, ((r.lhs, r) for r in derivation))
    def make_tree(sym):
        r = d[sym]
        return Tree(r.lhs, (child if child not in d else make_tree(child) for child in r.rhs))
    return make_tree(derivation[top].lhs)

