from nltk.tree import Tree
from collections import defaultdict
from .symbol import Terminal
import re
import gzip
import datetime
import tempfile
from io import TextIOWrapper


def smart_ropen(path):
    """Opens files directly or through gzip depending on extension."""
    if path.endswith('.gz'):
        return TextIOWrapper(gzip.open(path, 'rb'))
    else:
        return open(path, 'r')


def smart_wopen(path):
    """Opens files directly or through gzip depending on extension."""
    if path.endswith('.gz'):
        return TextIOWrapper(gzip.open(path, 'wb'))
    else:
        return open(path, 'w')


def _make_nltk_tree(derivation, top=0):
    """
    Recursively constructs an nlt Tree from a list of rules.
    @param top: index to the top rule (0 and -1 are the most common values)
    """ 
    d = defaultdict(None, ((r.lhs, r) for r in derivation))

    def make_tree(sym):
        r = d[sym]
        return Tree(r.lhs, (child if child not in d else make_tree(child) for child in r.rhs))

    return make_tree(derivation[top].lhs)


def make_nltk_tree(derivation, top=0, flatten_symbols=False):
    """
    Recursively constructs an nlt Tree from a list of rules.
    @param top: index to the top rule (0 and -1 are the most common values)
    """

    def replrb(sym):
        return sym.replace('(', '-LRB-').replace(')', '-RRB-')

    def pprint(symbol, flatten_symbol=False):
        if flatten_symbol:
            symbol = symbol.flatten()
        if isinstance(symbol, Terminal):
            return replrb(str(symbol.surface))
        else:
            if not isinstance(symbol.label, tuple):
                return replrb(str(symbol.label))
            else:
                if symbol.label[1] is None and symbol.label[2] is None:
                    return replrb(str(symbol.label[0].label))
                else:
                    return replrb('{0}:{1}-{2}'.format(symbol.label[0].label, symbol.label[1], symbol.label[2]))
        
    d = defaultdict(None, ((r.lhs, r) for r in derivation))

    def make_tree(sym):
        r = d[sym]
        return Tree(pprint(r.lhs, flatten_symbols), (pprint(child, flatten_symbols) if child not in d else make_tree(child) for child in r.rhs))

    return make_tree(derivation[top].lhs)


def inlinetree(t):
    s = str(t).replace('\n', '')
    return re.sub(' +', ' ', s)


def make_unique_directory(dir=None):
    return tempfile.mkdtemp(prefix=datetime.datetime.now().strftime("%y%m%d_%H%M%S_"), dir=dir)

