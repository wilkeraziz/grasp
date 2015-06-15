"""
:Authors: - Wilker Aziz
"""

from collections import defaultdict
from .symbol import Terminal, Nonterminal
from .rule import SCFGProduction
from .scfg import SCFG
from .utils import smart_open


def is_nonterminal(sym):
    return sym.startswith('[') and sym.endswith(']')


def iterpairs(features_str):
    for kv in features_str.split():
            k, v = kv.split('=')
            yield k, float(v)


def iterrules(path, linear_model):
    istream = smart_open(path)
    for line in istream:
        if line.startswith('#'):
            continue
        line = line.strip()
        if not line:
            continue
        fields = line.split(' ||| ')
        if len(fields) < 4:
            raise ValueError('I expected at least 4 fields, got %d: %s' % (len(fields), fields))
        if not is_nonterminal(fields[0]):
            raise ValueError('Expected a nonterminal LHS, got something else: <%s>' % fields[0])
        lhs = Nonterminal(fields[0][1:-1])  # ignore brackets
        f_rhs = tuple(Nonterminal(x[1:-1]) if is_nonterminal(x) else Terminal(x) for x in fields[1].split())
        e_rhs = tuple(Nonterminal(x[1:-1]) if is_nonterminal(x) else Terminal(x) for x in fields[2].split())
        features = defaultdict(None, iterpairs(fields[3]))
        yield SCFGProduction.create(lhs, f_rhs, e_rhs, linear_model.dot(features))


def load_grammar(path, linear_model):
    return SCFG(iterrules(path, linear_model))


if __name__ == '__main__':
    import sys
    from .semiring import SumTimes
    from .model import cdec_basic
    model = cdec_basic()
    G = SCFG(iterrules(sys.stdin, model))
    print('# G')
    print(G)
    print('# F')
    print(G.f_projection(SumTimes))
    print('# E')
    print(G.e_projection(SumTimes, marginalise=True))