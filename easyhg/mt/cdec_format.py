"""
:Authors: - Wilker Aziz
"""

from collections import defaultdict
from easyhg.grammar.symbol import Terminal, Nonterminal
from easyhg.grammar.rule import SCFGProduction
from easyhg.grammar.scfg import SCFG
from easyhg.recipes import smart_ropen


def is_nonterminal(sym):
    return sym.startswith('[') and sym.endswith(']')


def iterpairs(features_str):
    for kv in features_str.split():
            k, v = kv.split('=')
            yield k, float(v)


def iterrules(istream):
    """
    Iterates through an input stream yielding synchronous rules.
    :param istream:
    :param linear_model:
    :return:
    """
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
        yield SCFGProduction.create(lhs, f_rhs, e_rhs, features)


def load_grammar(path):
    """
    Load a grammar from a text file.
    :param path:
    :return:
    """
    return SCFG(iterrules(smart_ropen(path)))


if __name__ == '__main__':
    import sys
    from easyhg.grammar.semiring import SumTimes
    G = SCFG(iterrules(sys.stdin))
    print('# G')
    print(G)
    #print('# F')
    #print(G.input_projection(SumTimes))
