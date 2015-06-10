"""
Read grammars encoded in discodop's format.

:Authors: - Wilker Aziz
"""

import sys
import numpy as np
from itertools import chain
from .symbol import Terminal, Nonterminal
from .rule import CFGProduction
from .cfg import CFG
from .utils import smart_open


def iterrules(path, transform):
    fi = smart_open(path, 'rt')
    for line in fi:
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        lhs = fields[0]
        (num, den) = fields[-1].split('/')
        num = float(num)
        den = float(den)
        rhs = fields[1:-2]  # fields[-2] is the yield function, which we are ignoring
        #if len(rhs) != 2:
        #    logging.debug('Unary rule: %s %s' % (lhs, rhs))
        yield CFGProduction(Nonterminal(lhs), [Nonterminal(s) for s in rhs], transform(num/den))


def iterlexicon(path, transform):
    fi = smart_open(path, 'rt')
    for line in fi:
        line = line.strip()
        if not line:
            continue
        fields = line.split('\t')
        word = fields[0]
        for pair in fields[1:]:
            tag, fraction = pair.split(' ')
            num, den = fraction.split('/')
            num = float(num)
            den = float(den)
            r = CFGProduction(Nonterminal(tag), [Terminal(word)], transform(num/den))
            yield r


def read_grammar(rules_file, lexicon_file, transform=np.log):
    return CFG(chain(iterrules(rules_file, transform), iterlexicon(lexicon_file, transform)))

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: %s rules lexicon' % (sys.argv[0]), file=sys.stderr)
        sys.exit(0)
    print(read_grammar(sys.argv[1], sys.argv[2]))
