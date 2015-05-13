"""
@author wilkeraziz
"""

from collections import defaultdict
from symbol import Terminal
import numpy as np

class WDFSA(object):
    """This is a deterministic wFSA"""

    def __init__(self):
    
        self._arcs = []
        self._initial_states = set()
        self._final_states = set()
        self._vocabulary = set()

    def _create_state(self, state):
        if len(self._arcs) <= state:
            for i in range(len(self._arcs), state + 1):
                self._arcs.append(defaultdict(lambda : defaultdict(float)))
            return True
        return False

    def iterstates(self):
        return xrange(len(self._arcs))

    def add_arc(self, sfrom, sto, symbol, weight):
        self._create_state(sfrom)  # create sfrom if necessary
        self._create_state(sto)  # create sto if necessary
        self._arcs[sfrom][symbol][sto] = weight
        self._vocabulary.add(symbol)

    def make_initial(self, state):
        self._initial_states.add(state)

    def make_final(self, state):
        self._final_states.add(state)

    def get_arcs(self, sfrom, symbol):
        if len(self._arcs) <= sfrom:
            raise ValueError('State %d does not exist' % sfrom)
        return list(self._arcs[sfrom].get(symbol, {}).iteritems())
        
    def path_weight(self, path, semiring): 
        """Returns the weight of a path given by a sequence of tuples of the kind (sfrom, sto, sym)"""
        total = semiring.one
        for (sfrom, sto, sym) in arcs:
            arcs = self._arcs[sfrom].get(sym, None)
            if arcs is None:
                raise ValueError('Invalid transition sfrom=%s sym=%s' % (sfrom, sym))
            w = arcs.get(sto, None)
            if w is None:
                raise ValueError('Invalid transition sfrom=%s sto=%s sym=%s' % (sfrom, sto, sym))
            total = semiring.times(total, w)
        return total
    
    def arc_weight(self, sfrom, sto, sym):
        """Returns the weight of an arc"""
        if not (0 <= sfrom < len(self._arcs)):
            raise ValueError('Unknown state sfrom=%s' % (sfrom))
        arcs = self._arcs[sfrom].get(sym, None)
        if arcs is None:
            raise ValueError('Invalid transition sfrom=%s sym=%s' % (sfrom, sym))
        w = arcs.get(sto, None)
        if w is None:
            raise ValueError('Invalid transition sfrom=%s sto=%s sym=%s' % (sfrom, sto, sym))
        return w
        
    @property
    def initial_states(self):
        return self._initial_states
    
    @property
    def final_states(self):
        return self._final_states

    @property
    def vocabulary(self):
        return self._vocabulary

    def __str__(self):
        lines = []
        for sfrom, arcs_by_sym in enumerate(self._arcs):
            for symbol, arcs in arcs_by_sym.iteritems():
                for sto, weight in sorted(arcs.iteritems(), key=lambda (s,w): s):
                    lines.append('(%d, %d, %s, %s)' % (sfrom, sto, symbol, weight))
        return '\n'.join(lines)


def make_linear_fsa(input_str, semiring, terminal_constructor=Terminal):
    wfsa = WDFSA()
    tokens = input_str.split()
    for i, token in enumerate(tokens):
        wfsa.add_arc(i, i + 1, terminal_constructor(token), semiring.one)
    wfsa.make_initial(0)
    wfsa.make_final(len(tokens))
    return wfsa

