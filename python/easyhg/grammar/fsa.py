"""
@author wilkeraziz
"""

from collections import defaultdict
from symbol import Terminal

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
                self._arcs.append(defaultdict(list))
            return True
        return False

    def add_arc(self, sfrom, sto, symbol, weight=0.0):
        self._create_state(sfrom)  # create sfrom if necessary
        self._create_state(sto)  # create sto if necessary
        self._arcs[sfrom][symbol].append((sto, weight))
        self._vocabulary.add(symbol)

    def make_initial(self, state):
        self._initial_states.add(state)

    def make_final(self, state):
        self._final_states.add(state)

    def get_arcs(self, sfrom, symbol):
        if len(self._arcs) <= sfrom:
            raise ValueError('State %d does not exist' % sfrom)
        return self._arcs[sfrom].get(symbol, [])
        
    @property
    def initial_states(self):
        return self._initial_states
    
    @property
    def final_states(self):
        return self._final_states

    def __str__(self):
        lines = []
        for sfrom, arcs_by_sym in enumerate(self._arcs):
            for symbol, arcs in arcs_by_sym.iteritems():
                for sto, weight in arcs:
                    lines.append('(%d, %d, %s, %s)' % (sfrom, sto, symbol, weight))
        return '\n'.join(lines)


def make_linear_fsa(input_str, terminal_constructor=Terminal):
    wfsa = WDFSA()
    tokens = input_str.split()
    for i, token in enumerate(tokens):
        wfsa.add_arc(i, i + 1, terminal_constructor(token))
    wfsa.make_initial(0)
    wfsa.make_final(len(tokens))
    return wfsa

