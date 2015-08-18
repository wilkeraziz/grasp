"""
This module contains class definitions for wFSAs.

:Authors: - Wilker Aziz
"""

from collections import defaultdict
from .symbol import Terminal


class WDFSA(object):
    """
    This is a deterministic wFSA.
    
    TODO: extend it to handle nondeterminism.
    """

    def __init__(self):
        self._arcs = []  # sfrom -> sym -> sto -> weight
        self._initial_states = set()
        self._final_states = set()
        self._vocabulary = set()

    def n_states(self):
        """Number of states."""
        return len(self._arcs)  # perhaps _arcs is an unfortunate, _arcs is indexed by states, thus len(_arcs) == n_states

    def n_arcs(self):
        """Count number of arcs."""
        return sum(1 for a in self.iterarcs())

    def n_symbols(self):
        """Number of different symbols."""
        return len(self._vocabulary)

    def _create_state(self, state):
        """This is meant for private use, it allocates memory for one or more states.
        
        :returns:
            whether or not any state was added."""

        if len(self._arcs) <= state:
            for i in range(len(self._arcs), state + 1):
                self._arcs.append(defaultdict(lambda: defaultdict(float)))
            return True
        return False

    def iterstates(self):
        """Iterate through all states in order of allocation."""
        return range(len(self._arcs))

    def iterinitial(self):
        """Iterate through all initial states in no particular order."""
        return iter(self._initial_states)
    
    def iterfinal(self):
        """Iterate through all final states in no particular order."""
        return iter(self._final_states)
    
    def itersymbols(self):
        """Iterate through all symbols labelling transitions."""
        return iter(self._vocabulary)

    def iterarcs(self):
        """Iterate through all arcs/transitions in no particular order.
        
        arc:
            a tuple of the kind (origin, destination, label, weight).
        """

        for sfrom, arcs_by_sym in enumerate(self._arcs):
            for sym, w_by_sto in arcs_by_sym.items():
                for sto, w in w_by_sto.items():
                    yield (sfrom, sto, sym, w)
    
    def get_arcs(self, origin, symbol):
        """Return a list of pairs representing a destination and a weight.
        
        :param origin: origin state.
        :param symbol: label.
        """
        if len(self._arcs) <= origin:
            raise ValueError('Origin state %d does not exist' % origin)
        return list(self._arcs[origin].get(symbol, {}).items())
        
    def is_initial(self, state):
        """Whether or not a state is initial."""
        return state in self._initial_states

    def is_final(self, state):
        """Whether or not a state is final."""
        return state in self._final_states

    def add_arc(self, sfrom, sto, symbol, weight):
        """Add an arc creating the necessary states."""
        self._create_state(sfrom)  # create sfrom if necessary
        self._create_state(sto)  # create sto if necessary
        self._arcs[sfrom][symbol][sto] = weight
        self._vocabulary.add(symbol)

    def make_initial(self, state):
        """Make a state initial."""
        self._initial_states.add(state)

    def make_final(self, state):
        """Make a state final."""
        self._final_states.add(state)

    def path_weight(self, path, semiring): 
        """Returns the weight of a path given by a sequence of tuples of the kind (origin, destination, sym)"""
        total = semiring.one
        for (origin, destination, sym) in path:
            arcs = self._arcs[origin].get(sym, None)
            if arcs is None:
                raise ValueError('Invalid transition origin=%s sym=%s' % (origin, sym))
            w = arcs.get(destination, None)
            if w is None:
                raise ValueError('Invalid transition origin=%s destination=%s sym=%s' % (origin, destination, sym))
            total = semiring.times(total, w)
        return total
    
    def arc_weight(self, origin, destination, sym):
        """Returns the weight of an arc."""
        if not (0 <= origin < len(self._arcs)):
            raise ValueError('Unknown state origin=%s' % (origin))
        arcs = self._arcs[origin].get(sym, None)
        if arcs is None:
            raise ValueError('Invalid transition origin=%s sym=%s' % (origin, sym))
        w = arcs.get(destination, None)
        if w is None:
            raise ValueError('Invalid transition origin=%s destination=%s sym=%s' % (origin, destination, sym))
        return w
        
    def __str__(self):
        lines = []
        for origin, arcs_by_sym in enumerate(self._arcs):
            for symbol, arcs in arcs_by_sym.items():
                for destination, weight in sorted(arcs.items()):
                    lines.append('(%r, %r, %r, %r)' % (origin, destination, symbol, weight))
        return '\n'.join(lines)


def make_linear_fsa(sentence, semiring, terminal_constructor=Terminal):
    """
    Return a linea (unweighted) FSA representing an input sentence.

    :param sentence:
        a string or a sequence of tokens
    :param semiring:
        how we perform operations.
    :param terminal_constructor:
        which class we use to construct terminal symbols.
    :returns:
        WDFSA

    >>> from grasp.semiring import Prob
    >>> fsa = make_linear_fsa('a dog barks', Prob)
    >>> fsa.n_states()
    4
    >>> fsa.n_arcs()
    3
    >>> fsa.n_symbols()
    3
    >>> list(fsa.iterinitial()) == [0]
    True
    >>> list(fsa.iterfinal()) == [3]
    True
    """

    wfsa = WDFSA()
    if type(sentence) is str:
        tokens = sentence.split()
    else:
        tokens = list(sentence)
    for i, token in enumerate(tokens):
        wfsa.add_arc(i, i + 1, terminal_constructor(token), semiring.one)
    wfsa.make_initial(0)
    wfsa.make_final(len(tokens))
    return wfsa

