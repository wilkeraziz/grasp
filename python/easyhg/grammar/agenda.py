"""
This is an implementation of an agenda of items in agenda-based parsing/intersection.
This agenda is common to many algorithms including Earley's, CKY and Nederhof's.

We offer two types of containers for active items:
    1) ActiveSet behaves as a set
    2) ActiveQueue behaves as a queue, however, ensuring that an item never queues more than once

This module also provides a top-down algorithm that constructs products that are all reachable from the goal item.

@author wilkeraziz
"""

from collections import deque, defaultdict
from itertools import ifilter, chain
from symbol import Nonterminal
from rule import CFGProduction
from cfg import CFG


class ActiveSet(object):
    """
    Implement a set of active items
    """

    def __init__(self):
        self._active = set()
    
    def __len__(self):
        """Number of active items queuing to be processed"""
        return len(self._active)

    def pop(self):
        """Returns the next active item"""
        return self._active.pop()

    def add(self, item):
        """Add an active item if possible"""
        self._active.add(item)
        return True


class ActiveQueue(object):
    """
    Implement a queue of active items.
    However the queue guarantee that an item never queues more than once.
    """
    
    def __init__(self):
        self._active = deque()  # items to be processed
        self._queuing = set()  # items that are queuing (or have already left the queue)
    
    def __len__(self):
        """Number of active items queuing to be processed"""
        return len(self._active)

    def pop(self):
        """Returns the next active item"""
        return self._active.popleft()

    def add(self, item):
        """Add an active item if possible"""
        if item is not self._queuing:
            self._active.append(item)
            self._queuing.add(item)
            return True
        return False


class Agenda(object):
    """
    This is a CKY agenda which implements the algorithm by Nederhof and Satta (2008).
    It consists of:
        1) a queue of active items
        2) a set of generating intersected nonterminals
        3) a set of passive items
        4) a set of complete items
    """

    def __init__(self, active_container_type=ActiveQueue):
        self._active_container_type = active_container_type
        self._active = active_container_type()  # items to be processed
        self._passive = defaultdict(set)  # passive items waiting for completion: (LHS, start) -> items
        self._generating = defaultdict(lambda : defaultdict(set))  # generating symbols: LHS -> start -> ends
        self._complete = defaultdict(set)  # complete items

    def __len__(self):
        """Number of active items queuing to be processed"""
        return len(self._active)

    def pop(self):
        """Returns the next active item"""
        return self._active.pop()

    def add(self, item):
        """Add an active item if possible"""
        return self._active.add(item)

    def extend(self, items):
        for item in items:
            self.add(item)
    
    def is_passive(self, item):
        """Whether or not an item is passive"""
        return item in self._passive.get((item.next, item.dot), set())

    def add_generating(self, sym, sfrom, sto):
        """
        Tries to add a newly discovered generating symbol.
        Returns False if the symbol already exists, True otherwise.
        """
        destinations = self._generating[sym][sfrom]
        n = len(destinations)
        destinations.add(sto)
        return len(destinations) > n  

    def make_passive(self, item):
        """
        Tries to make passive an active item.
        Returns False if the item is already passive, True otherwise.
        """
        waiting = self._passive[(item.next, item.dot)]
        n = len(waiting)
        waiting.add(item)
        return len(waiting) > n

    def make_complete(self, item):
        """Stores a complete item"""
        self._complete[(item.rule.lhs, item.start, item.dot)].add(item)
        self.add_generating(item.rule.lhs, item.start, item.dot)

    def discard(self, item):
        waiting = self._passive.get((item.next, item.dot), None)
        if waiting:
            try:
                waiting.remove(item)
            except KeyError:
                pass
    
    def itergenerating(self, sym):
        """Returns an iterator to pairs of the kind (start, set of ends) for generating items based on a given symbol"""
        return self._generating.get(sym, {}).iteritems()
            
    def itercomplete(self, lhs=None, start=None, end=None):
        """
        Iterates through complete items whose left hand-side is (start, lhs, end)
        or through all of them if lhs is None
        """
        return chain(*self._complete.itervalues()) if lhs is None else iter(self._complete.get((lhs, start, end), set()))
            
    def iterwaiting(self, sym, start):
        """Returns items waiting for a certain symbol to complete from a certain state"""
        return iter(self._passive.get((sym, start), frozenset()))

    def itercompletions(self, sym, start):
        """Return possible completions of the given item"""
        return iter(self._generating.get(sym, {}).get(start, frozenset()))


def make_cfg(goal, root, itergenerating, itercomplete, fsa, semiring, make_symbol, recursive=False):
    """
    Constructs the CFG by visiting complete items in a top-down fashion.
    This is effectively a reachability test and it serves the purpose of filtering nonterminal symbols 
    that could never be reached from the root.
    Note that bottom-up intersection typically does enumerate a lot of useless (unreachable) items.
    """
    
    def recursive_reachability():
        """This is the recursive procedure described in the paper (Nederhof and Satta, 2008)."""
        G = CFG()
        processed = set()
        def make_rules(lhs, start, end):
            if (start, lhs, end) in processed:
                return
            processed.add((lhs, start, end))
            for item in itercomplete(lhs, start, end):
                G.add(item.cfg_production(fsa, semiring, make_symbol))
                fsa_states = item.inner + (item.dot,)
                for i, sym in ifilter(lambda (_, s): isinstance(s, Nonterminal), enumerate(item.rule.rhs)):
                    if (sym, fsa_states[i], fsa_states[i + 1]) not in processed:  # Nederhof does not perform this test, but in python it turned out crucial
                        make_rules(sym, fsa_states[i], fsa_states[i + 1])

        # create goal items
        for start, ends in itergenerating(root):
            if not fsa.is_initial(start):
                continue
            for end in ifilter(lambda q: fsa.is_final(q), ends):
                make_rules(root, start, end)
                G.add(CFGProduction(goal,
                    [make_symbol(root, start, end)],
                    semiring.one))
        return G

    def nonrecursive_reachability():
        """This version is non-recursive (it uses a deque)."""
        G = CFG()
        queuing = set()  # output symbols queuing (or that have already left the queue)
        Q = deque()  # queue of LHS annotated symbols whose rules are to be created
        # first we create rules for the roots
        for start, ends in itergenerating(root):
            if not fsa.is_initial(start):  # must span from an initial state
                continue
            for end in ifilter(lambda q: fsa.is_final(q), ends):  # to a final state
                Q.append((root, start, end)) 
                queuing.add((root, start, end)) 
                G.add(CFGProduction(goal,
                        [make_symbol(root, start, end)],
                        semiring.one))
        # create rules for symbols which are reachable from other generating symbols (starting from the root ones)
        while Q:
            (lhs, start, end) = Q.pop()
            for item in itercomplete(lhs, start, end):
                G.add(item.cfg_production(fsa, semiring, make_symbol))
                fsa_states = item.inner + (item.dot,)
                for i, sym in ifilter(lambda (_, s): isinstance(s, Nonterminal), enumerate(item.rule.rhs)):
                    if (sym, fsa_states[i], fsa_states[i + 1]) not in queuing:  # make sure the same symbol never queues more than once
                        Q.append((sym, fsa_states[i], fsa_states[i + 1]))
                        queuing.add((sym, fsa_states[i], fsa_states[i + 1]))
        return G
    
    return recursive_reachability() if recursive else nonrecursive_reachability()

