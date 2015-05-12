"""
@author wilkeraziz
"""

import logging
import collections
import itertools
import numpy as np
from itertools import ifilter
from weakref import WeakValueDictionary
from collections import defaultdict
from symbol import Terminal, Nonterminal, make_symbol
from rule import project_rhs, CFGProduction, SCFGProduction


EMPTY_SET = frozenset()


class Item(object):

    _items = WeakValueDictionary()

    def __new__(cls, rule, dot, inner=[]):
        """The symbols in lhs and rhs must be hashable"""
        skeleton = (rule, dot, tuple(inner))
        obj = Item._items.get(skeleton, None)
        if not obj:
            obj = object.__new__(cls)
            Item._items[skeleton] = obj
            obj._skeleton = skeleton
            obj._start = inner[0] if inner else dot
            obj._next = rule.rhs[len(inner)] if len(inner) < len(rule.rhs) else None
            obj._active = True
        return obj
    
    @property
    def rule(self):
        return self._skeleton[0]

    @property
    def dot(self):
        return self._skeleton[1]

    @property
    def inner(self):
        return self._skeleton[2]

    @property
    def start(self):
        return self._start

    @property
    def next(self):
        return self._next

    def is_active(self):
        return self._active

    def make_passive(self):
        self._active = False

    def nextsymbols(self):
        return tuple(self.rule.rhs[len(self.inner):])

    def is_complete(self):
        return len(self.inner) == len(self.rule.rhs)

    def __str__(self):
        return '%s ||| %s ||| %d' % (str(self.rule), self.inner, self.dot)


class Agenda(object):

    def __init__(self):
        # a queue of active states
        #self._active = collections.OrderedDict()
        self._active = collections.deque()
        # these are the passive states
        # they are organized in sets
        # and distinguished between 'complete' and 'waiting for completion'
        self._waiting_completion = collections.defaultdict(set)
        self._complete = collections.defaultdict(set)

    def __str__(self):
        lines = ['passive (incomplete)']
        for key, items in self._waiting_completion.iteritems():
            lines.extend(str(item) for item in items)
        lines.append('passive (complete)')
        for key, items in self._complete.iteritems():
            lines.extend(str(item) for item in items)
        lines.append('active')
        for item in self._active.iterkeys():
            lines.append(str(item))
        return '\n'.join(lines)

    def clear(self):
        self._active.clear()
        self._waiting_completion.clear()
        self._complete.clear()

    def __len__(self):
        return len(self._active)

    def pop(self):
        return self._active.popleft()

    def is_passive(self, item):
        """Whether or not this state is passive (that is, is complete or waiting for completion)."""
        return not item.is_active()

    def make_passive(self, item):
        """Make a state passive storing it in the appropriate container."""
        if item.is_active():
            item.make_passive()
            if item.is_complete():
                self._complete[(item.start, item.rule.lhs)].add(item)
            else:
                self._waiting_completion[(item.dot, item.next)].add(item)

    def discard(self, item):
        pass

    def store_complete(self, item):
        """Stores a complete state, this is syntactic sugar for makePassive in case the input state is complete."""
        assert item.is_complete(), 'This state is not complete: %s' % item
        # TODO: [(item.start, item.rule.lhs)][item.dot].add(item)
        self._complete[(item.start, item.rule.lhs)].add(item)

    def itercomplete(self):
        """Iterates over the complete states in no particular order."""
        return itertools.chain(*self._complete.itervalues())

    def extend(self, items):
        """Adds states to the active queue if necessary and returns how many states were added."""
        before = len(self._active)
        # it is important not to add passive items to the active agenda
        # it would not break the intersection, but it would make it a lot less efficient
        self._active.extend(itertools.ifilter(lambda item: not self.is_passive(item), items))
        return len(self._active) - before

    def match_items_waiting_completion(self, complete):
        """Returns all the passive items that are waiting for the complete input item."""
        assert complete.is_complete(), 'This is not a complete state: %s' % complete
        return self._waiting_completion.get((complete.start, complete.rule.lhs), EMPTY_SET)

    def match_complete_items(self, incomplete):
        """Returns all the complete items that can make an incomplete item progress."""
        assert not incomplete.is_complete(), 'This is not an incomplete item: %s' % incomplete
        return self._complete.get((incomplete.dot, incomplete.next), EMPTY_SET)


class Earley(object):
    """
    Weights are interpreted as log-probabilities.
    """

    def __init__(self, wcfg, wfsa, semiring, scfg=None):
        """
        @param wcfg: is the set or rules
        @type RuleSet
        @param fsaId2Symbol: a dictionary that maps fsa ids in terminal symbols of the wcfg
        @param log: whether or not output log information
        """

        self._wcfg = wcfg
        self._wfsa = wfsa
        self._agenda = Agenda()
        self._predictions = set()
        self._scfg = scfg
        self._semiring = semiring

    def _initialize(self):
        pass

    def do(self, root=Nonterminal('S'), goal=Nonterminal('GOAL')):

        wfsa = self._wfsa
        wcfg = self._wcfg

        self._initialize()
        agenda = self._agenda

        # start items of the kind 
        # GOAL -> * ROOT, where * is an intial state of the wfsa
        if not any(self.axioms(root, start) for start in wfsa.initial_states):
            raise ValueError('No rule for the start symbol %s' % root)
        new_roots = set()
        
        it = 0
        while agenda:
            it += 1
            item = agenda.pop()

            # sometimes there are no active items left in the agenda
            if agenda.is_passive(item):
                continue

            if item.is_complete():
                # complete root item spanning from a start wfsa state to a final wfsa state
                if item.rule.lhs == root and item.start in wfsa.initial_states and item.dot in wfsa.final_states:
                    agenda.store_complete(item)
                    new_roots.add((root, item.start, item.dot))
                    agenda.make_passive(item)
                else:
                    if self.complete_others(item): 
                        agenda.store_complete(item)
                        agenda.make_passive(item)
                    else:  # a complete state is only kept in case it could potentially complete others
                        agenda.discard(item)
            else: 
                if isinstance(item.next, Terminal):
                    # fire the operation 'scan'
                    self.scan(item)
                    agenda.discard(item)  # scanning renders incomplete items of this kind useless
                else: 
                    if item.next not in wcfg:  # if the NT does not exist this item is useless
                        agenda.discard(item)
                    else:
                        if not self.prediction(item):  # try to predict, otherwise try to complete itself
                            self.complete_itself(item)
                        agenda.make_passive(item)

        # the intersected grammar
        if new_roots:
            # converts complete items into rules
            R = set(self.get_intersected_cfg_rules())
            for sym, si, sf in new_roots:
                R.add(CFGProduction(goal, [make_symbol(sym, si, sf)], 0.0))
            return True, R
        else:
            return False, frozenset()

    def get_item(self, rule, dot, inner=[]):
        return Item(rule, dot, inner)

    def axioms(self, symbol, start):
        rules = self._wcfg.get(symbol, None)
        if rules is None:  # impossible to rewrite the symbol
            return False

        if (start, symbol) in self._predictions:  # already predicted
            return True 

        # otherwise add rewritings to the agenda
        self._predictions.add((start, symbol))
        self._agenda.extend(self.get_item(rule, start) for rule in rules)
        return True

    def prediction(self, item):
        """
        This operation tris to create items from the rules associated with the nonterminal ahead of the dot.
        It returns True when prediction happens, and False if it already happened before.
        """
        if (item.dot, item.next) in self._predictions:  # prediction already happened
            return False
        # predict
        self._predictions.add((item.dot, item.next))
        new_items = [self.get_item(rule, item.dot) for rule in self._wcfg.get(item.next, frozenset())]
        self._agenda.extend(new_items) # how many were successfully made active
        return True


    def scan(self, item):
        """
        This operation tries to scan over as many terminals as possible, 
        but we only go as far as determinism allows. 
        If we get to a nondeterminism, we stop scanning and add the relevant items to the agenda.
        """
        states = [item.dot]
        weight = 0
        failed = False
        for sym in item.nextsymbols():
            if isinstance(sym, Terminal):
                #sto, w = self._wfsa.destination_and_weight(sfrom=states[-1], symbol=sym)
                arcs = self._wfsa.get_arcs(sfrom=states[-1], symbol=sym)
                if len(arcs) == 0:  # cannot scan the symbol
                    return False
                elif len(arcs) == 1:  # symbol is scanned deterministically
                    sto, w = arcs[0]
                    states.append(sto)  # we do not create intermediate items, instead we scan as much as we can
                    weight += w
                else:  # here we found a nondeterminism, we create all relevant items and add them to the agenda
                    # create items
                    for sto, w in arcs:
                        #rule = Rule(item.rule.lhs, item.rule.rhs, item.rule.weight + weight + w)
                        new = self.get_item(item.rule, sto, item.inner + tuple(states))
                        self._agenda.extend([new])
                    return True
            else:  # that's it, scan bumped into a nonterminal symbol, time to wrap up
                break
        # here we should have scanned at least one terminal symbol 
        # and we defined a deterministic path
        #rule = CFGProduction(item.rule.lhs, item.rule.rhs, item.rule.weight + weight)
        new = self.get_item(item.rule, states[-1], item.inner + tuple(states[:-1]))
        self._agenda.extend([new])
        return True


    def complete_others(self, item):
        """
        This operation creates new item by advancing the dot of passive items that are waiting for a certain given complete item.
        It returns whether or not at least one passive item awaited for the given complete item.
        """
        assert item.is_complete(), 'Complete (others) can only handle complete states.'
        incompletes = self._agenda.match_items_waiting_completion(item)
        self._agenda.extend(self.get_item(incomplete.rule, item.dot, incomplete.inner + (incomplete.dot,)) for incomplete in incompletes)
        return len(incompletes) > 0  # was there any item waiting for the complete one? 

    def complete_itself(self, item):
        """
        This operation tries to merge a given incomplete item with a previosly completed one.
        """
        destinations = frozenset(complete.dot for complete in self._agenda.match_complete_items(item))
        new_items = [self.get_item(item.rule, destination, item.inner + (item.dot,)) for destination in destinations]
        return len(destinations) > 0

    def get_intersected_cfg_rules(self):
        semiring = self._semiring
        for item in self._agenda.itercomplete():
            lhs = make_symbol(item.rule.lhs, item.start, item.dot)
            fsa_states = item.inner + (item.dot,)

            fsa_weights = []
            for i, sym in ifilter(lambda (_, s): isinstance(s, Terminal), enumerate(item.rule.rhs)):
                fsa_weights.append(self._wfsa.arc_weight(fsa_states[i], fsa_states[i + 1], sym))
            weight = reduce(semiring.times, fsa_weights, item.rule.weight)

            rhs = [make_symbol(sym, fsa_states[i], fsa_states[i + 1]) for i, sym in enumerate(item.rule.rhs)] 
            yield CFGProduction(lhs, rhs, weight)

    """
    def get_intersected_scfg_rules(self):
        for item in self._agenda.itercomplete():
            lhs = make_symbol(item.rule.lhs, item.start, item.dot)
            positions = item.inner + (item.dot,)
            f_rhs = [make_symbol(sym, positions[i], positions[i + 1]) for i, sym in enumerate(item.rule.rhs)]
            for syncr in self._projector.iterrulesbyf(item.rule.lhs, item.rule.rhs):
                yield SCFGProduction(lhs, f_rhs, syncr.e_rhs, syncr.alignment, times(item.rule.weight, reduce(plus, )))
                e_rhs = project_rhs(f_rhs, syncr.e_rhs, syncr.alignment)
                print '>', e_rhs
                yield CFGProduction(lhs, e_rhs, )
    """
