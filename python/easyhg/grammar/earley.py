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
from symbol import Terminal, Nonterminal, make_flat_symbol, make_recursive_symbol
from rule import CFGProduction, SCFGProduction
from dottedrule import DottedRule as Item
from agenda import ActiveQueue, Agenda, make_cfg
from cfg import CFG
from scfg import SCFG

EMPTY_SET = frozenset()


class Earley(object):
    """
    """

    def __init__(self, wcfg, wfsa, semiring, scfg=None, make_symbol=make_flat_symbol):
        """
        @param wcfg: is the set or rules
        @type RuleSet
        @param fsaId2Symbol: a dictionary that maps fsa ids in terminal symbols of the wcfg
        @param log: whether or not output log information
        """

        self._wcfg = wcfg
        self._wfsa = wfsa
        self._agenda = Agenda(active_container_type=ActiveQueue)
        self._predictions = set()  # (LHS, start)
        self._scfg = scfg
        self._semiring = semiring
        self._make_symbol = make_symbol
    
    def axioms(self, symbol, start):
        rules = self._wcfg.get(symbol, None)
        if rules is None:  # impossible to rewrite the symbol
            return False
        if (symbol, start) in self._predictions:  # already predicted
            return True 
        # otherwise add rewritings to the agenda
        self._predictions.add((symbol, start))
        self._agenda.extend(Item(rule, start) for rule in rules)
        return True
    
    def prediction(self, item):
        """
        This operation tris to create items from the rules associated with the nonterminal ahead of the dot.
        It returns True when prediction happens, and False if it already happened before.
        """
        if (item.next, item.dot) in self._predictions:  # prediction already happened
            return False
        self._predictions.add((item.next, item.dot))
        new_items = [Item(rule, item.dot) for rule in self._wcfg.get(item.next, frozenset())]
        self._agenda.extend(new_items)
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
                        self._agenda.add(Item(item.rule, sto, item.inner + tuple(states)))
                    return True
            else:  # that's it, scan bumped into a nonterminal symbol, time to wrap up
                break
        # here we should have scanned at least one terminal symbol 
        # and we defined a deterministic path
        self._agenda.add(Item(item.rule, states[-1], item.inner + tuple(states[:-1])))
        return True

    def complete_others(self, item):
        """
        This operation creates new item by advancing the dot of passive items that are waiting for a certain given complete item.
        It returns whether or not at least one passive item awaited for the given complete item.
        """
        new_items = [incomplete.advance(item.dot) for incomplete in self._agenda.iterwaiting(item.rule.lhs, item.start)]
        self._agenda.extend(new_items)
        return len(new_items) > 0  # was there any item waiting for the complete one? 

    def complete_itself(self, item):
        """
        This operation tries to merge a given incomplete item with a previosly completed one.
        """
        new_items = [item.advance(sto) for sto in self._agenda.itercompletions(item.next, item.dot)]
        self._agenda.extend(new_items)
        return len(new_items) > 0

    def do(self, root=Nonterminal('S'), goal=Nonterminal('GOAL')):

        wfsa = self._wfsa
        wcfg = self._wcfg
        agenda = self._agenda

        # start items of the kind 
        # GOAL -> * ROOT, where * is an intial state of the wfsa
        if not any(self.axioms(root, start) for start in wfsa.initial_states):
            raise ValueError('No rule for the start symbol %s' % root)
        new_roots = set()
        
        while agenda:
            item = agenda.pop()

            # sometimes there are no active items left in the agenda
            if agenda.is_passive(item):
                continue

            if item.is_complete():
                # complete root item spanning from a start wfsa state to a final wfsa state
                if item.rule.lhs == root and item.start in wfsa.initial_states and item.dot in wfsa.final_states:
                    agenda.make_complete(item)
                    new_roots.add((root, item.start, item.dot))
                    agenda.make_passive(item)
                else:
                    if self.complete_others(item): 
                        agenda.make_complete(item)
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

        return make_cfg(goal, root, 
                self._agenda.itergenerating, self._agenda.itercomplete, 
                self._wfsa, self._semiring, self._make_symbol)

        # converts complete items into rules
        if self._scfg is None:
            if new_roots:
                return self.get_intersected_cfg(new_roots, goal)
            else:
                return CFG()
        else:
            if new_roots:
                return self.get_intersected_scfg(new_roots, goal)
            else:
                return SCFG()

    def get_intersected_cfg(self, new_roots, goal):
        semiring = self._semiring
        make_symbol = self._make_symbol
        G = CFG()
        for item in self._agenda.itercomplete():
            lhs = make_symbol(item.rule.lhs, item.start, item.dot)
            fsa_states = item.inner + (item.dot,)
            fsa_weights = []
            for i, sym in ifilter(lambda (_, s): isinstance(s, Terminal), enumerate(item.rule.rhs)):
                fsa_weights.append(self._wfsa.arc_weight(fsa_states[i], fsa_states[i + 1], sym))
            weight = reduce(semiring.times, fsa_weights, item.rule.weight)

            rhs = [make_symbol(sym, fsa_states[i], fsa_states[i + 1]) for i, sym in enumerate(item.rule.rhs)] 
            G.add(CFGProduction(lhs, rhs, weight))
        for sym, si, sf in new_roots:
            G.add(CFGProduction(goal, [make_symbol(sym, si, sf)], semiring.one))

    def get_intersected_scfg(self, new_roots, goal):
        semiring = self._semiring
        make_symbol = self._make_symbol
        G = SCFG()
        for item in self._agenda.itercomplete():
            lhs = make_symbol(item.rule.lhs, item.start, item.dot)
            fsa_states = item.inner + (item.dot,)
            fsa_weights = []
            for i, sym in ifilter(lambda (_, s): isinstance(s, Terminal), enumerate(item.rule.rhs)):
                fsa_weights.append(self._wfsa.arc_weight(fsa_states[i], fsa_states[i + 1], sym))
            weight = reduce(semiring.times, fsa_weights, item.rule.weight)
            f_rhs = [make_symbol(sym, fsa_states[i], fsa_states[i + 1]) for i, sym in enumerate(item.rule.rhs)]  
            for syncr in self._scfg.iterrulesbyf(item.rule.lhs, item.rule.rhs):
                G.add(SCFGProduction(lhs, f_rhs, syncr.e_rhs, syncr.alignment, semiring.times(weight, syncr.weight)))
        for sym, si, sf in new_roots:
            G.add(SCFGProduction(goal, [make_symbol(sym, si, sf)], [Nonterminal('1')], [1], semiring.one))
