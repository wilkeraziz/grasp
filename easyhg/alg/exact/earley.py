"""
This is an implementation of Earley intersection as presented in (Dyer and Resnik, 2010).

:Authors: - Wilker Aziz
"""

from itertools import chain
from easyhg.grammar.symbol import Terminal, Nonterminal
from easyhg.grammar.grammar import Grammar
from .dottedrule import DottedRule as Item
from .agenda import ActiveQueue, Agenda, make_cfg


EMPTY_SET = frozenset()


class Earley(object):
    """
    """

    def __init__(self, grammars,
                 wfsa,
                 semiring,
                 glue_grammars=[]):
        """

        :param grammars: one or more CFGs
        :param wfsa:
        :param semiring:
        :param glue grammars: one or more glue CFGs (glue rules are only applied to initial states)
        :return:
        """

        if isinstance(grammars, Grammar):
            self._grammars = [grammars]
        else:
            self._grammars = list(grammars)

        if isinstance(glue_grammars, Grammar):
            self._glue = [glue_grammars]
        else:
            self._glue = list(glue_grammars)

        self._wfsa = wfsa
        self._agenda = Agenda(active_container_type=ActiveQueue)
        self._predictions = set()  # (LHS, start)
        self._semiring = semiring

    def itergrammar(self, origin):
        """Returns an iterator over grammars which depends on the origin state.
        If the origin is an initial FSA state, then we include glue grammars."""

        if self._wfsa.is_initial(origin):
            return chain(self._grammars, self._glue)
        else:
            return iter(self._grammars)

    def can_rewrite(self, symbol, origin):
        """
        Whether or not any of the grammars can rewrite the a certain nonterminal symbol.
        For glue grammars, it matters whether the origin state is initial.
        :param symbol: a Nonterminal
        :param origin: a state
        """
        return any(grammar.can_rewrite(symbol) for grammar in self.itergrammar(origin))

    def axioms(self, symbol, start):
        """
        Apply the axioms using all grammars
        :param symbol:
        :param start: an initial FSA symbol
        :return:
        """
        if (symbol, start) in self._predictions:  # already predicted
            return True
        status = False
        for grammar in self.itergrammar(start):
            rules = grammar.get(symbol, set())
            if not rules:  # impossible to rewrite the symbol
                continue
            self._agenda.extend(Item(rule, start) for rule in rules)
            status = True
        # otherwise add rewritings to the agenda
        self._predictions.add((symbol, start))
        return status

    def prediction(self, item):
        """
        Prediction using all grammars.
        :param item:
        :return:
        """
        key = (item.next, item.dot)
        if key in self._predictions:  # prediction already happened
            return False
        self._predictions.add(key)

        for grammar in self.itergrammar(item.dot):
            self._agenda.extend(Item(rule, item.dot) for rule in grammar.get(item.next, frozenset()))
        return True

    def scan(self, item):
        """
        This operation tries to scan over as many terminals as possible, 
        but we only go as far as determinism allows. 
        If we get to a nondeterminism, we stop scanning and add the relevant items to the agenda.
        """
        states = [item.dot]
        for sym in item.nextsymbols():
            if isinstance(sym, Terminal):
                arcs = self._wfsa.get_arcs(origin=states[-1], symbol=sym)
                if len(arcs) == 0:  # cannot scan the symbol
                    return False
                elif len(arcs) == 1:  # symbol is scanned deterministically
                    sto, _ = arcs[0]
                    states.append(sto)  # we do not create intermediate items, instead we scan as much as we can
                else:  # here we found a nondeterminism, we create all relevant items and add them to the agenda
                    # create items
                    for sto, _ in arcs:
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
        if self._agenda.is_generating(item.rule.lhs, item.start, item.dot):
            return True
        new_items = [incomplete.advance(item.dot) for incomplete in self._agenda.iterwaiting(item.rule.lhs, item.start)]
        self._agenda.extend(new_items)
        return len(new_items) > 0  # was there any item waiting for the complete one?

    def complete_itself(self, item):
        """
        This operation tries to merge a given incomplete item with a previosly completed one.
        """
        self._agenda.extend(item.advance(sto) for sto in self._agenda.itercompletions(item.next, item.dot))
        return True

    def do(self, root=Nonterminal('S'), goal=Nonterminal('GOAL')):

        wfsa = self._wfsa
        agenda = self._agenda

        # start items of the kind 
        # GOAL -> * ROOT, where * is an intial state of the wfsa
        for start in wfsa.iterinitial():
            self.axioms(root, start)
        if not agenda:
            raise ValueError('No rule for the start symbol %s' % root)

        while agenda:
            item = agenda.pop()
            # sometimes there are no active items left in the agenda
            #assert not agenda.is_passive(item), 'This is strange!'

            discard = False

            if item.is_complete():
                # complete root item spanning from a start wfsa state to a final wfsa state
                if item.rule.lhs == root and wfsa.is_initial(item.start) and wfsa.is_final(item.dot):
                    agenda.make_complete(item)
                else:
                    if self.complete_others(item):
                        agenda.make_complete(item)
                    else:  # a complete state is only kept in case it could potentially complete others
                        discard = True
            else:
                if isinstance(item.next, Terminal):
                    # fire the operation 'scan'
                    self.scan(item)
                    discard = True  # scanning renders incomplete items of this kind useless
                else:
                    if not self.can_rewrite(item.next, item.dot):  # if the NT does not exist this item is useless
                        discard = True
                    else:
                        if not self.prediction(item):  # try to predict, otherwise try to complete itself
                            self.complete_itself(item)

            if discard:
                agenda.discard(item)
            else:
                agenda.make_passive(item)

        return make_cfg(goal, root, self._agenda.itergenerating, self._agenda.itercomplete, self._wfsa, self._semiring)