"""
Forest rescoring based on the Earley parsing algorithm.

The forest can be rescored by arbitrarily complex scorers.
They manage their own states, thus states can be of very different nature.
This module assumes a state to be represented by a single integer (much like a state in an FSA), however,
using StatefulScorerWrapper one can encapsulate one or more scorers whose states are arbitrary hashable objects
and use this module to rescore a forest.

Note: a DFA (deterministic finite-state automaton) can be seen as a scorer and this module could be used
to compute the intersection. However, in this case, impossible transitions would have to be seen as
transitions with semiring.zero weight. The more general Earley intersection might remain a convenient option, as
it can handle an arbitrary FSA (including non-determinism).

:Authors: - Wilker Aziz
"""

import logging
from easyhg.grammar.symbol import Terminal, Nonterminal, make_span
from easyhg.grammar.cfg import CFG
from easyhg.grammar.rule import CFGProduction
from .dottedrule import DottedRule as Item
from .agenda import ActiveQueue, Agenda


EMPTY_SET = frozenset()


class EarleyRescoring(object):
    """
    """

    def __init__(self, forest,
                 scorer,
                 semiring):
        """
        :param forest:
        :param scorer: a stateful Scorer which produces integer states (consider using StatefulScorerWrapper).
        :param semiring:
        :return:
        """

        self._forest = forest
        self._scorer = scorer
        self._agenda = Agenda(active_container_type=ActiveQueue)
        self._predictions = set()  # (LHS, start)
        self._semiring = semiring

    def axioms(self, root, start):
        """
        Apply the axioms using all grammars
        :param root: forest's root node
        :param start: an initial state
        :return:
        """
        if (root, start) in self._predictions:  # already predicted
            return True
        self._predictions.add((root, start))

        rules = self._forest.get(root, set())
        if not rules:  # impossible to rewrite the symbol
            return False
        self._agenda.extend(Item(rule, start, weight=self._semiring.one) for rule in rules)
        return True

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

        rules = self._forest.get(item.next, set())
        if not rules:
            return False

        self._agenda.extend(Item(rule, item.dot, weight=self._semiring.one) for rule in rules)
        return True

    def scan(self, item):
        """
        This operation tries to scan over as many terminals as possible,
        but we only go as far as determinism allows.
        If we get to a nondeterminism, we stop scanning and add the relevant items to the agenda.
        """

        # retrieve the weight and the destination
        weight, to = self._scorer.score(item.next, context=item.dot)
        # scan the item
        self._agenda.add(item.advance(to, self._semiring.times(item.weight, weight)))
        return True

    def complete_others(self, item):
        """
        This operation creates new item by advancing the dot of passive items that are waiting for a certain given complete item.
        It returns whether or not at least one passive item awaited for the given complete item.
        """
        if self._agenda.is_generating(item.rule.lhs, item.start, item.dot):
            return True
        new_items = [incomplete.advance(item.dot, incomplete.weight) for incomplete in self._agenda.iterwaiting(item.rule.lhs, item.start)]
        self._agenda.extend(new_items)
        return len(new_items) > 0  # was there any item waiting for the complete one?

    def complete_itself(self, item):
        """
        This operation tries to merge a given incomplete item with a previosly completed one.
        """
        self._agenda.extend(item.advance(sto, item.weight) for sto in self._agenda.itercompletions(item.next, item.dot))
        return True

    def do(self, root=Nonterminal('S'), original_weights=None):
        """

        :param root: the root of the input forest
            The goal symbol, i.e. root of the output forest, is a Span(root, None, None).
        :param original_weights: a table to memorise a correspondence between the output edges and their input weights.
            This is necessary when exact rescoring is used within a slice sampling procedure.
        :return: the rescored forest
        """

        agenda = self._agenda

        initial = self._scorer.initial()
        final = self._scorer.final()
        self.axioms(root, initial)
        if not agenda:
            raise ValueError('I cannot rewrite the root: %s' % root)

        while agenda:
            item = agenda.pop()
            if item.is_complete():
                self.complete_others(item)
                agenda.make_complete(item)
                agenda.make_passive(item)
            else:
                if isinstance(item.next, Terminal):
                    self.scan(item)
                else:
                    if not self.prediction(item):  # try to predict, otherwise try to complete itself
                        self.complete_itself(item)
                    agenda.make_passive(item)

        logging.info('Making forest')
        rescored = self.make_cfg(root, initial, final, original_weights)
        logging.info('Done!')
        return rescored

    def make_cfg(self, root, initial, final, original_weights):
        """
        Constructs the CFG by visiting complete items in a top-down fashion.
        This is effectively a reachability test and it serves the purpose of filtering nonterminal symbols
        that could never be reached from the root.

        :param goal: goal symbol (to be decorated with initial and final after intersection).
        :param root: root of the input forest.
        :param initial: initial state.
        :param final: final state.
        :return: a CFG
        """

        semiring = self._semiring
        scorer = self._scorer
        G = CFG()
        processed = set()

        if original_weights is not None:
            def store_original(new_rule, old_weight):
                original_weights[new_rule] = old_weight
        else:
            store_original = lambda r, w: None

        def intersect_rule(rule, states, weight):
            """
            Create an intersected rule.
            :param rule: original rule
            :param states: sequence of intersected states (rule's arity plus one)
            :return: a CFGProduction
            """
            lhs = make_span(rule.lhs, states[0], states[-1])
            rhs = [None] * len(rule.rhs)
            for i, sym in enumerate(rule.rhs):
                rhs[i] = make_span(sym, states[i], states[i + 1])
            return CFGProduction(lhs, rhs, semiring.times(rule.weight, weight))

        def make_rules(lhs, start, end):
            """
            Make rules for edges outgoing from (lhs, start, end)
            :param lhs: underlying nonterminal
            :param start: origin state
            :param end: destination state
            """
            if (start, lhs, end) in processed:
                return
            processed.add((lhs, start, end))
            for item in self._agenda.itercomplete(lhs, start, end):
                states = item.inner + (item.dot,)
                new_rule = intersect_rule(item.rule, states, item.weight)
                G.add(new_rule)
                store_original(new_rule, item.rule.weight)
                # make rules for the intersected nonterminals
                for i, sym in filter(lambda i_s: isinstance(i_s[1], Nonterminal), enumerate(item.rule.rhs)):
                    if (sym, states[i], states[i + 1]) not in processed:
                        make_rules(sym, states[i], states[i + 1])

        # create the goal items

        for start, ends in self._agenda.itergenerating(root):
            if start != initial:
                continue
            for end in ends:
                make_rules(root, start, end)
                # the goal item incorporates p(bos) and p(eos|end)
                new_rule = CFGProduction(make_span(root, None, None),
                                    [make_span(root, start, end)],
                                    semiring.times(scorer.initial_score(), scorer.final_score(context=end)))
                G.add(new_rule)
                store_original(new_rule, semiring.one)

        # without make_rules above, one could run the following
        #for item in self._agenda.itercomplete():
        #    states = item.inner + (item.dot,)
        #    G.add(intersect_rule(item.rule, states, item.weight))

        return G