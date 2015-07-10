"""
This is an implementation of the bottom-up intersection by Nederhof and Satta (2008) described in the paper:

    @inproceedings{Nederhof+2008:probparsing,
        Author = {Mark-Jan Nederhof and Giorgio Satta},
        Booktitle = {New Developments in Formal Languages and Applications, Studies in Computational Intelligence},
        Editor = {G. Bel-Enguix, M. Dolores Jim{\\'e}nez-L{\\'o}pez, and C. Mart{\\'\\i}n-Vide},
        Pages = {229-258},
        Publisher = {Springer},
        Title = {Probabilistic Parsing},
        Volume = {113},
        Year = {2008}
    }

:Authors: - Wilker Aziz
"""

from collections import defaultdict
from .symbol import Nonterminal, make_flat_symbol
from .dottedrule import DottedRule as Item
from .agenda import ActiveQueue, Agenda, make_cfg
from .grammar import Grammar


class Nederhof(object):
    """
    This is an implementation of the CKY-inspired intersection due to Nederhof and Satta (2008).
    It handles arbitrary epsilon-free FSAs.

    Optionally, pruning (or *slicing*) can be controlled by "slice variables" as in Blunsom and Cohn (2010).

    """

    def __init__(self,
                 grammars,
                 wfsa,
                 semiring,
                 glue_grammars,
                 make_symbol=make_flat_symbol,
                 slice_variables=None):
        """
        :param grammars: a list of wCFGs
        :param wfsa: a wFSA
        :param semiring: a Semiring
        :param glue_grammars: list of glue CFGs (whose LHS symbols rewrite only from initial FSA states)
        :param make_symbol: how to compose intersected symbols
        :param slice_variables: control pruning if specified
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
        self._semiring = semiring
        self._make_symbol = make_symbol
        self._agenda = Agenda(active_container_type=ActiveQueue)
        self._firstsym = defaultdict(set)  # index rules by their first RHS symbol
        self._glue_firstsym = defaultdict(set)  # index glue rules by their first RHS symbol
        self._u = slice_variables

        # whether or not we are slicing
        if self._u is None:
            self._process_complete = self._exact_process_complete
        else:
            self._process_complete = self._sliced_process_complete

        self._index_rules()

    def _index_rules(self):
        """Indexes rules by their first RHS symbol to ease some inference rules."""

        # normal grammars
        for grammar in self._grammars:
            for r in grammar:
                self._firstsym[r.rhs[0]].add(r)
        # glue grammars
        for grammar in self._glue:
            for r in grammar:
                self._glue_firstsym[r.rhs[0]].add(r)

    def _axioms(self):
        """
        The axioms are based on the transitions (E) of the FSA.
        Every rule whose RHS starts with a terminal matching a transition in E gives rise to an item.

        1) instantiate FSA transitions

            <i, z, j> in E

        2) populate the agenda by calling the operation "Delayed axioms"

                 <i, z, j>
            -------------------------   (X -> y alpha) in R
            [X -> y * alpha, [i, j]]
        """

        for origin, destination, terminal, weight in self._wfsa.iterarcs():
            self._delayed_axioms(terminal, origin, destination)
            self._agenda.add_generating(terminal, origin, destination)

    def _delayed_axioms(self, sym, sfrom, sto):
        """
        A delayed axiom (combination between a prediction and a completion).

                 <i, delta, j>
            -----------------------------    (X -> delta alpha) in R and delta in (N v Sigma)
            [X -> delta * alpha, [i, j]]

        :param sym: a symbol (terminal or nonterminal)
        :param sfrom: origin state
        :param sto: destination state
        """

        # regulars grammars
        for r in self._firstsym.get(sym, set()):
            self._agenda.add(Item(r, sto, inner=(sfrom,)))  # can be interpreted as a lazy axiom

        # glue grammars: origin state must be initial in the FSA
        if self._wfsa.is_initial(sfrom):
            for r in self._glue_firstsym.get(sym, set()):
                self._agenda.add(Item(r, sto, inner=(sfrom,)))  # can be interpreted as a lazy axiom

    def _complete_others(self, sym, sfrom, sto):
        """
        This operation complete items waiting for `sym` from `sfrom`.

        Complete:

            [X -> alpha * delta beta, [i ... j]] <j, delta, k>
            ---------------------------------------------------
               [X -> alpha delta * beta, [i ... j, k]]
        """

        for item in self._agenda.iterwaiting(sym, sfrom):
            self._agenda.add(item.advance(sto))
        return True

    def _complete_itself(self, item):
        """
        This operation tries to extend the given item by advancing the dot over generating symbols.

            [X -> alpha * delta beta, [i ... j]] <j, delta, k>
            ---------------------------------------------------
               [X -> alpha delta * beta, [i ... j, k]]

        :param item:
        :return: whether the item should remain in the program (always True).
        """

        for sto in self._agenda.itercompletions(item.next, item.dot):
            self._agenda.add(item.advance(sto))  # move the dot forward
        return True

    def _exact_process_complete(self, item):
        """
        Complete other items by calling:
            1. complete others
            2. delayed axioms

        :param item: a complete item.
        :return: True (which means the item should remain in the program).
        """
        (lhs, start, end) = item.rule.lhs, item.start, item.dot
        if not self._agenda.is_generating(lhs, start, end):  # if not yet discovered
            self._complete_others(lhs, start, end)  # complete other items
            self._delayed_axioms(lhs, start, end)   # instantiate new items from matching rules
        return True

    def _sliced_process_complete(self, item):
        """
        Execute `_exact_complete_others` if the item belongs to the slice.

        :param item: a complete item
        :return: whether an item should remain in the program or be discarded
        """
        if self._u.is_outside((item.rule.lhs, item.start, item.dot), self._semiring.as_real(item.rule.weight)):  # TODO: incorporate weight of the item
            return False
        else:
            return self._exact_process_complete(item)

    def _inference(self):
        """Exhausts the queue of active items"""
        agenda = self._agenda
        while agenda:
            item = agenda.pop()
            keep = self._process_complete(item) if item.is_complete() else self._complete_itself(item)
            if keep:
                agenda.make_passive(item)
            else:
                self._agenda.discard(item)

    def do(self, root=Nonterminal('S'), goal=Nonterminal('GOAL')):
        """Runs the program and returns the intersected CFG"""
        self._axioms()
        self._inference()
        return make_cfg(goal, root, 
                self._agenda.itergenerating, self._agenda.itercomplete, 
                self._wfsa, self._semiring, self._make_symbol)

    def reweight(self, forest):
        if self._semiring.LOG:
            return defaultdict(None,
                ((rule, self._u.logpr(rule.lhs.label, self._semiring.as_real(rule.weight))) for rule in forest))
        else:
            return defaultdict(None,
                ((rule, self._semiring.from_real(self._u.pr(rule.lhs.label, self._semiring.as_real(rule.weight)))) for rule in forest))