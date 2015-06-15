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
    """

    def __init__(self,
                 grammars,
                 wfsa,
                 semiring,
                 glue_grammars,
                 slice_variables,
                 make_symbol=make_flat_symbol):

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
        
    def add_symbol(self, sym, sfrom, sto):
        """
        This operation:
            1) completes items waiting for `sym` from `sfrom`
            2) instantiate delayed axioms
        Returns False if the annotated symbol had already been added, True otherwise
        """

        # every item waiting for `sym` from `sfrom`
        for item in self._agenda.iterwaiting(sym, sfrom):
            self._agenda.add(item.advance(sto))

        # you may interpret this as a delayed axiom
        # every compatible rule in the grammar
        for r in self._firstsym.get(sym, set()):  
            self._agenda.add(Item(r, sto, inner=(sfrom,)))  # can be interpreted as a lazy axiom

        # again for glue rules, however, only if the origin state is initial in the FSA
        if self._wfsa.is_initial(sfrom):
            for r in self._glue_firstsym.get(sym, set()):
                self._agenda.add(Item(r, sto, inner=(sfrom,)))  # can be interpreted as a lazy axiom

        return True

    def axioms(self):
        """
        The axioms of the program are based on the FSA transitions. 
        """
        # you may interpret the following as a sort of lazy axiom (based on grammar rules)
        for grammar in self._grammars:
            for r in grammar:
                self._firstsym[r.rhs[0]].add(r)
        # again for glue grammars
        for grammar in self._glue:
            for r in grammar:
                self._glue_firstsym[r.rhs[0]].add(r)

        # these are axioms based on the transitions of the automaton
        for sfrom, sto, sym, w in self._wfsa.iterarcs():
            self.add_symbol(sym, sfrom, sto)  
        # here we could deal with empty productions
        # for q in Q do  # every state in the wfsa
        #   for all (X -> epsilon) in R do
        #       A = A v {(q, A-> *, q)}  # would need to check the slice variables

    def inference(self):
        """Exhausts the queue of active items"""
        agenda = self._agenda
        while agenda:
            item = agenda.pop()  # always returns an ACTIVE item
            # complete other items (by calling add_symbol), in case the input item is complete
            if item.is_complete():
                # slice check  (TODO: incorporate weights from intersected transitions before performing the check)
                if self._u.is_outside((item.rule.lhs, item.start, item.dot), self._semiring.as_real(item.rule.weight)):
                    self._agenda.discard(item)  # should I keep it (perhaps 'block' it somehow)?
                elif self._agenda.make_complete(item):  # if we have discovered a new generating symbol
                    self.add_symbol(item.rule.lhs, item.start, item.dot)
            else:
                # merges the input item with previously completed items effectively moving the input item's dot forward
                agenda.make_passive(item)
                for sto in agenda.itercompletions(item.next, item.dot):
                    agenda.add(item.advance(sto))  # move the dot forward

    def do(self, root=Nonterminal('S'), goal=Nonterminal('GOAL')):
        """Runs the program and returns the intersected CFG"""
        self.axioms()
        self.inference()
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