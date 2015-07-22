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

    def __init__(self, hg,
                 wfsa,
                 semiring,
                 glue_edges=set(),
                 make_symbol=make_flat_symbol):
        """

        :param grammars: one or more CFGs
        :param wfsa:
        :param semiring:
        :param glue_grammars: one or more glue CFGs (glue rules are only applied to initial states)
        :param make_symbol:
        :return:
        """
        self._hg = hg
        self._glue_edges = glue_edges
        self._wfsa = wfsa
        self._semiring = semiring
        self._make_symbol = make_symbol
        self._agenda = Agenda(active_container_type=ActiveQueue)
        self._firstsym = defaultdict(set)  # index rules by their first RHS symbol
        self._glue_firstsym = defaultdict(set)  # index glue rules by their first RHS symbol
        
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
        hg = self._hg

        node2label = lambda n: n.carry.get('symbol')

        # you may interpret the following as a sort of lazy axiom (based on grammar rules)
        for edge in filter(lambda e: e.id not in self._glue_edges, self._hg.iteredges()):
            self._firstsym[node2label(edge.tail[0])].add(edge.id)

        for e in self._glue_edges:
            self._glue_firstsym[node2label(hg.edge(e).tail[0])].add(e)

        # these are axioms based on the transitions of the automaton
        for sfrom, sto, sym, w in self._wfsa.iterarcs():
            self.add_symbol(sym, sfrom, sto)  
        # here we could deal with empty productions
        # for q in Q do  # every state in the wfsa
        #   for all (X -> epsilon) in R do
        #       A = A v {(q, A-> *, q)}

    def inference(self):
        """Exhausts the queue of active items"""
        agenda = self._agenda
        while agenda:
            item = agenda.pop()  # always returns an ACTIVE item
            # complete other items (by calling add_symbol), in case the input item is complete
            if item.is_complete():
                if agenda.make_complete(item):  # if we have discovered a new generating symbol
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

