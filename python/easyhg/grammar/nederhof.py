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


@author wilkeraziz
"""

from collections import defaultdict, deque
from itertools import ifilter
from symbol import Nonterminal, make_flat_symbol
from dottedrule import DottedRule as Item
from rule import CFGProduction
from cfg import CFG
from agenda import ActiveQueue, Agenda, make_cfg


class Nederhof(object):
    """
    This is an implementation of the CKY-inspired intersection due to Nederhof and Satta (2008).
    """

    def __init__(self, wcfg, wfsa, semiring, scfg=None, make_symbol=make_flat_symbol):
        self._wcfg = wcfg
        self._wfsa = wfsa
        self._semiring = semiring
        self._scfg = scfg
        self._make_symbol = make_symbol
        self._agenda = Agenda(active_container_type=ActiveQueue)
        self._firstsym = defaultdict(set)  # index rules by their first RHS symbol
        
    def add_symbol(self, sym, sfrom, sto):
        """
        This operation:
            1) completes items waiting for `sym` from `sfrom`
            2) instantiate delayed axioms
        Returns False if the annotated symbol had already been added, True otherwise
        """
        if not self._agenda.add_generating(sym, sfrom, sto):  # stop if this is known to be a generating symbol
            return False

        # every item waiting for `sym` from `sfrom`
        for item in self._agenda.iterwaiting(sym, sfrom):
            self._agenda.add(item.advance(sto))

        # you may interpret this as a delayed axiom
        # every compatible rule in the grammar
        for r in self._firstsym.get(sym, set()):  
            self._agenda.add(Item(r, sto, inner=(sfrom,)))  # can be interpreted as a lazy axiom

        return True
    
    def add_item(self, item):
        """
        This operation:
            1) complete other items (by calling add_symbol), in case the input item is complete
            2) merges the input item with previously completed items effectively moving the input item's dot forward

        """
        if item.is_complete(): # complete others
            self.add_symbol(item.rule.lhs, item.start, item.dot)
            self._agenda.make_complete(item)
        else:  # complete itself
            if self._agenda.make_passive(item):  # if not already passive
                for sto in self._agenda.itercompletions(item.next, item.dot):
                    self._agenda.add(item.advance(sto))  # move the dot forward

    def axioms(self):
        """
        The axioms of the program are based on the FSA transitions. 
        """
        # you may interpret the following as a sort of lazy axiom (based on grammar rules)
        for r in self._wcfg.iterrules():
            self._firstsym[r.rhs[0]].add(r)
        # these are axioms based on the transitions of the automaton
        for sfrom, sto, sym, w in self._wfsa.iterarcs():
            self.add_symbol(sym, sfrom, sto)  
        # here we could deal with empty productions
        # for q in Q do  # every state in the wfsa
        #   for all (X -> epsilon) in R do
        #       A = A v {(q, A-> *, q)}

    def inference(self):
        """Exhausts the queue of active items"""
        while self._agenda:
            item = self._agenda.pop()
            self.add_item(item)

    def do(self, root=Nonterminal('S'), goal=Nonterminal('GOAL')):
        """Runs the program and returns the intersected CFG"""
        self.axioms()
        self.inference()
        return self._make_cfg_recursively(goal, root)

    def _make_cfg(self, goal, root):
        """
        Constructs the CFG by visiting the complete items in a top-down fashion.
        This is effectively a reachability test and it serves the purpose of filtering nonterminal symbols 
        that could never be reached from the root.
        Note that bottom-up intersection typically does enumerate a lot of useless (unreachable) items.
        This version is non-recursive (it uses a deque).
        """
        queuing = set()  # output symbols queuing (or that have already left the queue)
        Q = deque()  # queue of LHS annotated symbols whose rules are to be created
        G = CFG()  # intersection
        # first we create rules for the roots
        for start, ends in self._agenda.itergenerating(root):
            if not self._wfsa.is_initial(start):  # must span from an initial state
                continue
            for end in ifilter(lambda q: self._wfsa.is_final(q), ends):  # to a final state
                Q.append((root, start, end)) 
                queuing.add((root, start, end)) 
                G.add(CFGProduction(goal,
                        [self._make_symbol(root, start, end)],
                        self._semiring.one))
        # create rules for symbols which are reachable from other generating symbols (starting from the root ones)
        while Q:
            (lhs, start, end) = Q.pop()
            for item in self._agenda.itercomplete(lhs, start, end):
                G.add(item.cfg_production(self._wfsa, self._semiring, self._make_symbol))
                fsa_states = item.inner + (item.dot,)
                for i, sym in ifilter(lambda (_, s): isinstance(s, Nonterminal), enumerate(item.rule.rhs)):
                    if (sym, fsa_states[i], fsa_states[i + 1]) not in queuing:  # make sure the same symbol never queues more than once
                        Q.append((sym, fsa_states[i], fsa_states[i + 1]))
                        queuing.add((sym, fsa_states[i], fsa_states[i + 1]))
        return G
                        
    def _make_cfg_recursively(self, goal, root):
        """
        Constructs the CFG by visiting the complete items in a top-down fashion.
        This is effectively a reachability test and it serves the purpose of filtering nonterminal symbols 
        that could never be reached from the root.
        Note that bottom-up intersection typically does enumerate a lot of useless (unreachable) items.
        This is the recursive procedure described in the paper (Nederhof and Satta, 2008).
        """
        processed = set()
        G = CFG()
        def make_rules(lhs, start, end):
            if (start, lhs, end) in processed:
                return
            processed.add((lhs, start, end))
            for item in self._agenda.itercomplete(lhs, start, end):
                G.add(item.cfg_production(self._wfsa, self._semiring, self._make_symbol))
                fsa_states = item.inner + (item.dot,)
                for i, sym in ifilter(lambda (_, s): isinstance(s, Nonterminal), enumerate(item.rule.rhs)):
                    if (sym, fsa_states[i], fsa_states[i + 1]) not in processed:
                        make_rules(sym, fsa_states[i], fsa_states[i + 1])

        # create goal items
        for start, ends in self._agenda.itergenerating(root):
            if not self._wfsa.is_initial(start):
                continue
            for end in ifilter(lambda q: self._wfsa.is_final(q), ends):
                make_rules(root, start, end)
                G.add(CFGProduction(goal,
                    [self._make_symbol(root, start, end)],
                    self._semiring.one))
        return G


if __name__ == '__main__':
    import sys
    from fsa import make_linear_fsa
    from semiring import Prob
    from ply_cfg import read_grammar
    from cfg import CFG
    from topsort import topsort_cfg
    cfg = read_grammar(open('../../example/cfg', 'r'))

    for input_str in sys.stdin:
        fsa = make_linear_fsa(input_str, Prob)
        for word in fsa.itersymbols():
            if not cfg.is_terminal(word):
                p = CFGProduction(Nonterminal('X'), [word], Prob.one)
                cfg.add(p)
        parser = Nederhof(cfg, fsa, semiring=Prob)
        forest = parser.do()
        print forest
