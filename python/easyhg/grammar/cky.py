"""
This is an implementation of a CKY-inspired bottom-up intersection as presented in (Dyer, 2010):

    @phdthesis{Dyer:2010:phd,
        Address = {College Park, MD, USA},
        Author = {Dyer, Christopher James},
        Title = {A formal model of ambiguity and its applications in machine translation},
        Year = {2010}
    }
        
@author wilkeraziz
"""

from collections import defaultdict, deque
from symbol import Terminal, Nonterminal, make_flat_symbol, make_recursive_symbol
from topsort import topsort_cfg
from dottedrule import DottedRule as Item
from rule import CFGProduction
from agenda import ActiveSet, Agenda, make_cfg


class CKY(object):
    """
    This is the deductive logic program:

        Axioms:
            For every rule X -> alpha (with weight u) and every state q
            [X -> * alpha, q, q]: u 

        Goals:
            [S -> alpha *, q, r] where q is initial and r is final

        Scan:
            [X -> alpha * x beta, q, s]:u  where (s, x, r) is a transition
            _____________________________________________
            [X -> alpha x * beta, q, r]: u times w(s,x,r)

        Complete:
            [X -> alpha * Y beta, q, s]:u [Y -> gamma *, s, r]:v
            ____________________________________________________
                     [X -> alpha Y * beta, q, r]:u

    The program is executed in bottom-up order, that is, for each level of the grammar we exhaustively apply
    the axioms and inference rules above.
    """

    def __init__(self, wcfg, wfsa, semiring, scfg=None, make_symbol=make_flat_symbol):
        self._wcfg = wcfg
        self._wfsa = wfsa
        self._semiring = semiring
        self._scfg = scfg
        self._make_symbol = make_symbol
        self._agenda = Agenda(active_container_type=ActiveSet)
    
    def axioms(self, symbols, top_level=False):
        """
        For each symbol:
            * create items based on each and every rule whose LHS is the given symbol
            starting form each and every state in the fsa
        Special treatment for top-level (root) symbols:
            * only initial states are used
        """
        fsa_states = self._wfsa.iterstates() if not top_level else self._wfsa.iterinitial()
        for lhs in symbols:
            for rule in self._wcfg.get(lhs, set()):
                for q in fsa_states:
                    item = Item(rule, q)
                    self._agenda.add(item)
    
    def scan(self, item):
        """scans every transition in the fsa from item.dot whose symbol is item.next"""
        for sto, w in self._wfsa.get_arcs(item.dot, item.next):
            self._agenda.add(item.advance(sto))

    def complete_itself(self, item):
        """completes item.next (a nonterminal) from item.dot"""
        for sto in self._agenda.itercompletions(item.next, item.dot):
            self._agenda.add(item.advance(sto))

    def complete_others(self, complete):
        """advances items that have been waiting for a complete nonterminal"""
        for waiting in self._agenda.iterwaiting(complete.rule.lhs, complete.start):
            self._agenda.add(waiting.advance(complete.dot))

    def do(self, root=Nonterminal('S'), goal=Nonterminal('GOAL')):
        # group nonterminals by bottom-up level
        # and discard terminals
        nonterminals = list(topsort_cfg(self._wcfg))[1:]
        top_level = len(nonterminals) - 1
        agenda = self._agenda

        for level, symbols in enumerate(nonterminals):  # we go bottom-up level by level
            self.axioms(symbols, level == top_level)
            while agenda:  # running the program exhaustively until no active items remain
                item = agenda.pop()
                if item.is_complete():
                    # merge with passive
                    self.complete_others(item)
                    # store complete item
                    agenda.make_complete(item)
                elif isinstance(item.next, Terminal):
                    self.scan(item)
                else:
                    self.complete_itself(item)
                    agenda.make_passive(item)
       
        return make_cfg(goal, root, 
                self._agenda.itergenerating, self._agenda.itercomplete, 
                self._wfsa, self._semiring, self._make_symbol)


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
        parser = CKY(cfg, fsa, semiring=Prob)
        forest = parser.do()
        if not forest:
            print 'NO PARSE FOUND'
            continue
        print forest
