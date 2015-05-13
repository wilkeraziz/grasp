"""
@author wilkeraziz
"""
from collections import defaultdict, deque
from symbol import Terminal, Nonterminal, make_flat_symbol, make_recursive_symbol
from topsort import topsort_cfg
from earley import Item

class CKY(object):


    def __init__(self, wcfg, wfsa, semiring, scfg=None, make_symbol=make_flat_symbol):
        self._wcfg = wcfg
        self._wfsa = wfsa
        self._semiring = semiring
        self._scfg = scfg
        self._make_symbol = make_symbol
        self._sorted = list(topsort_cfg(self._wcfg))[1:]  # we do not care about terminals
        self._agenda = deque() 
        self._passive = defaultdict(set) 
        self._complete = defaultdict(lambda : defaultdict(set))

    def do(self, root=Nonterminal('S'), goal=Nonterminal('GOAL')):
        pass
        """
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
        """
        agenda = self._agenda
        self.axioms()
        while agenda:
            item = agenda.popleft()
            print 'POP', item
            
            if item.is_complete():
                print ' Complete'
                # merge with passive
                self._complete[(item.start, item.next)][item.dot].add(item)
                self.merge(item)
            elif isinstance(item.next, Terminal):
                self.scan(item)
            else:
                self.complete(item)
                self._passive[(item.dot, item.next)].add(item)
                #item.make_passive()

        print 'FOREST'
        for (sfrom, lhs), items_by_sto in self._complete.iteritems():
            for sto, items in items_by_sto.iteritems():
                for item in items:
                    print item


    def axioms(self):
        for level, group in enumerate(self._sorted):
            for lhs in group:
                for rule in self._wcfg.get(lhs, set()):
                    for q in self._wfsa.iterstates():
                        item = Item(rule, q)
                        self._agenda.append(item)
                        print '+', item

    def scan(self, item):
        for sto, w in self._wfsa.get_arcs(item.dot, item.next):
            new_item = Item(item.rule, dot=sto, inner=item.inner + (item.dot,))
            #if new_item.is_complete():
            #    completions[(new_item.start, new_item.rule.lhs)].add(new_item.dot)
            #    self._complete[(new_item.start, new_item.rule.lhs)][new_item.dot].add(new_item)
            self._agenda.appendleft(new_item)

    def complete(self, item):
        for sto in self._complete.get((item.dot, item.next), {}).iterkeys():
            new_item = Item(item.rule, dot=sto, inner=item.inner + (item.dot,))
            #if new_item.is_complete():
            #    completions[(new_item.start, new_item.rule.lhs)].add(new_item.dot)
            #    self._complete[(new_item.start, new_item.rule.lhs)][new_item.dot].add(new_item)
            self._agenda.appendleft(new_item)

    def merge(self, complete):
        for waiting in self._passive.get((complete.start, complete.rule.lhs), set()):
            self._agenda.appendleft(Item(waiting.rule, complete.dot, waiting.inner + (waiting.dot,)))


if __name__ == '__main__':
    import sys
    from fsa import make_linear_fsa
    from semiring import Prob
    from ply_cfg import read_grammar
    cfg = read_grammar(open('../../example/cfg', 'r'))
    print 'GRAMMAR'
    print cfg

    for input_str in sys.stdin:
        fsa = make_linear_fsa(input_str, Prob)
        print 'FSA'
        print fsa
        parser = CKY(cfg, fsa, semiring=Prob)
        parser.do()
