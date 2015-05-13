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
        self._active = deque() 
        self._waiting = defaultdict(set) 
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
        active = self._active
        for symbols in self._sorted:
            self.axioms(symbols)
            while active:
                item = active.popleft()
                print 'POP', item
                
                if item.is_complete():
                    print ' Complete', item
                    # merge with passive
                    self.merge(item)
                    self._complete[(item.start, item.rule.lhs)][item.dot].add(item)
                elif isinstance(item.next, Terminal):
                    self.scan(item)
                else:
                    self.complete(item)
                    self._waiting[(item.dot, item.next)].add(item)
                    #item.make_passive()

        print 'FOREST'
        for (sfrom, lhs), items_by_sto in self._complete.iteritems():
            for sto, items in items_by_sto.iteritems():
                for item in items:
                    print item


    def axioms(self, symbols):
        for lhs in symbols:
            for rule in self._wcfg.get(lhs, set()):
                for q in self._wfsa.iterstates():
                    item = Item(rule, q)
                    self._active.append(item)
                    print 'A', item

    def scan(self, item):
        for sto, w in self._wfsa.get_arcs(item.dot, item.next):
            new_item = Item(item.rule, dot=sto, inner=item.inner + (item.dot,))
            #if new_item.is_complete():
            #    completions[(new_item.start, new_item.rule.lhs)].add(new_item.dot)
            #    self._complete[(new_item.start, new_item.rule.lhs)][new_item.dot].add(new_item)
            print ' S', new_item
            self._active.append(new_item)

    def complete(self, item):
        for sto in self._complete.get((item.dot, item.next), {}).iterkeys():
            new_item = Item(item.rule, dot=sto, inner=item.inner + (item.dot,))
            #if new_item.is_complete():
            #    completions[(new_item.start, new_item.rule.lhs)].add(new_item.dot)
            #    self._complete[(new_item.start, new_item.rule.lhs)][new_item.dot].add(new_item)
            print ' C', new_item
            self._active.append(new_item)

    def merge(self, complete):
        for waiting in self._waiting.get((complete.start, complete.rule.lhs), set()):
            new_item = Item(waiting.rule, complete.dot, waiting.inner + (waiting.dot,))
            self._active.append(new_item)
            print ' M', new_item


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
