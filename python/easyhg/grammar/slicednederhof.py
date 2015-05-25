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

    def __init__(self, wcfg, wfsa, semiring, slice_variables, scfg=None, make_symbol=make_flat_symbol):
        self._wcfg = wcfg
        self._wfsa = wfsa
        self._semiring = semiring
        self._scfg = scfg
        self._make_symbol = make_symbol
        self._agenda = Agenda(active_container_type=ActiveQueue)
        self._firstsym = defaultdict(set)  # index rules by their first RHS symbol        
        self._u = slice_variables
        
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
            # slice check  (TODO: incorporate weights from intersected transitions before performing the check)
            u_s = self._u[(item.rule.lhs, item.start, item.dot)]
            if self._semiring.as_real(item.rule.weight) < u_s:  # for now this check ignores the intersection and relies on the parameters of the CFG alone
                self._agenda.discard(item)  # should I keep it (perhaps 'block' it somehow)?
                #print >> sys.stderr, 'below threshold (%s): %s' % (u_s, item)
            else:
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
        for r in self._wcfg:
            self._firstsym[r.rhs[0]].add(r)
        # these are axioms based on the transitions of the automaton
        for sfrom, sto, sym, w in self._wfsa.iterarcs():
            self.add_symbol(sym, sfrom, sto)  
        # here we could deal with empty productions
        # for q in Q do  # every state in the wfsa
        #   for all (X -> epsilon) in R do
        #       A = A v {(q, A-> *, q)}  # would need to check the slice variables

    def inference(self):
        """Exhausts the queue of active items"""
        while self._agenda:
            item = self._agenda.pop()
            self.add_item(item)

    def do(self, root=Nonterminal('S'), goal=Nonterminal('GOAL')):
        """Runs the program and returns the intersected CFG"""
        self.axioms()
        self.inference()
        return make_cfg(goal, root, 
                self._agenda.itergenerating, self._agenda.itercomplete, 
                self._wfsa, self._semiring, self._make_symbol)

    def reweight(self, forest):
        return defaultdict(None, 
                ((rule, self._semiring.from_real(self._u.pr(rule.lhs.label, self._semiring.as_real(rule.weight)))) for rule in forest))


if __name__ == '__main__':
    import sys
    from fsa import make_linear_fsa
    from semiring import Prob, SumTimes, Count
    from ply_cfg import read_grammar
    from cfg import CFG, topsort_cfg
    from rule import CFGProduction
    from slicesampling import SliceVariables
    from inference import inside, sample, normalised_edge_inside
    from itertools import chain
    from utils import make_nltk_tree
    from collections import Counter
    from symbol import make_recursive_symbol
    import numpy as np

    semiring = SumTimes

    cfg = read_grammar(open('../../example/psg', 'r'), transform=semiring.from_real)
    
    input_str = '1 2 3 4'
    fsa = make_linear_fsa(input_str, semiring)

    for word in fsa.itersymbols():
        if not cfg.is_terminal(word):
            cfg.add(CFGProduction(Nonterminal('X'), [word], semiring.one))
   
    # TODO: beta shape parameters
    u = SliceVariables({}, a=0.2, b=1)
    samples = []
    # TODO: arg: number of samples
    while len(samples) < 1100:
        parser = Nederhof(cfg, fsa, semiring=semiring, slice_variables=u, make_symbol=make_recursive_symbol)
        forest = parser.do()
        if not forest:
            print 'NO PARSE FOUND'
            print
            u.reset()
        else:
            #for r in forest.iterrules_topdown():
            #    print r
            #print
            
            topsorted = list(chain(*topsort_cfg(forest)))
            uniformdist = parser.reweight(forest)
            #Ic = inside(forest, topsorted, Count, omega=lambda e: 1)
            #print 'FOREST: %d edges, %d nodes and %d paths' % (len(forest), len(forest.nonterminals), Ic[topsorted[-1]])
            Iv = inside(forest, topsorted, semiring, omega=lambda e: uniformdist[e])
            # TODO: arg: batch size
            batch = list(sample(forest, Nonterminal('GOAL'), semiring, Iv=Iv, N=1, omega=lambda e: uniformdist[e]))
            # resampling step
            # TODO: args: resampling
            i = np.random.randint(0, len(batch))
            d = batch[i]
            #print 'SAMPLE'
            #print make_nltk_tree(d)
            conditions = {r.lhs.label: semiring.as_real(r.weight) for r in d}
            #print 'CONDITIONS'
            #for k, v in conditions.iteritems():
            #    print '%s:%s-%s) %s' % (k[0], k[1], k[2], v)
            u.reset(conditions)
            samples.append(d)
            
            print
    
    print 'RESULT'
    count = Counter(samples[100:])
    for d, n in reversed(count.most_common()):
        t = make_nltk_tree(d)
        print '%dx %s' % (n, t)
        print
        #t.draw()

    """
    topsorted = chain(*topsort_cfg(forest))
    Iv = inside(forest, topsorted, Prob)
    from collections import Counter
    count = Counter()
    count.update(sample(forest, Nonterminal('GOAL'), Prob, Iv, N=100))
    for d, n in reversed(count.most_common()):
        t = make_nltk_tree(d)
        print '%dx %s' % (n, t)
        print
        #t.draw()
    """
