
import unittest

from grasp.formal.topsort import AcyclicTopSortTable, RobustTopSortTable
from grasp.inference._inference import viterbi_derivation, sample_derivations, AncestralSampler
from grasp.inference._value import EdgeWeight
from grasp.cfg import CFG, Terminal, Nonterminal
from grasp.cfg.rule import NewCFGProduction as CFGProduction
import grasp.semiring as semiring
from collections import Counter
from grasp.formal.scfgop import cfg_to_hg
from grasp.cfg.model import PCFG


def get_rule(lhs, rhs, logprob):
    return CFGProduction.MakeStandardCFGProduction(lhs, rhs, logprob, fname='LogProb', transform=float)


class InferenceTestCase(unittest.TestCase):

    def setUp(self):
        self.semiring = semiring.viterbi
        self.cfg = CFG()
        self.cfg.add(get_rule(Nonterminal('S02'),
                                   [Nonterminal('S01'), Nonterminal('X12'), Nonterminal('PUNC')],
                                   self.semiring.from_real(0.5)))
        self.cfg.add(get_rule(Nonterminal('S01'),
                                   [Nonterminal('X01')],
                                   self.semiring.from_real(0.1)))
        self.cfg.add(get_rule(Nonterminal('X01'),
                                   [Terminal('Hello')],
                                   self.semiring.from_real(0.7)))
        self.cfg.add(get_rule(Nonterminal('X01'),
                                   [Terminal('hello')],
                                   self.semiring.from_real(0.1)))
        self.cfg.add(get_rule(Nonterminal('X12'),
                                   [Terminal('World')],
                                   self.semiring.from_real(0.6)))
        self.cfg.add(get_rule(Nonterminal('X12'),
                                   [Terminal('world')],
                                   self.semiring.from_real(0.2)))
        self.cfg.add(get_rule(Nonterminal('PUNC'),
                                   [Terminal('!')],
                                   self.semiring.from_real(0.1)))
        self.cfg.add(get_rule(Nonterminal('PUNC'),
                                   [Terminal('!!!')],
                                   self.semiring.from_real(0.3)))
        self.cfg.add(get_rule(Nonterminal('A'),
                                   [Terminal('dead')],
                                   self.semiring.from_real(0.3)))
        self.cfg.add(get_rule(Nonterminal('B'),
                                   [],
                                   self.semiring.from_real(0.3)))
        self.forest = cfg_to_hg([self.cfg], [], PCFG('LogProb'))
        self.tsort = AcyclicTopSortTable(self.forest)
        self.omega = EdgeWeight(self.forest)

    def test_viterbi(self):
        d = viterbi_derivation(self.forest, self.tsort)
        score = self.omega.reduce(self.semiring.times, d)
        self.assertAlmostEqual(self.semiring.as_real(score), 0.006299999999999999)

        ten = [viterbi_derivation(self.forest, self.tsort) for _ in range(10)]
        self.assertEqual(len(set(ten)), 1)
        self.assertAlmostEqual(self.semiring.as_real(self.omega.reduce(self.semiring.times,
                                                                       ten[0])),
                               0.006299999999999999)

        #der = [self.forest.rule(e) for e in d]
        #print('\n', score, self.semiring.as_real(score))
        #for r in der:
        #    print(r)

    def test_sample(self):
        counts = Counter(sample_derivations(self.forest, self.tsort, 1000))
        ranking = counts.most_common()
        top, n = ranking[0]
        score = self.omega.reduce(semiring.inside.times, top)
        self.assertAlmostEqual(self.semiring.as_real(score), 0.006299999999999999)
        self.assertTrue(n/1000 > 0.4)
        """
        print()
        for d, n in counts.most_common():
            score = self.omega.reduce(semiring.inside.times, d)
            der = [self.forest.rule(e) for e in d]
            print(score, semiring.inside.as_real(score), n, n/1000)
            for r in der:
                print(r)
            print()
        """

    def test_ancestral(self):
        sampler = AncestralSampler(self.forest, self.tsort)
        size = 1000
        counts = Counter(sampler.sample(size))
        ranking = counts.most_common()
        top, n = ranking[0]
        #print()
        #print(n/size, sampler.prob(top))
        self.assertEqual(sampler.Z, -4.358310174252031)
        self.assertAlmostEqual(n/size, sampler.prob(top), places=1, msg='Random effects apply - double check.')



if __name__ == '__main__':
    unittest.main()
