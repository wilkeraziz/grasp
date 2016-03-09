"""
:Authors: - Wilker Aziz
"""

from collections import defaultdict
from grasp.cfg.symbol import Terminal
from grasp.cfg.srule import InputGroupView

class SCFG(object):
    """
    A Synchronous CFG. Note that SCFG does not inherit from CFG and should not be assumed
    to have a similar interface.

    We treat a SCFG pretty much as a hash table between
    from (lhs, irhs) to a list of synchronous rules.

    You can use SCFG objects to manage synchronous rules and to get input/output projections as CFG objects.
    """

    def __init__(self, syncrules=[]):
        """

        :param syncrules: an iterable sequence of synchronous rules.
        :return:
        """
        self._srules = defaultdict(lambda: defaultdict(list))
        self._sigma = set()
        self._delta = set()
        self._rules = []
        for srule in syncrules:
            self.add(srule)

    def __len__(self):
        """The total number of synchronous rules in the container."""
        return sum(sum(len(rules) for rules in by_irhs.values()) for by_irhs in self._srules.values())

    def in_ivocab(self, word):
        return word in self._sigma

    def in_ovocab(self, word):
        return word in self._delta

    def __iter__(self):
        return iter(self._rules)

    def add(self, srule):
        """Add a synchronous rule to the container."""
        self._rules.append(srule)
        self._srules[srule.lhs][srule.irhs].append(srule)
        self._sigma.update(filter(lambda s: isinstance(s, Terminal), srule.irhs))
        self._delta.update(filter(lambda s: isinstance(s, Terminal), srule.orhs))

    def __str__(self):
        lines = []
        for lhs, by_irhs in self._srules.items():
            for i_rhs, rules in by_irhs.items():
                for rule in rules:
                    lines.append(str(rule))
        return '\n'.join(lines)

    def iter_inputgroupview(self):
        for lhs, by_irhs in self._srules.items():
            for i_rhs, srules in by_irhs.items():
                yield InputGroupView(srules)



    #def input_projection(self, semiring, weighted=False):
    #    """
    #    A projection is the grammar resulting from marginalising over the alternative rules.
    #    You can set `weighted` to False if you want the projection to be unweighted.
    #    :param semiring: must provide `zero`, `one` and `plus`.
    #    :param weighted:
    #    :return: a CFG.
    #    """
    #    if not weighted:
    #        def make_rule(lhs, rhs, srules):
    #            return CFGProduction(lhs, rhs, semiring.one)
    #    else:
    #        def make_rule(lhs, rhs, srules):
    #            return CFGProduction(lhs, rhs, reduce(semiring.plus, (r.weight for r in srules), semiring.zero))
    #
    #    def iterrules():
    #        for lhs, by_irhs in self._srules.items():
    #            for f_rhs, srules in by_irhs.items():
    #                yield make_rule(lhs, f_rhs, srules)
    #
    #    return CFG(iterrules())

    #def iteroutputrules(self, lhs, irhs):
    #    """Iterate through synchronous rules matching given LHS symbol and a given input RHS sequence."""
    #    srules = self._srules.get(lhs, None)
    #    if srules is None:
    #        return iter([])
    #    return iter(srules.get(irhs, []))
