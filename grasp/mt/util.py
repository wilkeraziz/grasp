"""
:Authors: - Wilker Aziz
"""
import logging
import pickle

from grasp.scoring.lookup import RuleTable
from grasp.scoring.stateless import WordPenalty, ArityPenalty
from grasp.scoring.lm import StatelessLM, KenLM

from grasp.cfg.symbol import Nonterminal, Terminal
from grasp.cfg.srule import SCFGProduction, InputGroupView, OutputView

from grasp.recipes import smart_wopen


class GoalRuleMaker:

    def __init__(self, goal_str, start_str, n=0):
        self._n = n
        self._goal_str = goal_str
        self._start_str = start_str

    def _make_goal_str(self, n=None):
        return '{0}{1}'.format(self._goal_str, self._n) if n is None else '{0}{1}'.format(self._goal_str, n)

    def update(self):
        """Updates the state of the factory"""
        self._n += 1

    def get_srule(self):
        """Returns that goal rule based on the state of the factory"""
        if self._n == 0:
            rhs = [Nonterminal(self._start_str)]
        else:
            rhs = [Nonterminal(self._make_goal_str(self._n - 1))]
        return SCFGProduction(Nonterminal(self._make_goal_str()), rhs, rhs, [1], {'GoalRule': 1.0})

    def get_next_srule(self):
        """Returns what would be the next goal rule without updating the state of the factory."""
        rhs = [Nonterminal(self._make_goal_str(self._n))]
        return SCFGProduction(Nonterminal(self._make_goal_str(self._n + 1)), rhs, rhs, [1], {'GoalRule': 1.0})

    def get_iview(self):
        return InputGroupView([self.get_srule()])

    def get_oview(self):
        return OutputView(self.get_srule())


def make_dead_srule(lhs='X', dead='<dead-end>', fname='DeadRule'):
    return SCFGProduction(Nonterminal(lhs), (Terminal(dead),), (Terminal(dead),), [], {fname: 1.0})


def load_feature_extractors(args) -> 'list of extractors':  # TODO: generalise it and use a configuration file
    """
    Load feature extractors depending on command line options.

    For now we have the following extractors:

        * RuleTable: named features in the rule table (a lookup scorer)
        * WordPenalty
        * ArityPenalty
        * KenLMScorer

    :param args:
    :return: a vector of Extractor objects
    """
    extractors = []

    if args.rt:
        extractor = RuleTable(uid=len(extractors),
                              name='RuleTable')
        extractors.append(extractor)
        logging.debug('Scorer: %r', extractor)

    if args.wp:
        extractor = WordPenalty(uid=len(extractors),
                                name=args.wp[0],
                                penalty=float(args.wp[1]))
        extractors.append(extractor)
        logging.debug('Scorer: %r', extractor)

    if args.ap:
        extractor = ArityPenalty(uid=len(extractors),
                                 name=args.ap[0],
                                 penalty=float(args.ap[1]))
        extractors.append(extractor)
        logging.debug('Scorer: %r', extractor)

    if args.slm:
        extractor = StatelessLM(uid=len(extractors),
                                name=args.slm[0],
                                order=int(args.slm[1]),
                                path=args.slm[2])
        extractors.append(extractor)
        logging.debug('Scorer: %r', extractor)

    if args.lm:
        extractor = KenLM(uid=len(extractors),
                          name=args.lm[0],
                          order=int(args.lm[1]),
                          path=args.lm[2])
        extractors.append(extractor)
        logging.debug('Scorer: %r', extractor)

    return extractors


def save_forest(path, hg, ffs=None):
    with smart_wopen(path) as fo:
        print('# FOREST', file=fo)
        print(hg, file=fo)
        if ffs is not None:
            print('# FF')
            for ff in ffs:
                print(ff, file=fo)


def save_ffs(path, ffs):
    with open(path, 'wb') as fo:
        pickle.dump(ffs, fo)


def load_ffs(path):
    with open(path, 'rb') as fo:
        return pickle.load(fo)


