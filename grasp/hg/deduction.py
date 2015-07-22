"""
:Authors: - Wilker Aziz
"""
import itertools


def merge(states):
    return None

def init(scorers, bottomup=False):
    states = []
    for scorer in scorers:
        states.append(scorer.init(bottomup))
    return merge(states)

def axioms(goal, start, cfg, scorers, ifactory, bottomup=False):
    """
    Bottom-up

        [X -> * alpha, s] for X -> alpha \in G and s \in init()

        Parsing(w_1...w_n) states:
            i for i in 1 .. n

    Top-down

        [S' -> * S, s] for S start symbol in G and s \in init()

        Parsing(fsa) state:
            i where i is initial in fsa

        LM state:
            BOS_1^k where k = n-1 and n is the LM order

    :param goal:
    :param start:
    :param cfg:
    :param sfactory:
    :param ifactory:
    :param bottomup:
    :return:
    """

    if bottomup:
        for state in init(scorers, bottomup):
            for rule in cfg:
                ifactory.get(rule, state)
    else:
        for state in init(scorers, bottomup):
            for s in start:
                for rule in cfg.iterrules(s):
                    ifactory.get(rule, state)


def scan(item, scorers):
    if all(scorer.can_scan(item.state, item.next) for scorer in scorers):
        states = []
        for scorer in scorers:
            states.append(scorer.update(item.state, item.next))
        return merge(states)
    else:
        return None
