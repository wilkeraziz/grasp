"""
:Authors: - Wilker Aziz
"""

import re
from collections import defaultdict
from .symbol import Terminal, Nonterminal
from .rule import CFGProduction


_LHS_SPLIT_ = re.compile(r'(.+)_.+')
_LHS_PERMUTATION_ = re.compile(r'P([0-9]+)')


def remove_split(sym):
    if isinstance(sym, Nonterminal):
        m = _LHS_SPLIT_.match(sym.label)
        if m is None:
            return sym
        else:
            return Nonterminal(m.group(1))
    else:
        return sym


def permutation_length(sym):
    if isinstance(sym, Terminal):
        return 0
    m = _LHS_PERMUTATION_.match(sym.label)
    if m is None:
        return 0
    else:
        return len(m.group(1))


def itercoarse(cfg, semiring, get_symbol=remove_split):

    bylhs = defaultdict(lambda: defaultdict(lambda: semiring.zero))

    for lhs, rules in cfg.iteritems():
        if permutation_length(lhs) > 2:
            #logging.info('Skipping rule for %s', lhs)
            continue
        clhs = get_symbol(lhs)
        crules = bylhs[clhs]
        for rule in rules:
            crhs = tuple(get_symbol(s) for s in rule.rhs)
            crules[crhs] = semiring.plus(crules[crhs], rule.weight)
            #logging.info('i=%s o=%s', rule, (clhs, crhs))

    for lhs, byrhs in bylhs.items():
        for rhs, w in byrhs.items():
            yield CFGProduction(lhs, rhs, w)


def refine_conditions(coarse_spans, cfg, semiring, get_symbol=remove_split):
    conditions = defaultdict(None)
    for lhs in cfg.iternonterminals():
        clhs = get_symbol(lhs)
        spans = coarse_spans.get(clhs, set())
        if spans:
            w = semiring.zero
            for rule in cfg.iterrules(lhs):
                if semiring.gt(rule.weight, w):
                    w = rule.weight
            for span in spans:
                conditions[(lhs, span[0], span[1])] = w
    return conditions


def iteritg(cfg):

    for lhs, rules in cfg.iteritems():
        if permutation_length(lhs) > 2:
            #logging.info('Skipping rule for %s', lhs)
            continue
        for rule in rules:
            yield rule