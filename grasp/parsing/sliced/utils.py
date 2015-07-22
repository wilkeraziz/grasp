"""
:Authors: - Wilker Aziz
"""

from collections import defaultdict


def make_conditions(d, semiring):
    conditions = {r.lhs.label: semiring.as_real(r.weight) for r in d}
    return conditions


def make_batch_conditions(D, semiring):
    if len(D) == 1:
        d = D[0]
        conditions = {r.lhs.label: semiring.as_real(r.weight) for r in d}
    else:
        conditions = defaultdict(set)
        for d in D:
            [conditions[r.lhs.label].add(semiring.as_real(r.weight)) for r in d]
        conditions = {s: min(thetas) for s, thetas in conditions.items()}
    return conditions