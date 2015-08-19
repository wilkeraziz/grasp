"""
:Authors: - Wilker Aziz
"""

import logging
import re
from collections import defaultdict
from grasp.cfg.symbol import Terminal, Nonterminal
from grasp.cfg.rule import Rule


class SCFGProduction(Rule):
    """
    Implements a synchronous context-free production.

    Unlike CFGProduction, this class offers no reference management.
    """

    F_NT_INDEX_RE = re.compile('^([^,]+)(,(\d+))?$')
    E_NT_INDEX_RE = re.compile('^(([^,]+),)?(\d+)$')

    def __init__(self, lhs,
                 irhs,
                 orhs,
                 nt_alignment,
                 fmap):
        self._lhs = lhs
        self._irhs = tuple(irhs)
        self._orhs = tuple(orhs)
        self._nt_aligment = tuple(nt_alignment)
        self._fmap = defaultdict(None, fmap)

    @staticmethod
    def create(lhs, irhs, orhs, fmap):
        irhs = list(irhs)
        orhs = list(orhs)
        f_nts = [i for i, sym in filter(lambda i_s: isinstance(i_s[1], Nonterminal), enumerate(irhs))]
        e_nts = [j for j, sym in filter(lambda i_s: isinstance(i_s[1], Nonterminal), enumerate(orhs))]

        # adjust source RHS nonterminal symbols
        for k, i in enumerate(f_nts):
            f_sym = irhs[i].label
            result = SCFGProduction.F_NT_INDEX_RE.search(f_sym)
            if result is None or len(result.groups()) != 3:
                raise ValueError('Invalid source right-hand side nonterminal symbol: %s' % f_sym)
            label, index = result.group(1), result.group(3)
            if index is not None:
                index = int(index)
                if index != k + 1:
                    logging.warning('I am discarding the index of a source right-hand side nonterminal symbol for it is inconsistent. Expected %d, got %d.' % (k + 1, index))
            irhs[i] = Nonterminal(label)

        nt_alignment = []
        # adjust target RHS nonterminal symbols
        for j in e_nts:
            e_sym = orhs[j].label
            result = SCFGProduction.E_NT_INDEX_RE.search(e_sym)
            if result is None or len(result.groups()) != 3:
                raise ValueError('Invalid target right-hand side nonterminal symbol: %s' % e_sym)
            label, index = result.group(2), int(result.group(3))
            if not (0 < index <= len(f_nts)):
                raise ValueError('Reference to a nonexistent source right-hand side nonterminal: %d/%d' % (index, len(f_nts)))
            f_sym = irhs[f_nts[index - 1]]
            if label is None:
                label = f_sym
            elif label != f_sym.label:
                logging.warning('I am discarding a target right-hand side label for it differs from its aligned source label. Expected %s, got %s.' % (f_sym.label, label))
            orhs[j] = Nonterminal(label)
            nt_alignment.append(index)

        # stores the source production
        # and the target projection
        return SCFGProduction(lhs, irhs, orhs, nt_alignment, fmap)

    @property
    def lhs(self):
        return self._lhs

    @property
    def irhs(self):
        return self._irhs

    @property
    def orhs(self):
        return self._orhs

    def weight(self):
        return 0.0  # TODO: compute dot product

    @property
    def alignment(self):
        return self._nt_aligment

    @property
    def fvpairs(self):
        return self._fmap.items()

    @property
    def fmap(self):
        return self._fmap

    def __str__(self):
        A = iter(self.alignment)
        return '%s ||| %s ||| %s ||| %s' % (
            repr(self.lhs),
            ' '.join(repr(s) for s in self.irhs),
            ' '.join(repr(s) if isinstance(s, Terminal) else '[%d]' % (next(A)) for s in self.orhs),
            ' '.join('{0}={1}'.format(k, v) for k, v in sorted(self._fmap.items())))

    def project_rhs(self):
        """Computes the target context-free production by projecting source labels through nonterminal alignment."""
        alignment = iter(self.alignment)
        f_nts = tuple(filter(lambda s: isinstance(s, Nonterminal), self.irhs))
        return tuple(s if isinstance(s, Terminal) else f_nts[next(alignment) - 1] for s in self.orhs)
