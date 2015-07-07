"""
This module contains class definitions for rules, such as a context-free production.

:Authors: - Wilker Aziz
"""

import re
import logging
from weakref import WeakValueDictionary

from .symbol import Terminal, Nonterminal
from collections import defaultdict


class CFGProduction(object):
    """
    Implements a context-free production. 
    
    References to productions are managed by the CFGProduction class.
    We use WeakValueDictionary for builtin reference counting.
    The symbols in the production must all be immutable (thus hashable).

    >>> CFGProduction(1, [1,2,3], 0.0)  # integers are hashable
    CFGProduction(1, (1, 2, 3), 0.0)
    >>> CFGProduction(('S', 1, 3), [('X', 1, 2), ('X', 2, 3)], 0.0)  # tuples are hashable
    CFGProduction(('S', 1, 3), (('X', 1, 2), ('X', 2, 3)), 0.0)
    >>> CFGProduction(Nonterminal('S'), [Terminal('<s>'), Nonterminal('X'), Terminal('</s>')], 0.0)  # Terminals and Nonterminals are also hashable
    CFGProduction(Nonterminal('S'), (Terminal('<s>'), Nonterminal('X'), Terminal('</s>')), 0.0)
    """

    _rules = WeakValueDictionary()
    #_rules = defaultdict(None)

    def __new__(cls, lhs, rhs, weight):
        """The symbols in lhs and in the rhs must be hashable."""
        skeleton = (lhs, tuple(rhs), weight)
        obj = CFGProduction._rules.get(skeleton, None)
        if not obj:
            obj = object.__new__(cls)
            CFGProduction._rules[skeleton] = obj
            obj._skeleton = skeleton
        return obj
    
    @property
    def lhs(self):
        """Return the LHS symbol (a Nonterminal) aka the head."""
        return self._skeleton[0]

    @property
    def rhs(self):
        """A tuple of symbols (terminals and nonterminals) representing the RHS aka the tail."""
        return self._skeleton[1]
    
    @property
    def weight(self):
        return self._skeleton[2]

    def __repr__(self):
        return '%s(%s, %s, %s)' % (CFGProduction.__name__, repr(self.lhs), repr(self.rhs), repr(self.weight))

    def __str__(self):
        return '%s ||| %s ||| %s' % (self.lhs, ' '.join(str(s) for s in self.rhs), self.weight)

    def pprint(self, make_symbol):
        return '%s ||| %s ||| %s' % (make_symbol(self.lhs), ' '.join(str(make_symbol(s)) for s in self.rhs), self.weight)


class SCFGProduction(object):
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

    def __str__(self):
        A = iter(self.alignment)
        return '%s ||| %s ||| %s ||| %s' % (
            self.lhs,
            ' '.join(str(s) for s in self.irhs),
            ' '.join(str(s) if isinstance(s, Terminal) else '[%d]' % (next(A)) for s in self.orhs),
            self.weight())

    def project_rhs(self):
        """Computes the target context-free production by projecting source labels through nonterminal alignment."""
        alignment = iter(self.alignment)
        f_nts = tuple(filter(lambda s: isinstance(s, Nonterminal), self.irhs))
        return tuple(s if isinstance(s, Terminal) else f_nts[next(alignment) - 1] for s in self.orhs)


def get_oov_cfg_productions(oovs, unk_lhs, weight):
    for word in oovs:
        r = CFGProduction(Nonterminal(unk_lhs), [Terminal(word)], weight)
        logging.debug('Passthrough rule for %s: %s', word, r)
        yield r
