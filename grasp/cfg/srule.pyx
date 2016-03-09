"""
:Authors: - Wilker Aziz
"""

import logging
import re
from grasp.cfg.symbol cimport Symbol, Terminal, Nonterminal
from grasp.cfg.rule cimport Rule
from grasp.ptypes cimport weight_t

F_NT_INDEX_RE = re.compile('^([^,]+)(,(\d+))?$')
E_NT_INDEX_RE = re.compile('^(([^,]+),)?(\d+)$')


cdef class SCFGProduction:
    """
    Implements a synchronous context-free production.
    """

    def __init__(self, Nonterminal lhs,
                 irhs,
                 orhs,
                 nt_alignment,
                 fmap):
        """

        :param lhs:
        :param irhs:
        :param orhs:
        :param nt_alignment: sequence of 1-based nonterminal positions which the input nonterminals align to
            For example, for a rule such as X -> a X_1 b X_2 c X_3 d | r X_2 s X_3 t X_1 u,
             nonterminal alignments look like [3, 1, 2]
        :param fmap:
        :return:
        """
        self._lhs = lhs
        self._irhs = tuple(irhs)
        self._orhs = tuple(orhs)
        self._nt_alignment = tuple(nt_alignment)
        self._fmap = dict(fmap)

    @staticmethod
    def create(lhs, irhs, orhs, fmap):
        irhs = list(irhs)
        orhs = list(orhs)
        f_nts = [i for i, sym in filter(lambda i_s: isinstance(i_s[1], Nonterminal), enumerate(irhs))]
        e_nts = [j for j, sym in filter(lambda i_s: isinstance(i_s[1], Nonterminal), enumerate(orhs))]

        # adjust source RHS nonterminal symbols
        for k, i in enumerate(f_nts):
            f_sym = irhs[i].label
            result = F_NT_INDEX_RE.search(f_sym)
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
            result = E_NT_INDEX_RE.search(e_sym)
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

    property lhs:
        def __get__(self):
            return self._lhs

    property irhs:
        def __get__(self):
            return self._irhs

    property orhs:
        def __get__(self):
            return self._orhs

    #cpdef weight_t weight(self):
    #    return 0.0  # TODO: compute dot product

    property alignment:
        def __get__(self):
            return self._nt_alignment

    property fpairs:
        def __get__(self):
            return self._fmap.items()

    #property fmap:
    #    def __get__(self):
    #        return self._fmap

    cpdef weight_t fvalue(self, fname, weight_t default=0.0):
        return self._fmap.get(fname, default)

    def __str__(self):
        A = iter(self.alignment)
        return '%s ||| %s ||| %s ||| %s' % (
            repr(self.lhs),
            ' '.join(repr(s) for s in self.irhs),
            ' '.join(repr(s) if isinstance(s, Terminal) else '[%d]' % (next(A)) for s in self.orhs),
            ' '.join('{0}={1}'.format(k, v) for k, v in sorted(self._fmap.items())))

    def project_rhs(self):
        """Computes the target RHS by projecting source labels through nonterminal alignment."""
        alignment = iter(self.alignment)
        f_nts = tuple(filter(lambda s: isinstance(s, Nonterminal), self.irhs))
        return tuple(s if isinstance(s, Terminal) else f_nts[next(alignment) - 1] for s in self.orhs)


cdef class InputView(Rule):

    def __init__(self, SCFGProduction srule):
        self._srule = srule

    property lhs:
        def __get__(self):
            return self._srule.lhs

    property rhs:
        def __get__(self):
            return self._srule.irhs

    property srule:
        def __get__(self):
            return self._srule

    cpdef weight_t fvalue(self, fname, weight_t default=0.0):
        return self._srule.fvalue(fname, default)

    def __str__(self):
        return 'InputView(%s)' % self._srule


cdef class OutputView(Rule):

    def __init__(self, SCFGProduction srule):
        self._srule = srule

    property lhs:
        def __get__(self):
            return self._srule.lhs

    property rhs:
        def __get__(self):
            return self._srule.orhs

    property srule:
        def __get__(self):
            return self._srule

    cpdef weight_t fvalue(self, fname, weight_t default=0.0):
        return self._srule.fvalue(fname, default)

    def __str__(self):
        return 'OutputView(%s)' % self._srule


cdef class InputGroupView(Rule):

    def __init__(self, srules):
        assert len(srules) > 0, 'A group cannot be empty'
        assert len(frozenset(r.irhs for r in srules)) == 1, 'All synchronous rules in a group must share the same input RHS'
        self._srules = tuple(srules)

    property lhs:
        def __get__(self):
            return self._srules[0].lhs

    property rhs:
        def __get__(self):
            return self._srules[0].irhs

    property group:
        def __get__(self):
            return self._srules

    cpdef weight_t fvalue(self, fname, weight_t default=0.0):
        raise NotImplementedError()

    def __str__(self):
        return '%s ||| %s ||| %d projections' % (repr(self.lhs),
                                                 ' '.join(repr(s) for s in self.rhs),
                                                 len(self._srules))


cdef class OutputGroupView(Rule):

    def __init__(self, srules):
        assert len(srules) > 0, 'A group cannot be empty'
        assert len(frozenset(r.irhs for r in srules)) == 1, 'All synchronous rules in a group must share the same output RHS'
        self._srules = tuple(srules)

    property lhs:
        def __get__(self):
            return self._srules[0].lhs

    property rhs:
        def __get__(self):
            return self._srules[0].orhs

    property group:
        def __get__(self):
            return self._srules

    cpdef weight_t fvalue(self, fname, weight_t default=0.0):
        raise NotImplementedError()

    def __str__(self):
        return '%s ||| %s ||| %d projections' % (repr(self.lhs),
                                                 ' '.join(repr(s) for s in self.rhs),
                                                 len(self._srules))