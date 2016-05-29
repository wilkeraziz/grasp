"""

:Authors: - Wilker Aziz
"""

from grasp.ptypes cimport weight_t, id_t, boolean_t

import numpy as np
cimport numpy as np


cdef class WeightFunction:
    """
    Assigns a weight to an edge.
    """

    def __init__(self):
        pass

    def __call__(self, id_t e):
        return self.value(e)

    cpdef weight_t value(self, id_t e): pass

    cpdef weight_t reduce(self, BinaryOperator op, iterable):
        cdef id_t e
        return op.reduce([self.value(e) for e in iterable])


cpdef weight_t derivation_weight(Hypergraph forest, tuple edges, Semiring semiring, WeightFunction omega=None):
    """
    Compute the weight of a derivation by reducing a given weight function.
    :param forest: Hypergraph
    :param edges: derivation (sequence of edges)
    :param semiring: provides times operator
    :param omega: a weight function (defaults to HypergraphLookupFunction which consists in the edge's weight in the
        given hypergraph).
    :return: weight of derivation
    """
    cdef id_t e
    if omega is None:
        return semiring.times.reduce([forest.weight(e) for e in edges])
    else:
        return semiring.times.reduce([omega.value(e) for e in edges])


cdef class ConstantFunction(WeightFunction):
    """
    A function that always returns the same constant.
    """

    def __init__(self, weight_t constant):
        self.constant = constant

    cpdef weight_t value(self, id_t e):
        return self.constant

    def __repr__(self):
        return '%s(%r)' % (ConstantFunction.__name__, self.constant)


cdef class ReducedFunction(WeightFunction):
    """
    A function that reduces the weights returned by a sequence of other functions using a custom operator.
    """

    def __init__(self, BinaryOperator op, functions):
        self.functions = tuple(functions)
        self.op = op

    cpdef weight_t value(self, id_t e):
        cdef WeightFunction func
        return self.op.reduce([func.value(e) for func in self.functions])

    def __repr__(self):
        return '%s(%r, %r)' % (ReducedFunction.__name__, self.op, self.functions)


cdef class TableLookupFunction(WeightFunction):
    """
    A value function that consists in plain simple table lookup.
    """

    def __init__(self, weight_t[::1] table):
        self.table = table

    cpdef weight_t value(self, id_t e):
        return self.table[e]

    def __repr__(self):
        return '%s(...)' % TableLookupFunction.__name__


cdef class BooleanFunction(WeightFunction):

    def __init__(self, boolean_t[::1] table, weight_t zero, weight_t one):
        self.table = table
        self.zero = zero
        self.one = one

    cpdef weight_t value(self, id_t e):
        if self.table[e]:
            return self.one
        else:
            return self.zero


cdef class HypergraphLookupFunction(WeightFunction):
    """
    A value function which reproduces the edge's weight in a hypergraph.
    """

    def __init__(self, Hypergraph hg):
        self.hg = hg

    cpdef weight_t value(self, id_t e):
        return self.hg.weight(e)

    def __repr__(self):
        return '%s(...)' % HypergraphLookupFunction.__name__


cdef class ScaledFunction(WeightFunction):
    """
    A function that scales the result of another function.
    """

    def __init__(self, WeightFunction func, weight_t scalar):
        self.func = func
        self.scalar = scalar

    cpdef weight_t value(self, id_t e):
        return self.func.value(e) * self.scalar

    def __repr__(self):
        return '%s(%r, %r)' % (ScaledFunction.__name__, self.func, self.scalar)


cdef class ThresholdFunction(WeightFunction):
    """
    Applies an edge-dependent threshold to each edge:
            if func(e) > thresholdfunc(e):  # in an input semiring
                return one  # in an output semiring
            else:
                return zero  # in an output semiring
    """

    def __init__(self, WeightFunction func, Semiring input_semiring, WeightFunction thresholdfunc=None, Semiring output_semiring=None):
        """
        Applies an edge-dependent threshold to each edge:
            if func(e) > thresholdfunc(e):  # in the input semiring
                return output_semiring.one
            else:
                return output_semiring.zero

        :param func: the function that assesses the weight of an edge
        :param input_semiring: func(e) and thresholdfunc(e) is assumed to be in this semiring
        :param thresholdfunc: the function that determines the threshold associated with each edge
            defaults to ConstantFunction(input_semiring.zero)
        :param output_semiring: self.value(e) outputs output_semiring.one or output_semiring.zero
            defaults to input_semiring
        """
        self.func = func
        self.input_semiring = input_semiring
        if thresholdfunc is None:
            self.thresholdfunc = ConstantFunction(input_semiring.zero)
        else:
            self.thresholdfunc = thresholdfunc
        if output_semiring is None:
            self.output_semiring = input_semiring
        else:
            self.output_semiring = output_semiring

    cpdef weight_t value(self, id_t e):
        """
        Returns 1 (in the output semiring) if the edge's value is greater than the threshold (both in the input semiring),
        otherwise it returns 0 (in the output semiring).
        """
        if self.input_semiring.gt(self.func.value(e), self.thresholdfunc.value(e)):
            return self.output_semiring.one
        else:
            return self.output_semiring.zero

    def __repr__(self):
        return '%s(%r, %r, %r, %r)' % (ThresholdFunction.__name__,
                                       self.func,
                                       self.input_semiring,
                                       self.thresholdfunc,
                                       self.output_semiring)
