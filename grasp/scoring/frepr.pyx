"""
This module contains definitions for feature representations.


:Authors: - Wilker Aziz
"""
from grasp.ptypes cimport weight_t
import grasp.ptypes as ptypes

from grasp.semiring.operator cimport BinaryOperator

cimport numpy as np
import numpy as np

import itertools


cdef class FRepr:
    """
    A feature representation object manages containers for features.
    It abstracts away basic operations:
        1. scalar product
        2. dot product: which computes the dot product between two representations of the same type
            repr * repr -> scalar
        3. hadamard: which applies a component wise custom operation to two representations of the same type
            repr o repr -> repr
    """

    cpdef FRepr prod(self, weight_t scalar):
        raise NotImplementedError('I do not know how to operate over representations')

    cpdef weight_t dot(self, FRepr w) except *:
        raise NotImplementedError('I do not know how to operate over representations')

    cpdef FRepr hadamard(self, FRepr rhs, BinaryOperator op):
        """
        Hadamard performs elementwise binary operations, where this object contributes with the LHS and the
        argument contributes with the RHS of the binary operation.
        For example, this can be used to apply semiring.times to the components of two edges.
        """
        raise NotImplementedError('I do not know how to operate over representations')

    cpdef FRepr densify(self):
        raise NotImplementedError('I do not know how to make dense representations')

    cpdef FRepr elementwise(self, UnaryOperator op):
        """
        This is applies a unary operator on each element of the representation.
        For example, it can be used to apply semiring.inverse to the components.
        You can also wrap a power in FixedRHS(power, semiring.times.power) which will be equivalent
         to raising each element of the representation to that power.
        """
        raise NotImplementedError('I do not know how to operate over representations')

    cpdef FRepr elementwise_b(self, weight_t rhs, BinaryOperator op):
        """
        This is applies a binary operator on each element of the representation.
        The representation contributes with the LHS of the binary operation.
        For example, it can be used to raise the elements to a given power with semiring.times.power.
        """
        raise NotImplementedError('I do not know how to operate over representations')

    cpdef FRepr power(self, weight_t power, Semiring semiring):
        raise NotImplementedError('I do not know how to operate over representations')


cdef class FValue(FRepr):
    """
    This feature representation is a container for a single value.
    """

    def __init__(self, weight_t value):
        self.value = value

    cpdef FRepr prod(self, weight_t scalar):
        return FValue(self.value * scalar)

    cpdef weight_t dot(self, FRepr w) except *:
        return self.value * (<FValue?>w).value

    cpdef FRepr hadamard(self, FRepr rhs, BinaryOperator op):
        return FValue(op.evaluate(self.value, (<FValue?>rhs).value))

    cpdef FRepr densify(self):
        return self

    cpdef FRepr elementwise(self, UnaryOperator op):
        return FValue(op.evaluate(self.value))

    cpdef FRepr elementwise_b(self, weight_t rhs, BinaryOperator op):
        return FValue(op.evaluate(self.value, rhs))

    cpdef FRepr power(self, weight_t power, Semiring semiring):
        return FValue(semiring.power(self.value, power))

    def __getstate__(self):
        return {'value': self.value}

    def __setstate__(self, d):
        self.value = ptypes.weight(d['value'])

    def __len__(self):
        return 1

    def __iter__(self):
        return iter([self.value])

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return 'FValue(%r)' % self.value


cdef class FVec(FRepr):
    """
    This feature representation is a container for a vector.
    """

    def __init__(self, iterable):
        self.vec = np.array(iterable, dtype=ptypes.weight)

    def __getstate__(self):
        return {'vec': np.array(self.vec, dtype=ptypes.weight)}

    def __setstate__(self, d):
        self.vec = np.array(d['vec'], dtype=ptypes.weight)

    cpdef FRepr prod(self, weight_t scalar):
        cdef weight_t w
        return FVec([w * scalar for w in self.vec])

    cpdef weight_t dot(self, FRepr w) except *:
        return np.dot(self.vec, (<FVec?>w).vec)

    cpdef FRepr hadamard(self, FRepr rhs, BinaryOperator op):
        cdef weight_t a, b
        cdef FVec othervec = <FVec?>rhs
        if len(self.vec) != len(othervec.vec):
            raise ValueError('I cannot compute Hadamard product over vectors of different length.')
        return FVec([op.evaluate(a, b) for a, b in zip(self.vec, othervec.vec)])

    cpdef FRepr elementwise(self, UnaryOperator op):
        cdef weight_t mine
        return FVec([op.evaluate(mine) for mine in self.vec])

    cpdef FRepr elementwise_b(self, weight_t rhs, BinaryOperator op):
        cdef weight_t lhs
        return FVec([op.evaluate(lhs, rhs)for lhs in self.vec])

    cpdef FRepr power(self, weight_t power, Semiring semiring):
        cdef weight_t base
        return FVec([semiring.power(base, power) for base in self.vec])

    cpdef FRepr densify(self):
        return self

    def __len__(self):
        return len(self.vec)

    def __iter__(self):
        return iter(self.vec)

    def __str__(self):
        return ' '.join(str(x) for x in self.vec)

    def __repr__(self):
        return 'FVec(%r)' % list(self.vec)


cdef class FMap(FRepr):
    """
    This feature representation is a container for a vector of named features (represented as a dictionary).
    This is used for sparse features, thus, Hadamard accepts missing entries and uses the operator's identity
    for completion.
    """

    def __init__(self, iterable):
        self.map = dict(iterable)

    cpdef FRepr prod(self, weight_t scalar):
        return FMap([(k, v * scalar) for k, v in self.map.items()])

    cpdef weight_t dot(self, FRepr w) except *:
        cdef weight_t v
        cdef dict wmap = (<FMap?>w).map
        if len(self.map) < len(wmap):
            return np.sum([v * wmap.get(k, 0.0) for k, v in self.map.items()])
        else:
            return np.sum([v * self.map.get(k, 0.0) for k, v in wmap.items()])

    cpdef FRepr hadamard(self, FRepr rhs, BinaryOperator op):
        cdef object k
        cdef dict mapx = self.map
        cdef dict mapy = (<FMap?>rhs).map
        cdef set keys = set(mapx.keys())
        keys.update(mapy.keys())
        return FMap([(k, op.evaluate(mapx.get(k, op.identity), mapy.get(k, op.identity))) for k in keys])

    cpdef FRepr elementwise(self, UnaryOperator op):
        cdef weight_t v
        return FMap([(k, op.evaluate(v)) for k, v in self.map.items()])

    cpdef FRepr elementwise_b(self, weight_t rhs, BinaryOperator op):
        cdef weight_t lhs
        return FMap([(k, op.evaluate(lhs, rhs)) for k, lhs in self.map.items()])

    cpdef FRepr power(self, weight_t power, Semiring semiring):
        cdef weight_t base
        return FMap([(k, semiring.power(base, power)) for k, base in self.map.items()])

    cpdef FRepr densify(self):
        return FVec([self.map[k] for k in sorted(self.map.keys())])

    def __getstate__(self):
        return {'map': self.map}

    def __setstate__(self, d):
        self.map = dict(d['map'])

    def __len__(self):
        return len(self.map)

    def __iter__(self):
        return iter(self.map.items())

    def __str__(self):
        return ' '.join('{0}={1}'.format(k, v) for k, v in self.map.items())

    def __repr__(self):
        return 'FMap(%r)' % self.map


cdef class FComponents(FRepr):
    """
    This representation is a container for other representations.
    It is meant to be used by Scorer objects as those are the ones that manipulate representations of different types.
    """

    def __init__(self, iterable):
        """Here each component must be itself a FRepr"""
        self.components = list(iterable)
        if not all(isinstance(comp, FRepr) for comp in self.components):
            raise ValueError('Every component in an FComponents object must be itself a FRepr object')

    cpdef FComponents concatenate(self, FComponents other):
        if len(self) == 0:
            return other
        elif len(other) == 0:
            return self
        else:
            return FComponents(itertools.chain(self.components, other.components))

    cpdef FRepr prod(self, weight_t scalar):
        cdef FRepr comp
        return FComponents([comp.prod(scalar) for comp in self.components])

    cpdef weight_t dot(self, FRepr w) except *:
        cdef FRepr comp1, w1
        return np.sum([comp1.dot(w1) for comp1, w1 in zip(self.components, (<FComponents?>w).components)])

    cpdef FRepr hadamard(self, FRepr rhs, BinaryOperator op):
        cdef FRepr comp1, comp2
        if len(self) != len(rhs):
            raise ValueError('I need both objects to contain the same number of components.')
        return FComponents([comp1.hadamard(comp2, op) for comp1, comp2 in zip(self.components, (<FComponents?>rhs).components)])

    cpdef FRepr elementwise(self, UnaryOperator op):
        cdef FRepr frepr
        return FComponents([frepr.elementwise(op) for frepr in self.components])

    cpdef FRepr elementwise_b(self, weight_t rhs, BinaryOperator op):
        cdef FRepr frepr
        return FComponents([frepr.elementwise_b(rhs, op) for frepr in self.components])

    cpdef FRepr power(self, weight_t power, Semiring semiring):
        cdef FRepr frepr
        return FComponents([frepr.power(power, semiring) for frepr in self.components])

    cpdef FRepr densify(self):
        cdef list dense = []
        for comp in self.components:
            dense.extend(comp.densify())
        return FVec(dense)

    def __getstate__(self):
        return {'components': self.components}

    def __setstate__(self, d):
        self.components = list(d['components'])

    def __len__(self):
        return len(self.components)

    def __iter__(self):
        return iter(self.components)

    def __str__(self):
        cdef FRepr comp
        return ' | '.join(str(comp) for comp in self.components)

    def __repr__(self):
        return 'FComponents(%r)' % self.components
