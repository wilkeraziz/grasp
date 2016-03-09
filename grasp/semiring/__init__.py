"""
A semiring must define the following

    one => the multiplicative identity
    zero => the additive identity (and multiplicative annihilator)
    plus => addition
    times => multiplication

The interface of operators `plus` and `times` should be compatible with numpy operators.
That is, they can be applied to a pair of elements, or, a list of elements can be reduced through op.reduce.

A semiring may define the following

    divide => division
    as_real => return a Real number
    from_real => constructs from a Real number
    gt => comparison '>'
    heapify => return a value compatible with the logic of a heap (smaller first)
    convert(x, other_semiring) => convert x from another semiring into this one
    choice(items, values) => pick an item using the plus operator


:Authors: - Wilker Aziz
"""

#from .boolean import Boolean
#from .counting import Counting
#from .inside import SumTimes, Prob
#from .viterbi import MaxTimes

from . import _semiring

prob = _semiring.Prob()
viterbi = _semiring.Viterbi()
inside = _semiring.LogProb()
logprob = inside
