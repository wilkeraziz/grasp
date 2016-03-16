from grasp.ptypes cimport weight_t, id_t
from grasp.formal.hg cimport Hypergraph
from grasp.formal.topsort cimport AcyclicTopSortTable
from grasp.semiring._semiring cimport Semiring
from grasp.formal.wfunc cimport WeightFunction
from grasp.scoring.model cimport Model
from grasp.scoring.frepr cimport FComponents
from grasp.alg.value cimport acyclic_value_recursion, acyclic_reversed_value_recursion
from grasp.semiring._semiring cimport Prob
from grasp.semiring.operator cimport FixedLHS
import logging


cpdef tuple expected_components(Hypergraph forest,
                                WeightFunction omega,
                                AcyclicTopSortTable tsort,
                                Semiring semiring,
                                Model model,
                                list components):
    """
    This is the Inside-Outside speedup proposed by Li and Eisner (2009) [section 4.2 and figure 4].

        Basically, we compute <phi(d)>_p = \sum_e phi(e) outside(head(e)) * inside(tail(e))

    :param forest:
    :param omega:
    :param tsort:
    :param semiring:
    :param model:
    :param components:
    :return:
    """
    # expectations are computed in an additive semiring
    cdef Semiring expecsemiring = Prob()

    # inside computation
    cdef weight_t[::1] inside = acyclic_value_recursion(forest, tsort, semiring, omega=omega)
    # outside computation
    cdef weight_t[::1] outside = acyclic_reversed_value_recursion(forest, tsort, semiring, inside, omega=omega)
    # expectation computation
    cdef:
        weight_t w
        id_t e, child
        FComponents wff
        FComponents mean = model.constant(expecsemiring.zero)
    for e in range(forest.n_edges()):
        # 1. first we compute the "exclusive weight" of e in the given semiring
        w = outside[forest.head(e)]
        # we then incorporate the inside weight of each child node
        for child in forest.tail(e):
            w = semiring.times(w, inside[child])
        # 2. then we incorporate the weighted features in the expectation semiring
        wff = (<FComponents>components[e]).elementwise(FixedLHS(semiring.as_real(w), expecsemiring.times))
        mean = mean.hadamard(wff, expecsemiring.plus)

    cdef weight_t Z = inside[tsort.root()]
    # normalise features by real(Z)
    cdef weight_t factor = semiring.as_real(semiring.times.inverse(Z))
    mean = mean.elementwise(FixedLHS(factor, expecsemiring.times))

    return Z, mean