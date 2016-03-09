from grasp.alg.slicevars cimport SliceVariables
from grasp.alg.value cimport ValueFunction

from grasp.formal.hg cimport Hypergraph
from grasp.ptypes cimport weight_t, id_t
from grasp.formal.topsort cimport TopSortTable

from grasp.semiring._semiring cimport Semiring
from grasp.cfg.rule cimport Rule
from grasp.cfg.symbol cimport Symbol, Terminal


cdef class SliceReturn:

    cdef public:
        Hypergraph S
        ValueFunction l
        ValueFunction u
        list S2D_edge_mapping
        tuple d0_in_S
        weight_t mean_constrained, mean_unconstrained

    cdef tuple back_to_D(self, tuple d_in_S)


cdef SliceReturn slice_forest(Hypergraph D,
                        TopSortTable tsort,
                        tuple d0,
                        SliceVariables slicevars,
                        Semiring semiring,
                        Rule dead_rule,
                        Symbol dead_terminal=?)