from grasp.formal.wfunc cimport WeightFunction
from grasp.formal.hg cimport Hypergraph
from grasp.formal.topsort cimport TopSortTable
from grasp.semiring._semiring cimport Semiring
from grasp.cfg.rule cimport Rule
from grasp.ptypes cimport boolean_t


cdef class SliceReturn:
    """
    Return type of the slice method.
    Organises all the objects relevant to a slice, such as, residual values and local values.
    """

    cdef public:
        Hypergraph forest
        TopSortTable tsort
        tuple d0
        boolean_t[::1] selected_nodes
        boolean_t[::1] selected_edges
        WeightFunction local
        WeightFunction residual

    cdef tuple back_to_D(self, tuple d_in_S)


cdef SliceReturn slice_forest(Hypergraph forest,
                              WeightFunction omega,
                              WeightFunction slice_check,
                              TopSortTable tsort,
                              tuple d0,
                              Semiring semiring,
                              Rule dead_rule)