from grasp.alg.slicevars cimport SliceVariables
from grasp.formal.wfunc cimport WeightFunction
from grasp.formal.hg cimport Hypergraph
from grasp.formal.topsort cimport TopSortTable
from grasp.semiring._semiring cimport Semiring
from grasp.cfg.rule cimport Rule
from grasp.ptypes cimport weight_t, id_t, boolean_t, status_t


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


cdef void visit_edge(Hypergraph forest, SliceVariables slicevars, WeightFunction omega, Semiring semiring,
                     id_t edge,
                     status_t[::1] node_colour, status_t[::1] edge_colour,
                     boolean_t[::1] selected_nodes, boolean_t[::1] selected_edges,
                     weight_t[::1] new_weights)


cdef void visit_node(Hypergraph forest, SliceVariables slicevars, WeightFunction omega, Semiring semiring,
                     id_t node,
                     status_t[::1] node_colour,
                     status_t[::1] edge_colour,
                     boolean_t[::1] selected_nodes,
                     boolean_t[::1] selected_edges,
                     weight_t[::1] new_weights)


cdef SliceReturn slice_forest(Hypergraph forest,
                              WeightFunction omega,
                              TopSortTable tsort,
                              tuple d0,
                              SliceVariables slicevars,
                              Semiring semiring,
                              Rule dead_rule)