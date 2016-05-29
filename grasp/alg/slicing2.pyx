from grasp.alg.slicevars cimport SliceVariables
from grasp.formal.wfunc cimport WeightFunction
from grasp.formal.wfunc cimport TableLookupFunction

from grasp.formal.hg cimport Hypergraph
from grasp.formal.topsort cimport TopSortTable

from grasp.semiring._semiring cimport Semiring
from grasp.cfg.rule cimport Rule
from grasp.cfg.symbol cimport Symbol
from grasp.ptypes cimport weight_t, id_t, boolean_t, status_t
import grasp.ptypes as ptypes
import numpy as np
cimport numpy as np
cimport cython

import logging


cdef class SliceReturn:
    """
    Return type of the slice method.
    Organises all the objects relevant to a slice, such as, residual values and local values.
    """

    def __init__(self, Hypergraph forest,
                 TopSortTable tsort,
                 tuple d0,
                 boolean_t[::1] selected_nodes,
                 boolean_t[::1] selected_edges,
                 WeightFunction local,
                 WeightFunction residual):
        self.forest = forest
        self.tsort = tsort
        self.d0 = d0
        self.local = local
        self.residual = residual
        self.selected_nodes = selected_nodes
        self.selected_edges = selected_edges

    cdef tuple back_to_D(self, tuple d_in_S):
        return d_in_S


cdef void visit_edge(Hypergraph forest, SliceVariables slicevars, WeightFunction omega, Semiring semiring,
                     id_t edge,
                     status_t[::1] node_colour, status_t[::1] edge_colour,
                     boolean_t[::1] selected_nodes, boolean_t[::1] selected_edges,
                     weight_t[::1] new_weights):

    if edge_colour[edge] > 0:  # visiting (1) or visited (2)
        return

    # We are visiting this edge
    edge_colour[edge] = 1

    # Perform the slice check
    cdef:
        Symbol cell = forest.label(forest.head(edge))
        weight_t w_e = omega.value(edge)

    if not slicevars.is_inside(cell, semiring.as_real(w_e)):  # slice check
        # we have finished visiting the edge
        # and it has not been selected, because it did not pass the slice check
        edge_colour[edge] = 2
        return

    # at this point we are sure the edge passed the slice check
    # but we do not know about its children
    cdef:
        id_t child
        weight_t value = semiring.one
    for child in forest.tail(edge):
        # we do not visit nodes that have been visited (2) or are being visited (1)
        if node_colour[child] == 0:  # only those that have not been visited (0)
            # visit the node
            visit_node(forest, slicevars, omega, semiring, child,
                       node_colour, edge_colour,
                       selected_nodes, selected_edges, new_weights)
        # if the node has been visited (2), but not selected
        if node_colour[child] == 2 and not selected_nodes[child]:
            # then even though this edge passed the slice check,
            # it will be excluded because some of its children didn't
            edge_colour[edge] = 2
            return
    # at this point we know that all children are in the slice, thus so is the edge
    selected_edges[edge] = True
    # TODO: change the name of this method, this is not the pdf value
    # this is actually 1.0/phi(u_s)
    new_weights[edge] = slicevars.pdf_semiring(cell, w_e, semiring)
    # and we finish visiting the edge
    edge_colour[edge] = 2


cdef void visit_node(Hypergraph forest, SliceVariables slicevars, WeightFunction omega, Semiring semiring,
                     id_t node,
                     status_t[::1] node_colour,
                     status_t[::1] edge_colour,
                     boolean_t[::1] selected_nodes,
                     boolean_t[::1] selected_edges,
                     weight_t[::1] new_weights):

    if node_colour[node] > 0:  # visiting (1) or visited (2)
        return

    # We are visiting this node
    node_colour[node] = 1

    # A terminal node is always in the slice
    if forest.is_terminal(node):
        node_colour[node] = 2
        selected_nodes[node] = True
        return

    # A nonterminal node is in the slice only if at least one of its incoming edges is in the slice
    cdef:
        size_t selected = 0
        id_t edge
    for edge in forest.iterbs(node):
        visit_edge(forest, slicevars, omega, semiring, edge,
                   node_colour, edge_colour,
                   selected_nodes, selected_edges,
                   new_weights)
        if selected_edges[edge]:
            selected += 1
    if selected > 0:  # now we know whether this node is in the slice
        selected_nodes[node] = True
    # and we are done visiting the node
    node_colour[node] = 2


cdef SliceReturn slice_forest(Hypergraph forest,
                              WeightFunction omega,
                              TopSortTable tsort,
                              tuple d0,
                              SliceVariables slicevars,
                              Semiring semiring,
                              Rule dead_rule):
    """
    Prune a forest and compute a new weight function for edges.

    :param forest: weighted forest
    :param omega: a value function over edges in D
    :param tsort: forest's top-sort table
    :param slicevars: an instance of SliceVariable
    :param semiring: the appropriate semiring
    :param dead_terminal: a Terminal symbol which represents a dead-end.
    :return: pruned forest (edges are weighted by omega) and weight table mapping edges to u(e) weights.
    """
    cdef:
        id_t parent, e
        SliceReturn slice_return  # return container
        # nodes and edges have not yet been visited
        status_t[::1] node_colour = np.zeros(forest.n_nodes(), dtype=ptypes.status)
        status_t[::1] edge_colour = np.zeros(forest.n_edges(), dtype=ptypes.status)
        # we assume all nodes and edges have been pruned
        boolean_t[::1] selected_nodes = np.zeros(forest.n_nodes(), dtype=ptypes.boolean)
        boolean_t[::1] selected_edges = np.zeros(forest.n_edges(), dtype=ptypes.boolean)
        # we assume all edges to have 0 weight in the slice
        weight_t[::1] new_weights = semiring.zeros(forest.n_edges())

    # we go top-down visiting nodes and edges
    # selecting them and computing new weights for edges in the slice
    for parent in tsort.iternodes(reverse=True):
        visit_node(forest, slicevars, omega, semiring, parent,
                   node_colour, edge_colour,
                   selected_nodes, selected_edges,
                   new_weights)

    assert all([selected_edges[e] for e in d0]), 'I expected all edges of d0 to be in the slice'

    return SliceReturn(forest,tsort, d0, selected_nodes, selected_edges, omega, TableLookupFunction(new_weights))
