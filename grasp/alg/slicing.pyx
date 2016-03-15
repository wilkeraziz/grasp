from grasp.alg.slicevars cimport SliceVariables
from grasp.formal.wfunc cimport WeightFunction, HypergraphLookupFunction, TableLookupFunction

from grasp.formal.hg cimport Hypergraph
from grasp.formal.topsort cimport TopSortTable

from grasp.semiring._semiring cimport Semiring
from grasp.cfg.rule cimport Rule
from grasp.cfg.symbol cimport Symbol, Terminal
from grasp.ptypes cimport weight_t, id_t
import grasp.ptypes as ptypes
cimport numpy as np
import numpy as np
import itertools
from collections import deque

import logging

cdef class SliceReturn:
    """
    Return type of the slice method.
    Organises all the objects relevant to a slice, such as, residual values and local values.
    """

    def __init__(self, Hypergraph S,
                 WeightFunction local,
                 WeightFunction residual,
                 list S2D_edge_mapping,
                 tuple d0_in_S,
                 weight_t mean_constrained,
                 weight_t mean_unconstrained):
        self.S = S
        self.local = local
        self.residual = residual
        self.S2D_edge_mapping = S2D_edge_mapping
        self.d0_in_S = d0_in_S
        self.mean_constrained = mean_constrained
        self.mean_unconstrained = mean_unconstrained

    cdef tuple back_to_D(self, tuple d_in_S):
        cdef id_t e
        return tuple([self.S2D_edge_mapping[e] for e in d_in_S])

cdef SliceReturn slice_forest(Hypergraph D,
                              WeightFunction omega,
                              TopSortTable tsort,
                              tuple d0,
                              SliceVariables slicevars,
                              Semiring semiring,
                              Rule dead_rule):
    """
    Prune a forest and compute a new weight function for edges.

    :param D: weighted forest
    :param omega: a value function over edges in D
    :param tsort: forest's top-sort table
    :param slicevars: an instance of SliceVariable
    :param semiring: the appropriate semiring
    :param dead_terminal: a Terminal symbol which represents a dead-end.
    :return: pruned forest (edges are weighted by omega) and weight table mapping edges to u(e) weights.
    """
    cdef:
        Hypergraph S = Hypergraph()  # the slice
        set discovered = set([tsort.root()])
        id_t selected
        id_t head
        Symbol cell
        id_t input_edge, output_edge, dead_edge, e
        weight_t w_e
        list n_uniform = []
        list n_exponential = []
        list residuals = []
        list S2D_edge_map = []
        list translate = []
        SliceReturn slice_return  # return container
        frozenset D_edges_in_d0 = frozenset(d0)
        dict d0transform = {}


    # TODO: investigate this code... something strange is happening with the conditions

    logging.debug('Slicing the distribution conditioned on d0=%s', ' '.join(['{0}/{1}/{2}'.format(D.head(e), e, semiring.as_real(omega.value(e))) for e in d0]))
    cdef list check = []
    cdef list assignments = []
    for e in d0:
        cell = D.label(D.head(e))
        check.append(slicevars.is_inside(cell, semiring.as_real(omega(e))))
        assignments.append(slicevars.get_assignment(cell).u)
    logging.debug('Check: %s', check)
    assert all(check), 'Something strange here: checks=%s\nassignments=%s' % (check, assignments)

    for level in tsort.iterlevels(reverse=True):  # we go top-down level by level
        for head in level:
            if D.is_terminal(head):
                continue
            cell = D.label(head)
            if head not in discovered:  # symbols not yet discovered have been pruned
                #logging.debug(' Pruning: head=%s cell=%s', head, cell)
                continue

            #logging.debug(' Inspecting edges for head=%s cell=%s constrained=%s assignment=%s', head, cell, slicevars.has_conditions(cell), slicevars.get_assignment(cell).u)
            selected = 0

            # we sort the incoming edges in order to find out which ones are not in the slice
            for input_edge in sorted(D.iterbs(head), key=lambda e: semiring.heapify(omega.value(e))):
                #u = semiring.from_real(slicevars[cell])
                w_e = omega.value(input_edge)
                if slicevars.is_inside(cell, semiring.as_real(w_e)):  # inside slice
                    # copy the edge to the output
                    output_edge = S.add_xedge(D.label(head),
                                      tuple([D.label(n) for n in D.tail(input_edge)]),
                                      w_e,
                                      D.rule(input_edge),
                                      D.is_glue(input_edge))
                    S2D_edge_map.append(input_edge)

                    # map the edge in case we need it to reconstruct d0 in the slice
                    if input_edge in D_edges_in_d0:
                        d0transform[input_edge] = output_edge

                    # the contribution of this edge to p(d|u)
                    assert len(residuals) == output_edge, 'An edge got an id out of order'
                    residuals.append(slicevars.pdf_semiring(cell, w_e, semiring))

                    # update discovered heads with the nonterminal children
                    for n in D.tail(input_edge):
                        if D.is_nonterminal(n):
                            discovered.add(n)
                    selected += 1
                    #logging.debug('  selected: input-edge=%d output-edge=%d', input_edge, output_edge)
                else:  # this edge and the remaining are pruned (because we sorted the edges by score)
                    #logging.debug('pruning: %s vs %s', p, slicevars[cell])
                    assert input_edge not in D_edges_in_d0, 'I cannot prune an edge in d0.'
                    #logging.debug('  pruned: input-edge=%d weight=%s', input_edge, semiring.as_real(w_e))
                    break
            # update information about slice variables
            if slicevars.has_conditions(cell):
                n_uniform.append(float(selected)/D.n_incoming(head))
            else:
                n_exponential.append(float(selected)/D.n_incoming(head))
            #logging.debug('cell=%s slice=%d/%d conditioning=%s', cell, n, len(incoming), slicevars.has_conditions(cell))
            if selected == 0:  # this is a dead-end, instead of pruning bottom-up, we add an edge with 0 weight for simplicity
                # create a dead-end edge for this node with zero weight
                # this simplifies bottom-up pruning
                dead_edge = S.add_xedge(D.label(head),
                                         tuple([dead_rule.rhs[0]]),
                                         semiring.zero,
                                         dead_rule,
                                         glue=False)
                S2D_edge_map.append(-1)
                assert len(residuals) == dead_edge, 'A dead edge got an id out of order'
                residuals.append(semiring.zero)

    if len(d0transform) != len(d0):
        raise ValueError('The slice is missing edges that it should contain by construction: %s' % str(D_edges_in_d0 - frozenset(d0transform.keys())))

    #logging.debug('Pruning: cells=%s/%s in=%s out=%s', pruned, total_cells, np.array(_in).mean(), np.array(_out).mean())

    slice_return = SliceReturn(S,
                               HypergraphLookupFunction(S),
                               TableLookupFunction(np.array(residuals, dtype=ptypes.weight)),
                               S2D_edge_map,
                               tuple([d0transform[e] for e in d0]),
                               np.mean(n_uniform),
                               np.mean(n_exponential))

    logging.debug('D: nodes=%d edges=%d', D.n_nodes(), D.n_edges())
    logging.debug('S: nodes=%d edges=%d', S.n_nodes(), S.n_edges())
    logging.debug('Constrained=%f Unconstrained=%f', slice_return.mean_constrained, slice_return.mean_unconstrained)

    return slice_return