from grasp.alg.slicevars cimport SliceVariables
from grasp.alg.value cimport ValueFunction, EdgeWeight, LookupFunction

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


cdef class SliceReturn:
    """
    Return type of the slice method.
    Organises all the objects relevant to a slice, such as, residual values and local values.
    """

    def __init__(self, Hypergraph S,
                 ValueFunction l,
                 ValueFunction u,
                 list S2D_edge_mapping,
                 tuple d0_in_S,
                 weight_t mean_constrained,
                 weight_t mean_unconstrained):
        self.S = S
        self.l = l
        self.u = u
        self.S2D_edge_mapping = S2D_edge_mapping
        self.d0_in_S = d0_in_S
        self.mean_constrained = mean_constrained
        self.mean_unconstrained = mean_unconstrained

    cdef tuple back_to_D(self, tuple d_in_S):
        cdef id_t e
        return tuple([self.S2D_edge_mapping[e] for e in d_in_S])


cdef SliceReturn slice_forest(Hypergraph D,
                        TopSortTable tsort,
                        tuple d0,
                        SliceVariables slicevars,
                        Semiring semiring,
                        Rule dead_rule,
                        Symbol dead_terminal=Terminal('<null>')):
    """
    Prune a forest and compute a new weight function for edges.

    :param D: weighted forest
    :param tsort: forest's top-sort table
    :param slicevars: an instance of SliceVariable
    :param semiring: the appropriate semiring
    :param dead_terminal: a Terminal symbol which represents a dead-end.
    :return: pruned forest and weight table mapping edges to weights.
    """
    cdef:
        Hypergraph S = Hypergraph()  # the slice
        set discovered = set([tsort.root()])
        id_t selected
        id_t cell, head
        id_t input_edge, output_edge, dead_edge, e
        list n_uniform = []
        list n_exponential = []
        list weight_table = []
        list S2D_edge_map = []
        list translate = []
        SliceReturn slice_return  # return container

    d0_in_D = deque(d0)
    d0_in_S = deque([])


    for level in tsort.iterlevels(reverse=True):  # we go top-down level by level
        for head in itertools.chain(*level):  # flatten the buckets

            if head not in discovered:  # symbols not yet discovered have been pruned
                continue

            cell = head
            selected = 0

            # we sort the incoming edges in order to find out which ones are not in the slice
            for input_edge in sorted(D.iterbs(head), key=lambda e: semiring.heapify(D.weight(e))):
                #u = semiring.from_real(slicevars[cell])
                if slicevars.is_inside(cell, semiring.as_real(D.weight(input_edge))):  # inside slice
                    # copy the edge to the output
                    output_edge = S.add_xedge(D.label(head),
                                      tuple([D.label(n) for n in D.tail(input_edge)]),
                                      D.weight(input_edge),
                                      D.rule(input_edge),
                                      D.is_glue(input_edge))
                    S2D_edge_map.append(input_edge)

                    if d0_in_D and d0_in_D[0] == input_edge:
                        d0_in_D.popleft()
                        d0_in_S.append(output_edge)

                    # the contribution of this edge to p(d|u)
                    assert len(weight_table) == output_edge, 'An edge got an id out of order'
                    weight_table.append(slicevars.pdf_semiring(cell, D.weight(input_edge), semiring))

                    # update discovered heads with the nonterminal children
                    for n in D.tail(input_edge):
                        if D.is_nonterminal(n):
                            discovered.add(n)
                    selected += 1
                else:  # this edge and the remaining are pruned (because we sorted the edges by score)
                    #logging.info('pruning: %s vs %s', p, slicevars[cell])
                    break
            # update information about slice variables
            if slicevars.has_conditions(cell):
                n_uniform.append(float(n)/D.n_incoming(head))
            else:
                n_exponential.append(float(n)/D.n_incoming(head))
            #logging.info('cell=%s slice=%d/%d conditioning=%s', cell, n, len(incoming), slicevars.has_conditions(cell))
            if selected == 0:  # this is a dead-end, instead of pruning bottom-up, we add an edge with 0 weight for simplicity
                # create a dead-end edge for this node with zero weight
                # this simplifies bottom-up pruning
                dead_edge = S.add_xedge(D.label(head),
                                         tuple([dead_terminal]),
                                         semiring.zero,
                                         dead_rule,
                                         glue=False)
                S2D_edge_map.append(-1)
                assert len(weight_table) == dead_edge, 'A dead edge got an id out of order'
                weight_table.append(semiring.zero)

    if d0_in_D:
        raise ValueError('The slice is missing edges that it should contain by construction: %s' % str(d0_in_D))

    #logging.info('Pruning: cells=%s/%s in=%s out=%s', pruned, total_cells, np.array(_in).mean(), np.array(_out).mean())
    # EdgeWeight(l_slice)
    slice_return = SliceReturn(S,
                               EdgeWeight(S),
                               LookupFunction(np.array(weight_table, dtype=ptypes.weight)),
                               S2D_edge_map,
                               tuple(d0_in_S),
                               np.mean(n_uniform),
                               np.mean(n_exponential))

    return slice_return