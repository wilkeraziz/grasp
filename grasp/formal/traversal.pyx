from grasp.formal.hg cimport Hypergraph
from grasp.ptypes cimport id_t
from grasp.cfg.symbol cimport Symbol


cdef _TDLR_visit_edge(Hypergraph forest, id_t edge, dict edges_by_head, list order, bint terminal_only):
    """
    Function meant for top-down left-to-right traversals.
    :param forest:
    :param edge:
    :param edges_by_head:
    :param order:
    """
    if not terminal_only:
        order.append(forest.head(edge))  # visit head
    for c in forest.tail(edge):
        if forest.is_terminal(c):
            order.append(c) # visit terminal child
        else:
            _TDLR_visit_edge(forest, edges_by_head[c], edges_by_head, order, terminal_only)  # visit edge from nonterminal child


cpdef tuple top_down_left_right(Hypergraph forest, tuple acyclic_derivation, bint terminal_only=False):
    """
    Return a top-down left-to-right traversal of nodes in a derivation.
    :param forest: a hypergraph
    :param derivation: an acyclic sequence of edges where the first edge starts the derivation.
    :return: sequence of nodes
    """
    cdef id_t e, c
    cdef edges_by_head = {forest.head(e): e for e in acyclic_derivation}
    cdef list order = []
    if acyclic_derivation:
        _TDLR_visit_edge(forest, acyclic_derivation[0], edges_by_head, order, terminal_only)
    return tuple(order)


cdef _bracketing_visit_edge(Hypergraph forest, id_t edge, dict edges_by_head, list order):
    """
    Function meant for bracketing traversals.
    :param forest:
    :param edge:
    :param edges_by_head:
    :param order:
    """

    order.append(('B', forest.label(forest.head(edge))))  # visit head
    for c in forest.tail(edge):
        if forest.is_terminal(c):
            order.append(('I', forest.label(c))) # visit terminal child
        else:
            _bracketing_visit_edge(forest, edges_by_head[c], edges_by_head, order)  # visit edge from nonterminal child
    order.append(('E', forest.label(forest.head(edge))))


cpdef tuple bracketing(Hypergraph forest, tuple acyclic_derivation):
    """
    Return a bracketed sequence representing the derivation.
    :param forest: a hypergraph
    :param derivation: an acyclic sequence of edges where the first edge starts the derivation.
    :return: bracketed sequence where each element is a pair (type, symbol) where type is one of B/I/E
        standing for begin-, inside-, and end- of span
    """
    cdef id_t e, c
    cdef edges_by_head = {forest.head(e): e for e in acyclic_derivation}
    cdef list order = []
    if acyclic_derivation:
        _bracketing_visit_edge(forest, acyclic_derivation[0], edges_by_head, order)
    return tuple(order)


cpdef str bracketed_string(Hypergraph forest, tuple acyclic_derivation,
                           str bos='(', str ios='', str eos=')',
                           bint label_bos=True, bint label_eos=False,
                           str bos_position='l', str ios_position='l', str eos_position='r'):
    """
    Return a bracketed string representing a derivation.
    :param forest: the hypergraph where the derivation comes from
    :param acyclic_derivation: a sequence of edges representing an acyclic derivation from the forest.
        Importantly, the first edge of the derivation must rewrite the root of the derivation.
    :param bos: begin of span symbol (e.g. '(', 'B-'), an empty string will cause the node to be skipped
    :param ios: inside of span symbol (e.g. '', 'I-')
    :param eos: end of span symbol (e.g. ')', 'E-'), an empty string will cause the node to be skipped
    :param label_bos: whether to label the BOS span or not
    :param label_eos: whether to label the EOS span or not
    :param bos_position: where to position the BOS label wrt symbol (options: 'l' for left, 'r' for right)
    :param ios_position: where to position the IOS label wrt symbol (options: 'l', 'r')
    :param eos_position: where to position the EOS label wrt symbol (options: 'l', 'r')
    :return: a bracketed string
    """
    cdef tuple spans = bracketing(forest, acyclic_derivation)
    cdef list tokens = []
    cdef str b
    cdef Symbol s
    for (b, s) in spans:
        if b == 'B':
            if bos:
                if label_bos:  # we are labelling the bracket
                    if bos_position == 'l':
                        tokens.append('{0}{1} '.format(bos, s))
                    else:
                        tokens.append('{0}{1} '.format(s, bos))
                else:  # we are not labelling the bracket
                    tokens.append(bos)  # we are not labelling the bracket
        elif b == 'E':
            if eos:
                if label_eos:  # we are labelling the bracket
                    if eos_position == 'l':
                        tokens.append('{0}{1} '.format(eos, s))
                    else:
                        tokens.append('{0}{1} '.format(s, eos))
                else:  # we are not labelling the bracket
                    tokens.append(eos)
        else:  # thus b == 'I'
            if ios:  # we are marking leaves
                if ios_position == 'l':
                    tokens.append('{0}{1} '.format(ios, s))
                else:
                    tokens.append('{0}{1} '.format(s, ios))
            else:  # we are not marking leaves
                tokens.append(str(s))
    return ' '.join(tokens)


cpdef str yield_string(Hypergraph forest, tuple acyclic_derivation):
    return bracketed_string(forest, acyclic_derivation, bos='', ios='', eos='')