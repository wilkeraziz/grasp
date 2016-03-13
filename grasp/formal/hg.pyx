from grasp.cfg.symbol cimport Terminal
from grasp.ptypes cimport id_t, weight_t
from .hg cimport tail_t
from cpython.object cimport Py_EQ, Py_NE

cdef class Node:
    """
    A container for a label (a Symbol object).
    """

    def __init__(self, Symbol label):
        self.label = label

    def __getstate__(self):
        return {'label': self.label}

    def __setstate__(self, d):
        self.label = d['label']

    def __str__(self):
        return 'label={0}'.format(repr(self.label))

    def __repr__(self):
        return repr(self.label)


cdef class Edge:
    """
    Connects a sequence of children nodes (the tail) under a head node with a weight.
    """

    def __init__(self, id_t head, tail_t tail, weight_t weight):
        self.head = head
        self.tail = tail
        self.weight = weight

    def __str__(self):
        return 'head={0} tail=({1}) weight={2}'.format(self.head,
                                                       ' '.join(str(x) for x in self.tail),
                                                       self.weight)

    def __repr__(self):
        return 'Edge(%r, %r, %r)' % (self.head, self.tail, self.weight)

    def __getstate__(self):
        return {'head': self.head, 'tail': self.tail, 'weight': self.weight}

    def __setstate__(self, d):
        self.head = d['head']
        self.tail = d['tail']
        self.weight = d['weight']


cdef class Hypergraph:
    """
    A backward-hypergraph (Gallo, 1992).
    """

    def __init__(self):
        self._nodes = []
        self._edges = []
        self._bs = []
        self._fs = []
        self._deps = []
        self._symbol_map = {}
        self._symbols = []
        self._rules = []
        self._glue = set()

    def __getstate__(self):
        return {'nodes': self._nodes,
                'edges': self._edges,
                'bs': self._bs,
                'fs': self._fs,
                'deps': self._deps,
                'symbol_map': self._symbol_map,
                'symbols': self._symbols,
                'rules': self._rules,
                'glue': self._glue}

    def __setstate__(self, d):
        self._nodes = list(d['nodes'])
        self._edges = list(d['edges'])
        self._bs = list(d['bs'])
        self._fs = list(d['fs'])
        self._deps = list(d['deps'])
        self._symbol_map = dict(d['symbol_map'])
        self._symbols = list(d['symbols'])
        self._rules = list(d['rules'])
        self._glue = set(d['glue'])

    def __str__(self):
        return 'nodes={0}\n{1}\nedges={2}\n{3}'.format(len(self._nodes),
                                                       '\n'.join(repr(n) for n in self._nodes),
                                                       len(self._edges),
                                                       '\n'.join('{0} rule={1}'.format(e, r)
                                                                 for e, r in zip(self._edges, self._rules)))

    def __len__(self):
        return self.n_edges()

    cdef _add_node(self, Node node):
        """
        Add a node to the underlying hypergraph.
        :param node: a Node
        :return: index of the node
        """
        cdef id_t n = len(self._nodes)
        self._nodes.append(node)
        self._bs.append([])
        self._fs.append(set())
        self._deps.append(set())
        return n

    cdef _add_edge(self, Edge edge):
        """
        Add an edge to the underlying hypergraph.
        :param edge: an Edge
        :return: index of the edge
        """
        cdef id_t e = len(self._edges)
        self._edges.append(edge)
        self._bs[edge.head].append(e)
        cdef id_t n
        for n in edge.tail:
            (<set>self._fs[n]).add(e)
            (<set>self._deps[edge.head]).add(n)
        return e

    cpdef id_t add_node(self, Symbol label):
        """
        Map a symbol to a node (constructing the node only if necessary).
        :param label: a Symbol
        :return: index of the corresponding node
        """
        cdef id_t n = self._symbol_map.get(label, -1)
        if n == -1:
            n = self._add_node(Node(label))
            self._symbol_map[label] = n
        return n

    cpdef id_t add_xedge(self, Symbol lhs, tuple rhs, weight_t weight, Rule rule, bint glue=False):
        cdef id_t head = self.add_node(lhs)
        cdef tuple tail = tuple([self.add_node(sym) for sym in rhs])
        cdef id_t e = self._add_edge(Edge(head, tail, weight))
        self._rules.append(rule)
        if glue:
            self._glue.add(e)
        return e

    #cpdef id_t add_edge(self, Rule rule, bint glue=False):
    #    """
    #    Map a rule to an edge (always creates a new edge).
    #    :param rule: a Rule
    #    :param glue: whether the rule is constrained to rewriting from initial states only.
    #    :return: index of the corresponding edge
    #    """
    #    cdef id_t head = self.add_node(rule.lhs)
    #    cdef tuple tail = tuple([self.add_node(sym) for sym in rule.rhs])
    #    cdef id_t e = self._add_edge(Edge(head, tail, rule.weight))
    #    self._rules.append(rule)
    #    if glue:
    #        self._glue.add(e)
    #    return e

    cpdef id_t fetch(self, Symbol sym, id_t default=-1):
        """Return the node associated with a given symbol or a default value."""
        return self._symbol_map.get(sym, default)

    cpdef id_t head(self, id_t e):
        """Return the id of the edge's head node"""
        return self._edges[e].head

    cpdef tail_t tail(self, id_t e):
        """Return the edge's tail nodes"""
        return self._edges[e].tail

    cpdef weight_t weight(self, id_t e):
        """Return the edge's weight"""
        return self._edges[e].weight

    cpdef size_t arity(self, id_t e):
        """Return the edge's arity"""
        return len(self._edges[e].tail)

    cpdef id_t child(self, id_t e, id_t i):
        return self._edges[e].tail[i]

    cpdef bint is_terminal(self, id_t n):
        """Whether the node is labelled with a terminal symbol"""
        return isinstance(self._nodes[n].label, Terminal)

    cpdef bint is_nonterminal(self, id_t n):
        """Whether the node is labelled with a nonterminal symbol"""
        return not self.is_terminal(n)

    cpdef bint is_source(self, id_t head):
        """Whether the node has an empty backward-star (no incoming edges)"""
        return len(self._bs[head]) == 0

    cpdef bint is_glue(self, id_t e):
        return e in self._glue

    cpdef Symbol label(self, id_t n):
        """The label associated with a node"""
        return self._nodes[n].label

    cpdef Rule rule(self, id_t e):
        """The rule associated with an edge"""
        return self._rules[e]

    #cpdef update(self, rules, bint glue=False):
    #    """
    #    Creates nodes and edges corresponding to a given CFG.
    #    :param rules: an iterable over Rule objects.
    #    :param glue: whether these rules come from a glue grammar
    #    """
    #    cdef Rule rule
    #    for rule in rules:
    #        self.add_edge(rule, glue=glue)

    cpdef size_t n_nodes(self):
        """Number of nodes"""
        return len(self._nodes)

    cpdef size_t n_edges(self):
        """Number of edges"""
        return len(self._edges)

    cpdef size_t n_incoming(self, id_t n):
        return len(self._bs[n])

    cpdef iterglue(self):
        """Iterator over indices of glue edges (in no particular order)."""
        return iter(self._glue)

    cpdef iterbs(self, id_t head):
        """An iterator over the node's incoming edges"""
        return iter(self._bs[head])

    cpdef id_t bs_i(self, id_t head, size_t i):
        """Return the ith edge in BS(head)"""
        return self._bs[head][i]

    cpdef iterfs(self, id_t node):
        """An iterator over the node's outgoing edges (in arbitrary order)."""
        return iter(self._fs[node])

    cpdef iterdeps(self, id_t node):
        """An iterator over the node's outgoing edges (in arbitrary order)."""
        return iter(self._deps[node])

    cpdef bint self_depends(self, id_t node):
        return node in <set>self._deps[node]


cdef class Derivation:

    def __init__(self, Hypergraph derivation):
        self._hg = derivation
        self._rules = None
        self._weights = None

    def __getstate__(self):
        return {'hg': self._hg,
                'rules': self._rules,
                'weights': self._weights}

    def __setstate__(self, d):
        self._hg = d['hg']
        self._rules = d['rules']
        self._weights = d['weights']

    cpdef Hypergraph hg(self):
        return self._hg

    cpdef tuple edges(self):
        return tuple(range(self._hg.n_edges()))

    cpdef tuple weights(self):
        cdef id_t e
        if self._weights is None:
            self._weights = tuple([self._hg.weight(e) for e in range(self._hg.n_edges())])
        return self._weights

    cpdef tuple rules(self):
        cdef id_t e
        if self._rules is None:
            self._rules = tuple([self._hg.rule(e) for e in range(self._hg.n_edges())])
        return self._rules

    cpdef iteritems(self):
        return zip(self._rules, self._weights)

    def __hash__(self):
        return hash(self.rules())

    def __richcmp__(Derivation x, Derivation y, int opt):
        if opt == Py_EQ:
            return x.rules() == y.rules()
        elif opt == Py_NE:
            return x.rules() != y.rules()
        else:
            raise ValueError('Cannot compare derivations with opt=%d' % opt)

    #cpdef tuple annotated_rules(self):
        #cdef id_t e
        #if self._annotated_rules is None:
        #    for e in range(self._hg.n_edges()):
        #        LHS = self._hg.label(self._hg.head(e))


cpdef Derivation make_derivation(Hypergraph forest, tuple edges):
    cdef:
        Hypergraph hg = Hypergraph()
        id_t e, c

    for e in edges:
        hg.add_xedge(forest.label(forest.head(e)),
                     tuple([forest.label(c) for c in forest.tail(e)]),
                     forest.weight(e),
                     forest.rule(e))

    return Derivation(hg)

# algorithms
# topsort
# value recursion

# The Extension type is more flexible and it allows for better interaction with python itself
# no need to convert back to python objects

# The struct is completely defined in C

# The struct is viable if every algorithm that deals with a hypergraph is ported to cython
# I think I will maintain both for the time being


# TODO:
# semirings
# viterbi
# ancestral
# kbest
# rescoring
# prune/reweight




