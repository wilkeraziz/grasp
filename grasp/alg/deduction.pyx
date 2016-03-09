from cpython.object cimport Py_EQ, Py_NE
from grasp.semiring._semiring cimport Semiring
from grasp.formal.fsa cimport Arc, DFA
from grasp.ptypes cimport id_t, weight_t
from collections import defaultdict, deque
cimport numpy as np
import numpy as np
import grasp.ptypes as ptypes

cdef class Item:
    """
    A dotted item encapsulates an edge and a tuple of intersected states.

    In principle, storing a weight is redundant, because it is computed as a function
    of the edge and the intersected states. Thus, the edge and the tuple of states suffice for hashing.
    We do store the weight in order to optimised some operations (such as conditional completions).
    """
    
    def __init__(self, id_t edge, tuple states, weight_t weight):
        self.edge = edge
        self.states = states
        self.weight = weight

    def __len__(self):
        """The length of an item is the number of states already intersected."""
        return len(self.states)
        
    def __str__(self):
        return 'edge={0} states=({1})'.format(self.edge, 
                                              ' '.join(str(s) for s in self.states))
    
    def __repr__(self):
        return 'Item(%r, %r)' % (self.edge, self.states)
    
    def __hash__(self):
        return hash((self.edge, self.states))
    
    def __richcmp__(x, y, opt):
        if opt == Py_EQ:
            return x.edge == y.edge and x.states == y.states
        elif opt == Py_NE:
            return x.edge != y.edge or x.states != y.states
        else:
            raise ValueError('cannot compare items with opt=%d' % opt)
    
    property start:
        def __get__(self):
            return self.states[0]
        
    property end:
        def __get__(self):
            return self.states[-1]
        
    property dot:
        def __get__(self):
            return len(self.states) - 1
        

cdef class ItemFactory:
    """
    A factory for Item objects. It implements instance management.
    """
    
    def __init__(self, id_t max_edges):
        self._items = []
        # in rescoring we do not know the number Q of states,
        # thus we can only directly index by edge (we could also index by dot)
        # if we knew Q, we could directly index by (edge, start, end)
        # consider writing a factory specialised in parsing
        self._items_by_key = [dict() for _ in range(max_edges)]
        
    def __len__(self):
        return len(self._items)
    
    cpdef Item item(self, id_t i):
        """
        Return the ith item.
        """
        return self._items[i]
        
    cpdef Item get_item(self, id_t edge, tuple states, weight_t weight):
        """
        Return an item constructing it if necessary.

        :param edge: the id of the underlying edge
        :param states: a sequence of states (node ids)
        :param weight: the item's weight
        """
        cdef id_t i = self._items_by_key[edge].get(states, -1)
        if i < 0:
            i = len(self._items)
            self._items_by_key[edge][states] = i
            self._items.append(Item(edge, states, weight))
        return self._items[i]

    cdef pair[id_t, bint] insert(self, id_t edge, tuple states, weight_t weight):
        """
        Return an item constructing it if necessary and an insertion flag.
        :param edge:
        :param states:
        :param weight:
        :return: pair (item id, insertion flag)
        """
        cdef id_t i = self._items_by_key[edge].get(states, -1)
        if i < 0:
            i = len(self._items)
            self._items_by_key[edge][states] = i
            self._items.append(Item(edge, states, weight))
            return pair[id_t, bint](i, True)
        else:
            return pair[id_t, bint](i, False)

    cdef pair[id_t, bint] advance(self, Item item, id_t to, weight_t weight):
        return self.insert(item.edge, item.states + (to,), weight)


cdef class Agenda:
    """
    An agenda of active/passive items.
    Active items are handled through a stack.
    Passive items are organised from fast queries.

    The agenda is aware of the input hypergraph and it knows how to generate an output hypergraph
    from complete items.
    """

    def __init__(self, Hypergraph hg):
        self._hg = hg
        self._waiting = [defaultdict(set) for _ in range(hg.n_nodes())]
        self._generating = [defaultdict(set) for _ in range(hg.n_nodes())]
        self._complete = dict()
        self._active = []

    def __len__(self):
        return len(self._active)

    def __str__(self):
        lines = []
        for lhs, items in self._complete.items():
            for item in items:
                lines.append('{2} ||| {0} ||| {1}'.format(item, self._hg.rule(item.edge), (self._hg.head(item.edge), item.start, item.end)))
        return '\n'.join(lines)

    cdef Item pop(self):
        return self._active.pop()

    cdef void push(self, Item item):
        self._active.append(item)

    cdef void make_generating(self, Item item):
        cdef:
            id_t head = self._hg.head(item.edge)
            id_t start = item.start
            id_t end = item.end

        self._generating[head][start].add(end)

        try:
            self._complete[(head, start, end)].append(item)
        except KeyError:
            self._complete[(head, start, end)] = [item]

    cdef void make_waiting(self, Item item):
        cdef:
            id_t n = self._hg.child(item.edge, item.dot)  # id of the next node
            id_t start = item.end  # the next node will rewrite from the last state of this item
        self._waiting[n][start].add(item)

    cdef void discard(self, Item item):
        pass

    cdef set waiting(self, id_t node, id_t start):
        return self._waiting[node].get(start, set())

    cdef set destinations(self, id_t node, id_t start):
        return self._generating[node].get(start, set())

    cdef Hypergraph make_output(self, id_t root, Rule goal_rule, set initial, set final, list mapping=None):
        """
        :param root: root node of the input hypergraph
        :param goal_rule: a rule that introduces a unique goal symbol
        :param initial: set of initial states (if None, does not constrain goal rules to spanning from initial states)
        :param final: set of final states (if None, does not constrain goal rules to spanning to final states)
        :param mapping: a list where to store a mapping from output edges to input edges
            when an output edge has no input correspondent (case of edges incoming to the goal node), we map it to -1
        :return: intersection
        """
        cdef:
            Hypergraph output = Hypergraph()
            Item item
            id_t head, start, end, nid, i, e
            tuple tail, skeleton
            list rhs
            set ends
            set discovered = set()
            Symbol lhs, sym

        Q = deque([])  # queue of LHS annotated symbols whose rules are to be created
        # first we create rules for the roots

        for start, ends in self._generating[root].items():
            if initial is not None and start not in initial:  # must span from an initial state
                continue
            for end in ends:
                if final is not None and end not in final:    # to a final state
                    continue
                Q.append((root, start, end))
                discovered.add((root, start, end))
                output.add_xedge(make_span(goal_rule.lhs),
                                 (make_span(self._hg.label(root), start, end),),
                                 0.0,
                                 goal_rule)  # CFGProduction(goal, [self._hg.label(root)], 0.0)
                if mapping is not None:
                    mapping.append(-1)

        # create rules for symbols which are reachable from other generating symbols (starting from the root ones)
        n_discovered = len(discovered)
        while Q:
            skeleton = Q.pop()
            for item in self._complete.get(skeleton, []):
                e = item.edge
                lhs = make_span(self._hg.label(self._hg.head(e)),
                                item.start,
                                item.end)
                tail = self._hg.tail(e)
                rhs = [None] * len(tail)
                for i in range(len(tail)):
                    nid = tail[i]
                    discovered.add((nid, item.states[i], item.states[i + 1]))
                    if n_discovered < len(discovered):
                        n_discovered += 1
                        Q.append((nid, item.states[i], item.states[i + 1]))
                    rhs[i] = make_span(self._hg.label(nid), item.states[i], item.states[i + 1])

                aux = output.add_xedge(lhs, tuple(rhs), item.weight, self._hg.rule(e))
                if mapping is not None:
                    mapping.append(e)

        return output


cdef class DeductiveIntersection:
    """
    This is implements the core of the deductive parser.

    Features:

        * real valued semirings
        * deterministic finite-state input
        * hypergraphs containing glue edges (edge which can only span from initial states)
        * slice variables to control pruning

    General procedure:

        * intialise an Agenda
        * instantiate axioms (virtual)
        * exhaustively apply inference rules to active items
        * offers a default version of the main inference step (but can be overloaded) which consists in
            applying process_complete/process_incomplete/scan dependending on the item
                * process_complete and process_incomplete are virtual
                * after process_complete, marks the complete item as generating
               * after process_incomplete, marks the incomplete item as waiting
                * after scan, marks the incomplete item as discarded
        * offers a default implementation of scan, complete_itself and complete_others
    """

    def __init__(self, Hypergraph hg,
                 Semiring semiring,
                 SliceVariables slicevars):
        self._hg = hg
        self._semiring = semiring
        self._slicevars = slicevars

        self._agenda = Agenda(hg)
        self._ifactory = ItemFactory(hg.n_edges())
        self._glue = np.zeros(hg.n_edges(), dtype=np.int8)

        cdef id_t i
        for i in hg.iterglue():
            self._glue[i] = 1

    cdef bint is_glue(self, id_t e):
        """Whether the edge comes from a glue grammar."""
        return self._glue[e] != 0

    cdef Symbol next_symbol(self, Item item):
        """Label associated with the next node. Never call this method if the item is complete!"""
        return self._hg.label(self._hg.child(item.edge, item.dot))

    cdef bint is_complete(self, Item item):
        """Whether the item is complete (dot has reached the end of the tail)."""
        return self._hg.arity(item.edge) + 1 == len(item)

    cdef bint push(self, const pair[id_t,bint]& insertion):
        """Make an item active if necessary and return an insertion flag."""
        if insertion.second:
            self._agenda.push(self._ifactory.item(insertion.first))
        return insertion.second

    cdef Item pop(self):
        """Pop and return an active item."""
        return self._agenda.pop()

    #cdef weight_t score_on_creation(self, id_t e):
    #    raise NotImplementedError('I do not know how to score an item on creation.')

    #cdef weight_t score_on_completion(self, Item item):
    #    raise NotImplementedError('I do not know how to score an item on completion.')

    #cdef weight_t score_on_scan(self, Item item, Arc arc):
    #    raise NotImplementedError('I do not know how to score an item on scan.')

    cdef bint scan(self, Item item):
        raise NotImplementedError('I do not know how to scan an item.')

    cdef bint complete_itself(self, Item item):
        """
        This operation tries to extend the given item by advancing the dot over generating symbols.

            [X -> alpha * delta beta, [i ... j]] <j, delta, k>
            ---------------------------------------------------
               [X -> alpha delta * beta, [i ... j, k]]

        :param item: an incomplete item
        :returns: whether new items were discovered
        """

        cdef:
            id_t n_items0 = len(self._ifactory)
            id_t n_i = self._hg.child(item.edge, item.dot)  # id of the next node
            id_t start = item.end  # the next node will rewrite from the last state of this item
            id_t end
        for end in self._agenda.destinations(n_i, start):
            self.push(self._ifactory.advance(item, end, item.weight))
        return n_items0 < len(self._ifactory)

    cdef bint complete_others(self, Item item):
        """
        Complete others:

            [X -> alpha * delta beta, [i ... j]] <j, delta, k>
            ---------------------------------------------------
               [X -> alpha delta * beta, [i ... j, k]]

        :param item: a complete item.
        :returns: whether new items were discovered
        """

        cdef:
            id_t n_items0 = len(self._ifactory)
            Item incomplete
        for incomplete in self._agenda.waiting(self._hg.head(item.edge), item.start):
            self.push(self._ifactory.advance(incomplete, item.end, incomplete.weight))
        return n_items0 < len(self._ifactory)

    cdef void process(self, Item item):
        """
        Apply inference rules.

        1) if an item is complete and belongs to the slice, we call process_complete and mark the item as 'generating';
        2) if an item is incomplete, we call process_incomplete and mark the item as 'waiting'.

        :param item: an active item.
        """
        if self.is_complete(item):
            if self._slicevars is None or self._slicevars.is_inside((self._hg.label(self._hg.head(item.edge)),
                                                                     item.start, item.end),
                                                                    self._semiring.as_real(item.weight)):
                self.process_complete(item)
                self._agenda.make_generating(item)
        else:
            if isinstance(self.next_symbol(item), Terminal):
                self.scan(item)
                self._agenda.discard(item)
            else:
                self.process_incomplete(item)
                self._agenda.make_waiting(item)

    cdef void inference(self, id_t root):
        """
        Compute the intersection between the hypergraph and the DFA.

        :param root: input root node.
        :param goal: symbol to be associated with the output goal node.
        :return: the intersection
        """

        self.axioms(root)

        cdef Agenda agenda = self._agenda
        cdef Item item

        while agenda:
            item = agenda.pop()
            self.process(item)

    cpdef Hypergraph do(self, id_t root, Rule goal_rule):
        raise NotImplementedError('I do not know what to do.')

    cdef void axioms(self, id_t root):
        """Instantiate trivial items."""
        raise NotImplementedError('I do not know how to create axioms.')

    cdef void process_incomplete(self, Item item):
        """Process a complete item which falls within the slice."""
        raise NotImplementedError('I do not know how to process incomplete items.')

    cdef void process_complete(self, Item item):
        """Process an incomplete item whose next node is nonterminal."""
        raise NotImplementedError('I do not know how to process complete items.')


cdef class Parser(DeductiveIntersection):

    def __init__(self, Hypergraph hg,
                 DFA dfa,
                 Semiring semiring,
                 SliceVariables slicevars):
        super(Parser, self).__init__(hg, semiring, slicevars)
        self._dfa = dfa

    #cdef weight_t score_on_creation(self, id_t e):
    #    return self._hg.weight(e)

    #cdef weight_t score_on_completion(self, Item item):
    #    return item.weight

    #cdef weight_t score_on_scan(self, Item item, Arc arc):
    #    return self._semiring.times(item.weight, arc.weight)

    cdef bint scan(self, Item item):
        """
        Scan:

                 [X -> alpha * x beta, [i ... j]]:w1
            ----------------------------------------------        <j, x, k>:w2 in E
              [X -> alpha x * beta, [i ... j k]]: w1 * w2

        :param item: an item whose next node is terminal.
        :return: whether the terminal could be scanned.
        """
        cdef:
            Symbol terminal = self.next_symbol(item)
            id_t sfrom = item.end
            id_t a_i = self._dfa.fetch(sfrom, terminal)

        if a_i < 0:
            return False

        cdef:
            Arc arc = self._dfa.arc(a_i)

        self.push(self._ifactory.advance(item,
                                         arc.destination,
                                         self._semiring.times(item.weight, arc.weight)))

        return True

    cpdef Hypergraph do(self, id_t root, Rule goal_rule):
        """
        Compute the intersection between the hypergraph and the DFA.

        :param root: input root node.
        :param goal: symbol to be associated with the output goal node.
        :return: the intersection
        """

        self.inference(root)

        return self._agenda.make_output(root,
                                        goal_rule,
                                        initial=set(self._dfa.iterinitial()),
                                        final=set(self._dfa.iterfinal()))


cdef class EarleyParser(Parser):
    """
    The Earley parser is specialises the skeleton deductive parser by implementing top-down prediction.
    """
    
    def __init__(self, Hypergraph hg, DFA dfa, Semiring semiring, SliceVariables slicevars=None):
        super(EarleyParser, self).__init__(hg, dfa, semiring, slicevars)
        self._predictions = set()
        
    cdef void axioms(self, id_t root):
        cdef id_t e, start
        for start in self._dfa.iterinitial():
            for e in self._hg.iterbs(root):
                self.push(self._ifactory.insert(e, (start,), self._hg.weight(e)))
                self._predictions.add((root, start))

    cdef bint _predict(self, Item item):
        cdef:
            id_t n_i = self._hg.child(item.edge, item.dot)  # id of the next node
            id_t start = item.end  # the next node will rewrite from the last state of this item
            id_t n_preds = len(self._predictions)

        self._predictions.add((n_i, start))
        if n_preds == len(self._predictions):  # no new candidates
            return False

        cdef id_t e_i

        if self._dfa.is_initial(start):  # for initial states we just add whatever edge matches
            for e_i in self._hg.iterbs(n_i):
                self.push(self._ifactory.insert(e_i, (start,), self._hg.weight(e_i)))
        else:  # if a state is not initial, we only create items based on edges which are not  *glue*
            for e_i in self._hg.iterbs(n_i):
                if self.is_glue(e_i):
                    continue
                self.push(self._ifactory.insert(e_i, (start,), self._hg.weight(e_i)))

        return True

    cdef void process_complete(self, Item item):
        self.complete_others(item)

    cdef void process_incomplete(self, Item item):
        if not self._predict(item):
            self.complete_itself(item)


cdef class NederhofParser(Parser):
    """
    This is an implementation of the CKY-inspired intersection due to Nederhof and Satta (2008).

    It specialises the skeleton deductive parser by implementing delayed axioms.
    """

    def __init__(self, Hypergraph hg, DFA dfa, Semiring semiring, SliceVariables slicevars=None):
        super(NederhofParser, self).__init__(hg, dfa, semiring, slicevars)

        # indexes edges by tail[0]
        self._edges_by_tail0 = [[] for _ in range(self._hg.n_nodes())]
        for e in range(self._hg.n_edges()):
            self._edges_by_tail0[self._hg.child(e, 0)].append(e)

    cdef void axioms(self, id_t root):
        """
        The axioms are based on the transitions (E) of the FSA.
        Every rule whose RHS starts with a terminal matching a transition in E gives rise to an item.

        1) instantiate FSA transitions

            <i, z, j>:w1 in E

        2) populate the agenda by calling the operation "Complete tail[0]"

                 <i, z, j>:w1
            -------------------------   (X --> y alpha):w2 in R
            [X -> y * alpha, [i, j]]: w1 * w2

        :param root: disregarded in this version.
        """

        cdef id_t node
        for arc in self._dfa.iterarcs():
            node = self._hg.fetch(arc.label)
            if node < 0:
                continue
            self._complete_tail0(node, arc.origin, arc.destination, arc.weight)

    cdef void _complete_tail0(self, id_t node, id_t start, id_t end, weight_t weight):
        """
        Complete tail[0] can be thought of as a delayed axiom.
        It works as a combination between a prediction and a completion.

                 <i, delta, j>:w1
            -----------------------------    (X -> delta alpha):w2 in R and delta in (N v Sigma)
            [X -> delta * alpha, [i, j]]: w1 * w2

        :param node:
        :param start: origin state
        :param end: destination state
        :param weight: 1 if the node is nonterminal, w(i,delta, j) from the automaton if the node is terminal
        """
        cdef id_t e
        cdef weight_t e_weight
        if self._dfa.is_initial(start):
            for e in self._edges_by_tail0[node]:
                e_weight = self._semiring.times(self._hg.weight(e), weight)
                self.push(self._ifactory.insert(e, (start, end), e_weight))
        else:  # not an initial finite state
            for e in self._edges_by_tail0[node]:
                if self.is_glue(e):  # skip glue edge
                    continue
                e_weight = self._semiring.times(self._hg.weight(e), weight)
                self.push(self._ifactory.insert(e, (start, end), e_weight))

    cdef void process_incomplete(self, Item item):
        self.complete_itself(item)

    cdef void process_complete(self, Item item):
        """
        Complete other items by calling:
            1. complete others
            2. complete tail[0]

        :param item: a complete item.
        """

        cdef:
            id_t head = self._hg.head(item.edge)
            id_t start = item.start
            id_t end = item.end
        if end not in self._agenda.destinations(head, start):  # not yet discovered
            self.complete_others(item)  # complete other items
            self._complete_tail0(head, start, end, self._semiring.one)   # instantiate new items from matching rules


cdef class Rescorer(DeductiveIntersection):

    def __init__(self, Hypergraph hg,
                 StatelessScorer stateless,
                 StatefulScorer stateful,
                 Semiring semiring,
                 SliceVariables slicevars):
        super(Rescorer, self).__init__(hg, semiring, slicevars)
        self._stateless = stateless
        self._stateful = stateful
        if self._stateful:
            self._initial = self._stateful.initial()
            self._final = self._stateful.final()
        else:
            self._initial = 0
            self._final = 0
        self._mapping = []

    cdef weight_t score_on_creation(self, id_t e):
        if self._stateless:
            return self._semiring.times(self._hg.weight(e), self._stateless.score(self._hg.rule(e)))
        else:
            return self._hg.weight(e)

    cdef bint scan(self, Item item):
        cdef weight_t weight
        cdef id_t to
        if self._stateful:
            weight, to = self._stateful.score(self.next_symbol(item), context=item.end)
            # scan the item
            self.push(self._ifactory.advance(item,
                                             to,
                                             self._semiring.times(item.weight, weight)))
            return True
        else:
            self.push(self._ifactory.advance(item,
                                             0,
                                             item.weight))
            return True

    cpdef Hypergraph do(self, id_t root, Rule goal_rule):
        self.inference(root)

        # a rescorer does not constrain the final state
        return self._agenda.make_output(root,
                                        goal_rule,
                                        initial=set([self._initial]),
                                        final=None,
                                        mapping=self._mapping)

    cpdef id_t maps_to(self, id_t e):
        """
        :param e: an annotated edge (that is, in the output forest)
        :return: the original unnanotated edge (that is, in the input forest)
        """
        return self._mapping[e]


cdef class EarleyRescorer(Rescorer):
    """
    The Earley parser is specialises the skeleton deductive parser by implementing top-down prediction.
    """

    def __init__(self, Hypergraph hg,
                 StatelessScorer stateless,
                 StatefulScorer stateful,
                 Semiring semiring,
                 SliceVariables slicevars=None):
        super(EarleyRescorer, self).__init__(hg,
                                             stateless=stateless,
                                             stateful=stateful,
                                             semiring=semiring,
                                             slicevars=slicevars)
        self._predictions = set()

    cdef void axioms(self, id_t root):
        cdef id_t start = self._initial
        cdef id_t e
        for e in self._hg.iterbs(root):
            # In MT for some reason, in some decoders, people don't score the top rule
            # that is, they would do something like this
            # self.push(self._ifactory.insert(e, (start,), self._hg.weight(e)))
            # I don't see a real good reason for that and I don't want to program bypasses, thus
            # I score top rules normally as follows
            self.push(self._ifactory.insert(e, (start,), self.score_on_creation(e)))
            self._predictions.add((root, start))


    cdef bint _predict(self, Item item):
        cdef:
            id_t n_i = self._hg.child(item.edge, item.dot)  # id of the next node
            id_t start = item.end  # the next node will rewrite from the last state of this item
            id_t n_preds = len(self._predictions)

        self._predictions.add((n_i, start))
        if n_preds == len(self._predictions):  # no new candidates
            return False

        cdef id_t e_i

        for e_i in self._hg.iterbs(n_i):
            self.push(self._ifactory.insert(e_i, (start,), self.score_on_creation(e_i)))

        if False:

            if start == self._initial:  # for initial states we just add whatever edge matches
                for e_i in self._hg.iterbs(n_i):
                    self.push(self._ifactory.insert(e_i, (start,), self.score_on_creation(e_i)))
            else:  # if a state is not initial, we only create items based on edges which are not  *glue*
                for e_i in self._hg.iterbs(n_i):
                    if self.is_glue(e_i):
                        continue
                    self.push(self._ifactory.insert(e_i, (start,), self.score_on_creation(e_i)))

        return True

    cdef void process_complete(self, Item item):
        self.complete_others(item)

    cdef void process_incomplete(self, Item item):
        if not self._predict(item):
            self.complete_itself(item)


cpdef weight_t[::1] reweight(Hypergraph forest, SliceVariables slicevars, Semiring semiring):
    cdef weight_t[::1] values = np.zeros(forest.n_edges(), dtype=ptypes.weight)
    cdef id_t e
    if semiring.LOG:
        for e in range(forest.n_edges()):
            values[e] = slicevars.logpdf(forest.label(forest.head(e)).underlying,  # the underlying object is e.g. a tuple (sym, start, end)
                                         semiring.as_real(forest.weight(e)))
    else:
        for e in range(forest.n_edges()):
            values[e] = semiring.from_real(slicevars.pdf(forest.label(forest.head(e)).underlying,
                                                         semiring.as_real(forest.weight(e))))
    return values


# TODO:
# sliced rescoring
# prune/reweight

