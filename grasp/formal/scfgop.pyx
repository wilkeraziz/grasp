from grasp.formal.hg cimport Hypergraph
from grasp.semiring._semiring cimport Semiring
from grasp.scoring.scorer cimport TableLookupScorer
from grasp.ptypes cimport id_t
from grasp.cfg.symbol cimport Symbol, Nonterminal, Terminal
from grasp.cfg.srule cimport SCFGProduction, OutputView
from grasp.cfg.rule cimport NewCFGProduction as CFGProduction


cpdef Hypergraph cfg_to_hg(grammars, glue_grammars, omega):  # TODO: make omega a ValueFunction
    """
    Construct a hypergraph from a collection of grammars.

    :param grammars: a sequence of CFG objects.
    :param glue_grammars: a sequence of CFG objects (the "glue" constraint applies).
    :return: a Hypergraph
    """
    cdef Hypergraph hg = Hypergraph()
    cdef CFGProduction rule

    for grammar in grammars:  # TODO make Grammar/CFG an Extension type
        for rule in grammar:
            hg.add_xedge(rule.lhs, rule.rhs, omega(rule), rule, glue=False)
    for grammar in glue_grammars:
        for rule in grammar:
            hg.add_xedge(rule.lhs, rule.rhs, omega(rule), rule, glue=True)

    return hg


cpdef Hypergraph make_hypergraph_from_input_view(main_grammars, glue_grammars, omega):
    """
    Construct a hypergraph representation for a collection of synchronous grammars.
    :param main_grammars:
    :param glue_grammars:
    :param semiring:
    :return: Hypergraph
    """

    cdef Hypergraph hg = Hypergraph()
    cdef SCFGProduction r

    for grammar in main_grammars:
        for view in grammar.iter_inputgroupview():
            hg.add_xedge(view.lhs, view.rhs, omega(view), view)
    for grammar in glue_grammars:
        for view in grammar.iter_inputgroupview():
            hg.add_xedge(view.lhs, view.rhs, omega(view), view, glue=True)

    return hg


cpdef Hypergraph make_hypergraph(main_grammars,
                                 glue_grammars,
                                 Semiring semiring):
    """
    Construct a hypergraph representation for a collection of synchronous grammars.
    :param main_grammars:
    :param glue_grammars:
    :param semiring:
    :return: Hypergraph
    """

    cdef Hypergraph hg = Hypergraph()
    cdef SCFGProduction r

    for grammar in main_grammars:
        for r in grammar:
            hg.add_xedge(r.lhs, r.irhs, semiring.one, r)
    for grammar in glue_grammars:
        for r in grammar:
            hg.add_xedge(r.lhs, r.irhs, semiring.one, r, glue=True)

    return hg


cpdef Hypergraph output_projection(Hypergraph ihg, Semiring semiring, TableLookupScorer localscorer):

    cdef:
        Hypergraph ohg = Hypergraph()
        id_t e, c
        Nonterminal lhs
        Symbol sym
        size_t nt, i, a
        SCFGProduction srule

    for e in range(ihg.n_edges()):
        lhs = ihg.label(ihg.head(e))

        input_nts = [ihg.label(c) for c in ihg.tail(e) if ihg.is_nonterminal(c)]

        view = ihg.rule(e)  # get the synchronous rule
        for srule in view.group:
            i = 0
            rhs = []
            for sym in srule.orhs:
                if isinstance(sym, Terminal):
                    rhs.append(sym)
                else:
                    a = srule.alignment[i] - 1  # alignment
                    rhs.append(input_nts[a])
                    i += 1
            ohg.add_xedge(lhs,
                          tuple(rhs),
                          semiring.times(ihg.weight(e), localscorer.score(OutputView(srule))),
                          OutputView(srule),
                          ihg.is_glue(e))

    # goal rules are treated independently because they are not part of the original grammar.
    #for goal_rule in f_forest.iterrules(f_root):
    #    gr = CFGProduction(goal_rule.lhs, [s for s in goal_rule.rhs], goal_rule.weight)
    #    e_forest.add(gr)
    #    logging.info('Goal rule: %s', gr)

    # the target forest has exactly the same root symbol as the source forest
    #e_root = f_root

    return ohg
