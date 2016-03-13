from grasp.formal.hg cimport Hypergraph

from grasp.semiring._semiring cimport Semiring

from grasp.scoring.scorer cimport TableLookupScorer


# TODO: omega should be some sort of model that scores rules locally
cpdef Hypergraph cfg_to_hg(grammars, glue_grammars, omega)


# TODO: omega should be some sort of model that scores rules locally
cpdef Hypergraph make_hypergraph_from_input_view(main_grammars, glue_grammars, omega)


cpdef Hypergraph make_hypergraph(main_grammars,
                                 glue_grammars,
                                 Semiring semiring)


cpdef Hypergraph output_projection(Hypergraph ihg, Semiring semiring, TableLookupScorer localscorer)


cpdef list stateless_components(Hypergraph forest, tuple extractors)


cpdef list lookup_components(Hypergraph forest, tuple extractors)