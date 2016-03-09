from grasp.formal.hg cimport Hypergraph
from grasp.scoring.scorer cimport StatelessScorer
from grasp.semiring._semiring cimport Semiring



cpdef Hypergraph stateless_rescoring(Hypergraph hg,
                                     StatelessScorer stateless,
                                     Semiring semiring)