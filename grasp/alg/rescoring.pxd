from grasp.formal.hg cimport Hypergraph
from grasp.scoring.scorer cimport StatelessScorer
from grasp.semiring._semiring cimport Semiring

from grasp.cfg.symbol cimport Terminal
from grasp.cfg.rule cimport Rule
from grasp.alg.inference cimport AncestralSampler
from grasp.scoring.scorer cimport StatelessScorer, StatefulScorer
from grasp.formal.topsort cimport TopSortTable
from grasp.alg.value cimport ValueFunction

from grasp.ptypes cimport weight_t


cdef class StatelessValueFunction(ValueFunction):

    cdef Hypergraph hg
    cdef StatelessScorer stateless


cpdef Hypergraph weight_edges(Hypergraph hg, ValueFunction omega)


cpdef Hypergraph stateless_rescoring(Hypergraph hg,
                                     StatelessScorer stateless,
                                     Semiring semiring)


cdef class SlicedRescoring:

    cdef:
        Hypergraph _forest
        TopSortTable _tsort
        StatelessScorer _stateless
        StatefulScorer _stateful
        Semiring _semiring
        Rule _goal_rule
        Rule _dead_rule
        Terminal _dead_terminal
        AncestralSampler _sampler0

    cdef _make_slice_variables(self, conditions, str prior_type, str prior_parameter)

    cdef weight_t _lfunc(self, Hypergraph forest, tuple derivation)

    cdef weight_t _nfunc(self, Hypergraph forest, tuple derivation)

    cdef weight_t _ffunc(self, Hypergraph forest, tuple derivation)

    cdef _weighted(self, Hypergraph forest, TopSortTable tsort, ValueFunction u, int batch_size, tuple d0)

    cdef _importance(self, Hypergraph forest, TopSortTable tsort, ValueFunction u, int batch_size)

    cdef _cimportance(self, Hypergraph forest, TopSortTable tsort, ValueFunction u, int batch_size, tuple d0)

    cdef _uniform(self, Hypergraph forest, TopSortTable tsort, ValueFunction u, int batch_size, tuple d0)

    cdef _exact(self, Hypergraph forest, TopSortTable tsort, ValueFunction u, int batch_size)

    cdef tuple _initialise(self, forest, tsort, semiring, strategy, temperature)

    cdef _sample(self, Hypergraph forest, TopSortTable tsort, ValueFunction u, tuple prev_d, int batch_size, str algorithm)

    cpdef sample(self, args)