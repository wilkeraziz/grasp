from grasp.formal.hg cimport Hypergraph
from grasp.scoring.scorer cimport StatelessScorer
from grasp.semiring._semiring cimport Semiring

from grasp.cfg.symbol cimport Terminal
from grasp.cfg.rule cimport Rule
from grasp.alg.inference cimport AncestralSampler
from grasp.scoring.frepr cimport FComponents
from grasp.scoring.scorer cimport TableLookupScorer, StatelessScorer, StatefulScorer
from grasp.formal.topsort cimport TopSortTable
from grasp.formal.wfunc cimport WeightFunction

from grasp.ptypes cimport weight_t


cdef class StatelessValueFunction(WeightFunction):

    cdef Hypergraph hg
    cdef StatelessScorer stateless


cpdef Hypergraph weight_edges(Hypergraph hg, WeightFunction omega)


cpdef Hypergraph stateless_rescoring(Hypergraph hg,
                                     StatelessScorer stateless,
                                     Semiring semiring)


cpdef tuple score_derivation(Hypergraph forest,
                               tuple derivation,
                               Semiring semiring,
                               TableLookupScorer lookup,
                               StatelessScorer stateless,
                               StatefulScorer stateful)


cdef class SampleReturn:

    cdef readonly tuple edges
    cdef readonly weight_t score
    cdef public FComponents components


cdef class SlicedRescoring:

    cdef:
        Hypergraph _forest
        WeightFunction _lfunc
        TopSortTable _tsort
        TableLookupScorer _lookup
        StatelessScorer _stateless
        StatefulScorer _stateful
        Semiring _semiring
        Rule _goal_rule
        Rule _dead_rule

    cdef _make_slice_variables(self, conditions, str prior_type, str prior_parameter)

    cdef weight_t _nfunc(self, Hypergraph forest, tuple derivation)

    cdef tuple _nfunc_and_component(self, Hypergraph forest, tuple derivation)

    cdef tuple _initialise(self, Hypergraph forest, TopSortTable tsort, WeightFunction lfunc, Semiring semiring, str strategy, weight_t temperature)

    cdef _weighted(self, Hypergraph forest, TopSortTable tsort, WeightFunction lfunc, WeightFunction ufunc, int batch_size, tuple d0)

    cdef _importance(self, Hypergraph forest, TopSortTable tsort, WeightFunction lfunc, WeightFunction ufunc, int batch_size)

    cdef _cimportance(self, Hypergraph forest, TopSortTable tsort, WeightFunction lfunc, WeightFunction ufunc, int batch_size, tuple d0)

    cdef _uniform(self, Hypergraph forest, TopSortTable tsort, WeightFunction lfunc, WeightFunction ufunc, int batch_size, tuple d0)

    cdef _uniform2(self, Hypergraph forest, TopSortTable tsort, WeightFunction lfunc, WeightFunction ufunc, int batch_size, tuple d0)

    cdef _exact(self, Hypergraph forest, TopSortTable tsort, WeightFunction lfunc, WeightFunction ufunc, int batch_size)

    cdef _sample(self, Hypergraph forest, TopSortTable tsort, WeightFunction lfunc, WeightFunction ufunc, tuple prev_d, int batch_size, str algorithm)

    cpdef sample(self, size_t n_samples, size_t batch_size, str within,
                 str initial, list prior, size_t burn, size_t lag, weight_t temperature0)