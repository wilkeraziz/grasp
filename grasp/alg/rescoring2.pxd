from grasp.formal.hg cimport Hypergraph
from grasp.semiring._semiring cimport Semiring

from grasp.cfg.rule cimport Rule
from grasp.scoring.frepr cimport FComponents
from grasp.scoring.scorer cimport TableLookupScorer, StatelessScorer, StatefulScorer
from grasp.scoring.model cimport ModelView
from grasp.formal.topsort cimport TopSortTable
from grasp.formal.wfunc cimport WeightFunction
from grasp.scoring.fvecfunc cimport FeatureVectorFunction
from grasp.alg.slicing2 cimport SliceReturn

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
    cdef readonly FComponents mean


cdef class LocalDistribution:

    cdef public Hypergraph forest
    cdef TopSortTable tsort
    cdef WeightFunction wfunc
    cdef FeatureVectorFunction ffunc


cdef class SlicedRescoring:

    cdef:
        ModelView _model
        LocalDistribution _local
        Semiring _semiring
        Rule _goal_rule
        Rule _dead_rule
        TableLookupScorer _lookup
        StatelessScorer _stateless
        StatefulScorer _stateful
        bint _log_slice_size

    cdef weight_t _nfunc(self, Hypergraph forest, tuple derivation)

    cdef tuple _nfunc_and_component(self, Hypergraph forest, tuple derivation)

    cdef tuple _initialise(self, Semiring semiring, str strategy, weight_t temperature)

    cdef _weighted(self, SliceReturn rslice, int batch_size, bint force=?)

    cdef _sample(self, SliceReturn rslice, int batch_size, str algorithm)

    cpdef sample(self,
                 size_t n_samples, size_t burn, size_t lag,  # Markov chain
                 size_t batch_size, str within,  # Slice sampler
                 str initial, weight_t temperature0,  # Initial state of the Markov chain
                 bint normalised_svars,  # nature of slice variables (Gamma or Beta)
                 str shape_type, float shape_parameter,  # distribution over the first parameter
                 str scale_type, float scale_parameter)