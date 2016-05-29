from grasp.ptypes cimport weight_t, id_t
from grasp.scoring.frepr cimport FComponents
from grasp.semiring._semiring cimport Semiring
from grasp.scoring.model cimport Model


cdef class FeatureVectorFunction:

    cpdef FComponents evaluate(self, id_t e)

    cpdef FComponents reduce(self, tuple edges)


cdef class TableLookupFVecFunction(FeatureVectorFunction):

    cdef:
        list components
        readonly FComponents one
        Semiring semiring


cpdef FComponents derivation_fvec(Model model, Semiring semiring, list components, tuple edges)

cpdef FComponents compute_expected_fvec(Model model, Semiring semiring,
                                       list derivations, weight_t[::1] posterior, list components)
