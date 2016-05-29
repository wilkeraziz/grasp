from grasp.ptypes cimport id_t, weight_t


cdef class FeatureVectorFunction:

    cpdef FComponents evaluate(self, id_t e):
        pass

    cpdef FComponents reduce(self, tuple edges):
        pass


cdef class TableLookupFVecFunction(FeatureVectorFunction):
    """
    A value function that consists in plain simple table lookup.
    """

    def __init__(self, list components, FComponents one, Semiring semiring):
        self.components = components
        self.one = one
        self.semiring = semiring

    cpdef FComponents evaluate(self, id_t e):
        return self.components[e]

    cpdef FComponents reduce(self, tuple edges):
        cdef FComponents vec = self.one
        cdef id_t e
        for e in edges:
            vec = vec.hadamard(<FComponents>self.components[e], self.semiring.times)
        return vec


cpdef FComponents derivation_fvec(Model model, Semiring semiring, list components, tuple edges):
    cdef FComponents vec = model.constant(semiring.one)
    cdef id_t e
    for e in edges:
        vec = vec.hadamard(<FComponents>components[e], semiring.times)
    return vec


cpdef FComponents compute_expected_fvec(Model model, Semiring semiring,
                                        list derivations, weight_t[::1] posterior, list components):
    cdef:
        size_t i
        size_t N = len(derivations)
        FComponents mean = model.constant(semiring.one)
        #weight_t H = 0.0
    for i in range(N):
        derivation = derivations[i]
        dvec = components[i]
        prob = posterior[i]
        mean = mean.hadamard(dvec.power(prob, semiring), semiring.times)
        #H += prob * cppmath.log(prob)
    return mean
