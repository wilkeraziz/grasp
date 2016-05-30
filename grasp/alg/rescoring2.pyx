cimport grasp.alg.slicevars as sv
from grasp.alg.slicevars cimport NormalisedSliceCheckFunction
from grasp.alg.slicevars cimport SliceCheckFunction
from grasp.alg.slicevars cimport SliceVariables
from grasp.alg.slicevars cimport Prior as SlicePrior
from grasp.alg.deduction cimport EarleyRescorer
from grasp.alg.slicing2 cimport slice_forest
from grasp.alg.inference cimport AncestralSampler
from grasp.alg.inference cimport SlicedAncestralSampler

from grasp.formal.wfunc cimport WeightFunction, HypergraphLookupFunction, ReducedFunction
from grasp.formal.wfunc cimport BooleanFunction
from grasp.formal.wfunc cimport TableLookupFunction, ThresholdFunction
from grasp.formal.wfunc cimport ScaledFunction, derivation_weight
from grasp.formal.traversal import bracketed_string
from grasp.formal.hg cimport Hypergraph
from grasp.formal.traversal cimport top_down_left_right
from grasp.formal.topsort cimport TopSortTable, AcyclicTopSortTable

from grasp.scoring.frepr cimport FComponents
from grasp.scoring.scorer cimport StatelessScorer, StatefulScorer
from grasp.scoring.fvecfunc cimport compute_expected_fvec

from grasp.semiring._semiring cimport Semiring

from grasp.cfg.rule cimport Rule

from grasp.recipes import progressbar

import grasp.ptypes as ptypes
from grasp.ptypes cimport id_t, weight_t

from types import SimpleNamespace
cimport numpy as np
import numpy as np
import itertools
from collections import deque, defaultdict, Counter
from time import time, strftime

from cpython.object cimport Py_EQ, Py_NE, Py_LT

import logging


cdef class StatelessValueFunction(WeightFunction):

    def __init__(self, Hypergraph hg, StatelessScorer stateless):
        self.hg = hg
        self.stateless = stateless

    cpdef weight_t value(self, id_t e):
        return self.stateless.score(self.hg.rule(e))


cpdef Hypergraph weight_edges(Hypergraph hg, WeightFunction omega):
    """
    Plain obvious stateless rescoring.

    :param forest: a CFG
    :param omega: a value function to score edges
    :return: a hypergraph scored by omega
    """

    cdef Hypergraph ohg = Hypergraph()
    cdef id_t e
    cdef weight_t weight
    # this is to make sure the label-node correspondence remains unchanged
    for n in range(hg.n_nodes()):
        ohg.add_node(hg.label(n))
    # this is to make sure the rule-edge correspondence remains unchanged
    for e in range(hg.n_edges()):
        weight = omega.value(e)
        ohg.add_xedge(hg.label(hg.head(e)),
                      tuple([hg.label(c) for c in hg.tail(e)]),
                      weight,
                      hg.rule(e),
                      hg.is_glue(e))
    return ohg


cpdef Hypergraph stateless_rescoring(Hypergraph hg,
                                     StatelessScorer stateless,
                                     Semiring semiring):
    return weight_edges(hg, ReducedFunction(semiring.times,
                                            [HypergraphLookupFunction(hg), StatelessValueFunction(hg, stateless)]))


cdef dict make_local_normalisers(Hypergraph forest, WeightFunction omega, Semiring semiring):
    """
    Sum potentials associated with each slice variable.

    :param forest: a forest
    :param omega: edge potentials
    :param semiring: a given semiring
    :return: array Z(s) of normalisers (in the given semiring)
    """
    cdef:
        dict normalisers = dict()
        id_t node, edge
        weight_t bs_sum, z_s
        size_t i
    for node in range(forest.n_nodes()):
        bs_sum = 0.0
        # sum omega(e) for e in BS(node)
        for i in range(forest.n_incoming(node)):
            edge = forest.bs_i(node, i)
            bs_sum = semiring.plus.evaluate(bs_sum, omega.value(edge))
        # accumulate the result in Z(s)
        z_s = semiring.plus.evaluate(normalisers.get(forest.label(node), semiring.zero), bs_sum)
        normalisers[forest.label(node)] = z_s
    return normalisers


cdef dict make_span_conditions(Hypergraph forest,
                               WeightFunction omega,
                               tuple raw_derivation,
                               Semiring semiring):
    """
    Return a map of conditions based on the head symbols participating in a given derivation.
    The conditions are indexed by the symbols themselves, because their ids are not consistent across slices.
    :param forest: a hypergraph forest
    :param raw_derivation: a raw derivation from this forest (edges are numbered consistently with the forest)
    :param semiring:
    :return: conditions (as a Real value) on some of the slice variables
    """
    cdef id_t e
    cdef dict conditions =  {forest.label(forest.head(e)): semiring.as_real(omega.value(e)) for e in raw_derivation}
    #for e in raw_derivation:
    #    print('e=%d s=%s phi(e)=%s' % (e, forest.label(forest.head(e)), semiring.as_real(omega.value(e))))
    #    print('')
    return conditions


cdef dict make_span_normalised_conditions(Hypergraph forest,
                                          WeightFunction omega,
                                          object normalisers,
                                          tuple raw_derivation,
                                          Semiring semiring):
    """
    Return a map of conditions based on the head symbols participating in a given derivation.
    The conditions are indexed by the symbols themselves, because their ids are not consistent across slices.
    :param forest: a hypergraph forest
    :param raw_derivation: a raw derivation from this forest (edges are numbered consistently with the forest)
    :param semiring:
    :return: (normalised) conditions (as a Real value) on some of the slice variables
    """
    cdef id_t e
    cdef dict conditions = {forest.label(forest.head(e)):
                semiring.as_real(semiring.divide(omega.value(e), normalisers[forest.label(forest.head(e))]))
            for e in raw_derivation}
    #for e in raw_derivation:
    #    print('e=%d s=%s phi(e)=%s' % (e, forest.label(forest.head(e)), conditions[forest.label(forest.head(e))]))
    #    print('')
    return conditions



cdef gamma_priors(Hypergraph forest, Semiring semiring, weight_t percentile=-1):
    """
    Return a table mapping a cell to a Gamma scale parameter.
    """
    priors = defaultdict(None)
    if percentile < 0:
        for head in range(forest.n_nodes()):
            thetas = np.array([semiring.as_real(forest.weight(e)) for e in forest.iterbs(head)])
            priors[head] = thetas.mean()
    else:
        for head in range(forest.n_nodes()):
            thetas = np.array([semiring.as_real(forest.weight(e)) for e in forest.iterbs(head)])
            priors[head] = np.percentile(thetas, percentile)
    return priors


cdef class SampleReturn:

    def __init__(self, tuple edges, weight_t score, FComponents components):
        self.edges = edges
        self.score = score
        self.components = components

    def __str__(self):
        return 'edges=%s score=%s components=%s' % (self.edges, self.score, self.components)

    def __len__(self):
        return len(self.edges)

    def __iter__(self):
        return iter(self.edges)

    def __getitem__(self, item):
        return self.edges[item]

    def __hash__(self):
        return hash(self.edges)

    def __richcmp__(x, y, opt):
        if opt == Py_EQ:
            return x.edges == y.edges
        elif opt == Py_NE:
            return x.edges != y.edges
        elif opt == Py_LT:
            return x.score < y.score
        else:
            raise ValueError('Cannot compare SampleReturn with opt=%d' % opt)


cdef class LocalDistribution:

    def __init__(self, Hypergraph forest,
                 TopSortTable tsort,
                 WeightFunction wfunc,
                 FeatureVectorFunction ffunc):
        self.forest = forest
        self.tsort = tsort
        self.wfunc = wfunc
        self.ffunc = ffunc


cdef SlicePrior make_prior(str prior_type, float prior_parameter):
    if prior_type == 'const':
        return sv.ConstantPrior(prior_parameter)
    elif prior_type == 'gamma':
        return sv.SymmetricGamma(scale=prior_parameter)
    else:
        raise ValueError('I do not know this type of Prior: %s' % prior_type)

cdef tuple make_slice_priors(str shape_type, float shape_parameter, str scale_type, float scale_parameter):
    """
    Make priors for unconstrained slice variables.

    :param shape: shape parameter of the distribution over unconstrained slice variables
    :param scale_type: type of scale parameters (e.g. const or sampled from a certain distribution)
    :param scale_parameter: scale parameter of the distribution over unconstrained slice variables
    :return: a shape prior and a scale prior
    """

    cdef SlicePrior shape_prior = make_prior(shape_type, shape_parameter)
    cdef SlicePrior scale_prior = make_prior(scale_type, scale_parameter)
    return shape_prior, scale_prior


cdef make_slice_check_function(Hypergraph forest,
                               WeightFunction potential,
                               dict normalisers,
                               Semiring semiring,
                               SlicePrior shape_prior,
                               SlicePrior scale_prior,
                               tuple edges):
    if normalisers:
        # make normalised conditions (in the Real semiring)
        normalised_conditions = make_span_normalised_conditions(forest, potential, normalisers, edges, semiring)
        # make slice variables based on normalised conditions
        # normalised variables are Beta distributed
        slicevars = sv.BetaSpanSliceVariables(normalised_conditions, shape_prior, scale_prior)
        # make a normalised slice check function
        return NormalisedSliceCheckFunction(forest, potential, normalisers, slicevars, semiring)
    else:
        # make conditions (in the Real semiring)
        conditions = make_span_conditions(forest, potential, edges, semiring)
        # make slice variables based on conditions
        # unnormalised variables are Gamma distributed
        slicevars = sv.GammaSpanSliceVariables(conditions, sv.VectorOfPriors(shape_prior, scale_prior))
        # make a slice check function
        return SliceCheckFunction(forest, potential, slicevars, semiring)


cdef class SlicedRescoring:
    """
    Consider sampling from a distribution proportional to
        f(d) = l(d) * n(d)
    where
        d is a derivation compatible with the input x, that is, \in D(x)
        l(d) is a locally parameterised function of d, that is, if r indexes independent steps in d, then
            l(d) = \prod_r \theta_r
        where
            theta is a parameter vector such that \theta_r \in R+ (or a subset)
            each rule r independently expands a left-hand side structure s
            (we sometimes make that explicit by writing r_s)
        n(d) is a nonlocally parameterised function of d, that is, we do not assume any convenient decomposition

    We do assume l(d) and n(d) tractable to assess for any given derivation.

    We introduce slice variables u, one for each left-hand side structure s.
    Each slice variable u_s \in (0,1) represents
        f(d,u) = n(d)
            * l(d)
            * \prod_{u_s: r_s \in d} I[u_s < \theta_rs]/\theta_rs
            * \prod_{u_s: r_s \not\in d} \phi(u_s)

    where
        the first term is the nonlocal part
        the second term is the local part
        the third term is a uniform distribution over slice variables associated with structures rewriten in d
        the fourth term is a general distribution over slice variables not active in d
            this distribution must assign non-zero probability everywhere in R+ (or the same space as \theta)

    With this choice of parameterisation we get

        f(u_s|d) =
            I[u_s < \theta_rs]/\theta_rs    if r_s \in d
            \phi(u_s)                       otherwise

        f(d|u) \propto n(d) * u(d)
        where
            u(d) = \prod_{r_s \in d} I[u_s < \theta_rs]/\phi(u_s)

    Observe that u(d) > 0 only for derivations observing \theta_rs > u_s for every r_s \in d.
    Let's call that set a *slice* and denote it S.

    Thus, conditioned on a given sample d', we first obtain the slice S (a pruned forest).
    We then sample from f(d|u) in one of three possible ways:

        1. Exactly, where we compute f(d|u) explicitly by intersecting n(d) with u(d) in S.
        2. Approximately, where we sample from S using a proposal distribution g(d).
            We choose g(d) = u(d), which leads to importance weights w(d) = f(d)/g(d) = n(d)
                This is equivalent to
                    i. sampling from S with respect to u(d)
                    ii. and resampling with respect to n(d)
        3. Approximately, where we introduce a selector variable z uniformly distributed over S and sample by Gibbs sampling.
                This is equivalent to
                    i. uniformly sampling from S a derivation d''
                    ii. stochastically choosing between d' and d'' using f(d|u), that is, n(d) * u(d)

    """


    def __init__(self, ModelView model,
                 LocalDistribution local,
                 Semiring semiring,
                 Rule goal_rule,
                 Rule dead_rule):
        """
        Here
            lfunc is l(d),
            n(d) is made of stateless and stateful scorers (for convenience we allow both types).

        :param forest:
        :param tsort:
        :param stateless:
        :param stateful:
        :param semiring:
        :param goal_rule:
        :param dead_rule:
        :param dead_terminal:
        :param temperature0:
        :return:
        """
        self._model = model
        self._local = local
        self._semiring = semiring
        self._goal_rule = goal_rule
        self._dead_rule = dead_rule

        self._lookup = TableLookupScorer(model.nonlocal_model().lookup)
        self._stateless = StatelessScorer(model.nonlocal_model().stateless)
        self._stateful = StatefulScorer(model.nonlocal_model().stateful)

    cdef weight_t _nfunc(self, Hypergraph forest, tuple derivation):
        """
        Rescores a derivation seen as a sequence of rule applications, thus abstracting away from the
        hypergraph representation.

        :param d: sequence of Rule applications
        :return: value
        """
        cdef:
            Semiring semiring = self._semiring
            weight_t lookup = semiring.one
            weight_t stateless = semiring.one
            weight_t stateful = semiring.one
            id_t leaf
            Rule r
            tuple rules = tuple([forest.rule(e) for e in derivation])
            tuple labels = tuple([forest.label(leaf) for leaf in top_down_left_right(forest, derivation, terminal_only=True)])

        # score using the complete f(d|u)
        # 1. Rule scorers
        if self._lookup:
            lookup = semiring.times.reduce([self._lookup.score(r) for r in rules])
        # 2. Stateless scorers
        if self._stateless:
            stateless = semiring.times.reduce([self._stateless.score(r) for r in rules])
        # 3. Stateful scorers
        if self._stateful:
            stateful = self._stateful.score_yield(labels)
        return semiring.times(lookup, semiring.times(stateless, stateful))

    cdef tuple _nfunc_and_component(self, Hypergraph forest, tuple derivation):
        """
        Rescores a derivation seen as a sequence of rule applications, thus abstracting away from the
        hypergraph representation.

        :param d: sequence of Rule applications
        :return: value
        """
        cdef:
            Semiring semiring = self._semiring
            FComponents comp, partial
            weight_t lookup = semiring.one
            weight_t stateless = semiring.one
            weight_t stateful = semiring.one
            id_t e, leaf
            Rule r
            tuple rules = tuple([forest.rule(e) for e in derivation])
            tuple labels = tuple([forest.label(leaf) for leaf in top_down_left_right(forest, derivation, terminal_only=True)])

        # score using the complete f(d|u)
        # 1. Rule scorers
        comp = FComponents([])
        if self._lookup:
            partial, lookup = self._lookup.featurize_and_score_derivation(rules, semiring)
            comp = comp.concatenate(partial)
        # 2. Stateless scorers
        if self._stateless:
            partial, stateless = self._stateless.featurize_and_score_derivation(rules, semiring)
            comp = comp.concatenate(partial)
        # 3. Stateful scorers
        if self._stateful:
            partial, stateful = self._stateful.featurize_and_score_yield(labels)
            comp = comp.concatenate(partial)
        return comp, semiring.times(lookup, semiring.times(stateless, stateful))


    cdef _weighted(self, SliceReturn rslice, int batch_size, bint force=False):
        """
        Draw from the slice by importance sampling.
        Proposal types:
            0) Standard proposal: edges are weighted 1/Exp(u; parameters)
                where u is the value of the slice variable associated with the edge's head symbol
        :param forest: locally scored sliced forest
        :param tsort: forest's top sort table
        :param lfunc: local component over edges (used to compute l(d))
        :param ufunc: residual over edges (used to compute u(d))
        :param batch_size: number of samples
        :param d0: previous state of the Markov chain (ignored by this method).
        :return: derivation (consistent with input slice), conditions, number of derivations in the slice
        """
        cdef:
            Semiring semiring = self._semiring
            list support, fvecs
            weight_t[::1] empdist, n_d, l_d
            id_t e
            tuple d, sampled_derivation
            size_t i, n
            weight_t fd
            FComponents nonlocal_comps, local_comps
            # sample from u(d)
            SlicedAncestralSampler sampler = SlicedAncestralSampler(rslice.forest,
                                                                    rslice.tsort,
                                                                    rslice.residual,
                                                                    rslice.selected_nodes,
                                                                    rslice.selected_edges)

        logging.info('[importance] Derivations in slice: %s', sampler.n_derivations())
        if force:
            logging.debug(' [cimportance] Importance sampling %d candidates from S to against previous state', batch_size)
        else:
            logging.debug(' [importance] Importance sampling %d candidates from S', batch_size)
        # 1. Sample a number of derivations from u(d) and group them by identity
        counts = Counter(sampler.sample(batch_size))
        if force:
            counts.update([rslice.d0])
        # 2. Compute the empirical (importance) distribution r(d) \propto f(d|u)/g(d|u)
        support_size = len(counts)
        support = [None] * support_size
        empdist = np.zeros(support_size, dtype=ptypes.weight)
        i = 0
        fvecs = [None] * len(counts)

        n_d = np.zeros(support_size, dtype=ptypes.weight)
        l_d = np.zeros(support_size, dtype=ptypes.weight)

        # Here we construct feature vectors and importance weights
        for d, n in counts.items():
            support[i] = d
            # n(d) * u(d) / u(d) = n(d)

            # This gets the nonlocal components
            nonlocal_comps, n_d[i] = self._nfunc_and_component(rslice.forest, d)

            # And this gets the local components
            local_comps = self._local.ffunc.reduce(d)
            l_d[i] = derivation_weight(rslice.forest, d, semiring, omega=rslice.local)

            # Merge the two making a complete feature vector
            fvecs[i] = self._model.merge(local_comps, nonlocal_comps)

            # Take the sample frequency into account:
            empdist[i] = semiring.times.evaluate(n_d[i], semiring.from_real(n))
            #empdist[i] = n * semiring.as_real(nonlocal_score)

            i += 1

        # Normalise the importance weights making an empirical distirbution
        empdist = semiring.normalise(empdist)
        #empdist /= empdist.sum()  # normalise the empirical distribution

        # Estimate the expected feature vector
        cdef FComponents mean = compute_expected_fvec(self._model, semiring,
                                                      support, np.exp(empdist), fvecs)

        # Re-sample a derivation (sampling importance resampling)
        i = semiring.plus.choice(empdist)
        sampled_derivation = support[i]

        # 4. Score it exactly wrt f(d)
        fd = semiring.times.evaluate(l_d[i], n_d[i])

        return SampleReturn(sampled_derivation, fd, fvecs[i]), mean

    cdef _sample(self, SliceReturn rslice, int batch_size, str algorithm):
        """
        Delegates sampling to the appropriate algorithm.

        :param forest: the slice
        :param tsort: slice's top sort table
        :param ufunc: the residual value function u(d)
        :param prev_d: last state of the Markov chain
        :param batch_size: number of samples
        :param algorithm: which sampling algorithm
        :return: whatever the sampling algorithm returns.
        """
        return self._weighted(rslice, batch_size, force=False)

    cdef tuple _initialise(self, Semiring semiring, str strategy, weight_t temperature):
        """
        Returns an initial sample and the size of the forest (in number of derivations).
        :param forest:
        :param tsort:
        :param semiring:
        :param strategy: sample uniformly (use: uniform) or from the forest's distribution (use: local)
        :param temperature:
        :return: (sample, number of derivations)
        """
        cdef:
            AncestralSampler sampler
            tuple d
            weight_t fd, nonlocal_score, local_score
            WeightFunction omega
            FComponents comp

        if strategy == 'uniform':
            omega = ThresholdFunction(self._local.wfunc, semiring)
        else:  # local
            if temperature == 1.0:
                omega = self._local.wfunc
            else:
                omega = ScaledFunction(self._local.wfunc, 1.0/temperature)
        sampler = AncestralSampler(self._local.forest, self._local.tsort, omega=omega)

        d = sampler.sample(1)[0]
        # reconstruct feature vectors
        nonlocal_comps, nonlocal_score = self._nfunc_and_component(self._local.forest, d)
        local_comps = self._local.ffunc.reduce(d)
        local_score = derivation_weight(self._local.forest, d, semiring, omega=self._local.wfunc)
        # make a complete feature vector
        fvec = self._model.merge(local_comps, nonlocal_comps)
        # and a complete score
        fd = semiring.times(nonlocal_score, local_score)
        return SampleReturn(d, fd, fvec), semiring.as_real(sampler.Z)

    cpdef sample(self,
                 size_t n_samples, size_t burn, size_t lag,  # Markov chain
                 size_t batch_size, str within,  # Slice sampler
                 str initial, weight_t temperature0,  # Initial state of the Markov chain
                 bint normalised_svars,  # nature of slice variables (Gamma or Beta)
                 str shape_type, float shape_parameter,  # distribution over the first parameter
                 str scale_type, float scale_parameter):  # distribution over the second parameter


        cdef:
            Hypergraph D = self._local.forest
            WeightFunction lfunc = self._local.wfunc
            TopSortTable tsort_D = self._local.tsort
            Semiring semiring = self._semiring
            object markov_chain = deque([])
            object mean_chain = deque([])
            dict conditions
            SliceReturn slice_return
            SlicePrior shape_prior
            SlicePrior scale_prior
            WeightFunction slice_check
            dict local_normalisers = {}

        shape_prior, scale_prior = make_slice_priors(shape_type, shape_parameter, scale_type, scale_parameter)

        # Draw the initial sample
        (d0, n_derivations) = self._initialise(semiring, initial, temperature0)

        if normalised_svars:
            # Here we get normalisers z(s) = sum_{v=v_s(x)} phi(v) for the purpose of slicing
            local_normalisers = make_local_normalisers(D, lfunc, semiring)

        # Make a slice check function
        # which also creates appropriate slice variables depending on
        #  whether or not variables are to be normalised (Beta vs Gamma distributed)
        #  priors for shape and scale parameters
        #  conditions (in the Real semiring) associated with d0.edges
        #   conditions are normalised if necessary
        slice_check = make_slice_check_function(D, lfunc, local_normalisers, semiring,
                                                shape_prior, scale_prior, d0.edges)

        # Initialise the Markov chain
        markov_chain.append(d0)

        logging.debug('The forest contains %d derivations', n_derivations)
        logging.debug('Initial sample f(d)=%f: %s', d0.score, bracketed_string(D, d0.edges))
        #logging.debug('Initial conditions:\n%s', '\n'.join(['{0}: {1}'.format(c, p) for c, p in cond0.items()]))
        # get slice variables

        # BEGIN LOG STUFF
        slices_sizes = [n_derivations]
        # mean ratio of sliced edges constrained by u either uniformly or exponentially distributed
        muni, mexp = 1.0, 1.0
        #report_size = lambda: ' |D|={:5d} uni={:5f} exp={:5f}'.format(slices_sizes[-1], muni, mexp)

        #if args.progress:
        #    bar = progressbar(range(args.burn + (args.samples * args.lag)), prefix='Sampling')  #, dynsuffix=report_size
        #else:
        #    bar = range(args.burn + (args.samples * args.lag))
        # END LOG STUFF

        for _ in range(burn + (n_samples * lag)):

            # get a truncated forest weighted by l(d)
            # and a weight table corresponding to g(d)

            # Remember:
            #   D is the unpruned support
            #   markov_chain contains derivations indexed by edges in D
            #   conditions are always computed with respect to head symbols


            # 1. First we obtain the slice
            logging.debug('1a. Computing the slice')

            slice_return = slice_forest(D, lfunc, slice_check, tsort_D,
                                        markov_chain[-1].edges, semiring, self._dead_rule)

            # Slice returns contains
            #   S: the support
            #   l: the local value function l(d)
            #   u: the residual value function u(d)
            #   d0_in_S: last state of the Markov chain indexed by edges in S

            # 2. Then we sample from the slice and map the sample back to D
            logging.debug('2. Sampling from slice')
            d_in_D, mean_in_D = self._sample(slice_return, batch_size, within)
            #d_in_D = SampleReturn(slice_return.back_to_D(d_in_S.edges), d_in_S.score, d_in_S.components)

            # 3. Update the Markov chain and slice variables
            logging.debug('3. Update the Markov chain and slice variables')
            # update the history
            markov_chain.append(d_in_D)
            mean_chain.append(mean_in_D)
            # reset slice variables based on the sampled derivation
            slice_check = make_slice_check_function(D, lfunc, local_normalisers,
                                                    semiring, shape_prior, scale_prior,
                                                    d_in_D.edges)

        #logging.info('Slice size: mean=%s std=%s', np.mean(slices_sizes[1:]), np.std(slices_sizes[1:]))
        # I always burn the initial derivation (because it is always byproduct of some heuristic)
        markov_chain.popleft()
        return d0, markov_chain, mean_chain


cpdef tuple score_derivation(Hypergraph forest,
                               tuple derivation,
                               Semiring semiring,
                               TableLookupScorer lookup,
                               StatelessScorer stateless,
                               StatefulScorer stateful):
        """
        Rescores a derivation seen as a sequence of rule applications, thus abstracting away from the
        hypergraph representation.

        :param d: sequence of Rule applications
        :return: value
        """
        cdef:
            FComponents comp1, comp2, comp3, partial
            weight_t w1 = semiring.one
            weight_t w2 = semiring.one
            weight_t w3 = semiring.one
            id_t e, leaf
            Rule r
            tuple rules = tuple([forest.rule(e) for e in derivation])
            tuple labels = tuple([forest.label(leaf) for leaf in top_down_left_right(forest, derivation, terminal_only=True)])

        # score using the complete f(d|u)
        # 1. Rule scorers

        if lookup:
            comp1, w1 = lookup.featurize_and_score_derivation(rules, semiring)
        # 2. Stateless scorers
        if stateless:
            comp2, w2 = stateless.featurize_and_score_derivation(rules, semiring)
        # 3. Stateful scorers
        if stateful:
            comp3, w3 = stateful.featurize_and_score_yield(labels)
        return FComponents([comp1, comp2, comp3]), semiring.times(w1, semiring.times(w2, w3))