from grasp.alg.deduction cimport EarleyRescorer
from grasp.alg.slicing cimport SliceReturn, slice_forest
cimport grasp.alg.slicevars as sv
from grasp.alg.value cimport ValueFunction, EdgeWeight, CascadeValueFunction
from grasp.alg.value cimport ScaledEdgeWeight, LookupFunction, ThresholdValueFunction
from grasp.alg.value cimport derivation_value
from grasp.alg.inference cimport AncestralSampler
from grasp.formal.traversal import bracketed_string

from grasp.formal.hg cimport Hypergraph
from grasp.formal.traversal cimport top_down_left_right
from grasp.formal.topsort cimport TopSortTable, AcyclicTopSortTable

from grasp.scoring.scorer cimport StatelessScorer, StatefulScorer
from grasp.semiring._semiring cimport Semiring
from grasp.cfg.symbol cimport Symbol, Terminal
from grasp.cfg.rule cimport Rule, get_leaves
from grasp.cfg.rule cimport NewCFGProduction as CFGProduction

from types import SimpleNamespace
import grasp.ptypes as ptypes
from grasp.ptypes cimport id_t, weight_t
cimport numpy as np
import numpy as np
import itertools
from collections import deque, defaultdict, Counter
from grasp.recipes import progressbar

from grasp.cfg.projection import DerivationYield

from cpython.object cimport Py_EQ, Py_NE, Py_LT

import logging


cdef class StatelessValueFunction(ValueFunction):

    def __init__(self, Hypergraph hg, StatelessScorer stateless):
        self.hg = hg
        self.stateless = stateless

    cpdef weight_t value(self, id_t e):
        return self.stateless.score(self.hg.rule(e))


cpdef Hypergraph weight_edges(Hypergraph hg, ValueFunction omega):
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
    return weight_edges(hg, CascadeValueFunction(semiring.times,
                                                 [EdgeWeight(hg), StatelessValueFunction(hg, stateless)]))


cdef dict make_span_conditions(Hypergraph forest, tuple raw_derivation, Semiring semiring):
    """
    Return a map of conditions based on the head symbols participating in a given derivation.
    The conditions are indexed by the symbols themselves, because their ids are not consistent across slices.
    :param forest: a hypergraph forest
    :param raw_derivation: a raw derivation from this forest (edges are numbered consistently with the forest)
    :param semiring:
    :return: dict mapping a head Symbol onto an edge score
    """
    cdef id_t e
    return {forest.label(forest.head(e)): semiring.as_real(forest.weight(e)) for e in raw_derivation}


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

    cdef readonly tuple edges
    cdef readonly weight_t score

    def __init__(self, tuple edges, weight_t score):
        self.edges = edges
        self.score = score

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


    def __init__(self, Hypergraph forest,
                 TopSortTable tsort,
                 StatelessScorer stateless,
                 StatefulScorer stateful,
                 Semiring semiring,
                 Rule goal_rule,
                 Rule dead_rule,
                 Terminal dead_terminal,
                 weight_t temperature0=1.0):
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
        self._forest = forest
        self._tsort = tsort
        self._stateless = stateless
        self._stateful = stateful
        self._semiring = semiring
        self._goal_rule = goal_rule
        self._dead_rule = dead_rule
        self._dead_terminal = dead_terminal

        #logging.info('l-forest: derivations=%d', self._sampler0.n_derivations())

        #self._make_conditions = make_span_conditions

        #self._report.init(terminals=self._forest.n_terminals(),
        #                  nonterminals=self._forest.n_nonterminals(),
        #                  edges=len(self._forest),
        #                  derivations=self._sampler0.n_derivations(),
        #                  ancestral_time=dt)



    cdef _make_slice_variables(self, conditions, str prior_type, str prior_parameter):
        if prior_type == 'const':
            prior = sv.ConstantPrior(float(prior_parameter))
        elif prior_type == 'sym':
            prior = sv.SymmetricGamma(scale=float(prior_parameter))
        elif prior_type == 'asym':
            if prior_parameter == 'mean':
                prior = sv.AsymmetricGamma(scales=gamma_priors(self._forest, self._semiring))
            else:
                try:
                    percentile = float(prior_parameter)
                except ValueError:
                    raise ValueError("The parameter of the asymmetric "
                                     "Gamma must be the keyword 'mean' or a number between 0-100: %s" % percentile)
                if not (0 <= percentile <= 100):
                    raise ValueError("A percentile is a real number between 0 and 100: %s" % percentile)
                prior = sv.AsymmetricGamma(scales=gamma_priors(self._forest, self._semiring, percentile=percentile))

        return sv.ExpSpanSliceVariables(conditions, prior)

    cdef weight_t _lfunc(self, Hypergraph forest, tuple derivation):
        return derivation_value(forest, derivation, self._semiring)

    cdef weight_t _nfunc(self, Hypergraph forest, tuple derivation):
        """
        Rescores a derivation seen as a sequence of rule applications, thus abstracting away from the
        hypergraph representation.

        :param d: sequence of Rule applications
        :return: value
        """
        cdef:
            Semiring semiring = self._semiring
            weight_t stateless, stateful
            id_t e, leaf
            Rule r
            list rules = [forest.rule(e) for e in derivation]
            tuple leaves = top_down_left_right(forest, derivation, terminal_only=True)

        # score using the complete f(d|u)
        # 1. Stateless scorers
        stateless = semiring.times.reduce([self._stateless.score(forest.rule(e)) for e in derivation])
        # 2. Stateful scorers
        stateful = self._stateful.score_yield([forest.label(leaf) for leaf in leaves])
        return semiring.times(stateless, stateful)

    cdef weight_t _ffunc(self, Hypergraph forest, tuple derivation):
        return self._semiring.times(self._nfunc(forest, derivation), self._lfunc(forest, derivation))

    cdef _uniform(self, Hypergraph forest, TopSortTable tsort, ValueFunction u, int batch_size, tuple d0):
        """
        Draw from the slice by uniform sampling from the support and then resampling.
        :param forest: locally scored sliced forest
        :param tsort: forest's top sort table
        :param u:
        :param batch_size: number of samples
        :param d0: the previous state of the Markov chain, we sample derivations to compete with this one.
        :return: derivation (consistent with input slice), conditions, number of derivations in the slice
        """
        cdef:
            Semiring semiring = self._semiring
            list support
            weight_t[::1] empdist, fd  # empirical distribution
            id_t e
            tuple d, sampled_derivation
            size_t i, n
            weight_t nonlocal_score, residual_score, score, denominator
            AncestralSampler sampler

        logging.debug(' [uniform] Uniform sampling %d candidates from S to compete against previous state', batch_size)
        assert d0 is not None, 'Uniform sampling requires the previous state of the Markov chain'

        # 1. A uniform sampler over state space
        sampler = AncestralSampler(forest, tsort, ThresholdValueFunction(u, semiring, semiring.zero, semiring))

        # 1. Sample a number of derivations and group them by identity
        counts = Counter(sampler.sample(batch_size))
        # make sure prev_d is in the sample
        counts.update([d0])

        # 2. Compute the empirical (importance) distribution r(d) \propto f(d|u)/g(d|u)
        support = [None] * len(counts)
        empdist = np.zeros(len(counts))
        fd = np.zeros(len(counts))
        for i, (d, n) in enumerate(counts.items()):
            support[i] = d
            # score using the complete f(d|u)
            nonlocal_score = self._nfunc(forest, d)  # this is n(d)
            residual_score = u.reduce(semiring.times, d)  # this is u(d)
            score = semiring.times(residual_score, nonlocal_score)
            # take the sample frequency into account
            empdist[i] = semiring.times(score, semiring.from_real(n))
            # f(d) = n(d) * l(d)
            fd[i] = semiring.times(nonlocal_score, self._lfunc(forest, d))
        empdist = semiring.normalise(empdist)

        # 3. Sample a derivation
        i = semiring.plus.choice(empdist)
        sampled_derivation = support[i]
        # make conditions based on the slice variables and l(d)
        #c_i = make_span_conditions(forest, d_i, semiring)
        return SampleReturn(sampled_derivation, fd[i])  #, c_i, sampler.n_derivations()

    cdef _weighted(self, Hypergraph forest, TopSortTable tsort, ValueFunction u, int batch_size, tuple d0):
        """
        Draw from the slice by importance sampling.
        Proposal types:
            0) Standard proposal: edges are weighted 1/Exp(u; parameters)
                where u is the value of the slice variable associated with the edge's head symbol
        :param forest: locally scored sliced forest
        :param tsort: forest's top sort table
        :param u: residual function
        :param batch_size: number of samples
        :param d0: previous state of the Markov chain (ignored by this method).
        :return: derivation (consistent with input slice), conditions, number of derivations in the slice
        """
        cdef:
            Semiring semiring = self._semiring
            list support
            weight_t[::1] empdist, fd
            id_t e
            tuple d, sampled_derivation
            size_t i, n
            weight_t nonlocal_score
            # sample from u(d)
            AncestralSampler sampler = AncestralSampler(forest, tsort, u)

        if d0:
            logging.debug(' [cimportance] Importance sampling %d candidates from S to against previous state', batch_size)
        else:
            logging.debug(' [importance] Importance sampling %d candidates from S', batch_size)
        # 1. Sample a number of derivations from u(d) and group them by identity
        counts = Counter(sampler.sample(batch_size))
        if d0:
            counts.update([d0])
        # 2. Compute the empirical (importance) distribution r(d) \propto f(d|u)/g(d|u)
        support = [None] * len(counts)
        empdist = np.zeros(len(counts))
        fd = np.zeros(len(counts))
        for i, (d, n) in enumerate(counts.items()):
            support[i] = d
            # n(d) * u(d) / u(d) = n(d)
            # this is the global part n(d)
            nonlocal_score = self._nfunc(forest, d)  # this is n(d)
            # take the sample frequency into account
            empdist[i] = semiring.times(nonlocal_score, semiring.from_real(n))
            # f(d) = n(d) * l(d)
            fd[i] = semiring.times(nonlocal_score, self._lfunc(forest, d))
        empdist = semiring.normalise(empdist)

        # 3. Sample a derivation
        i = semiring.plus.choice(empdist)
        sampled_derivation = support[i]
        # make conditions based on the slice variables and l(d)
        #c_i = make_span_conditions(forest, d_i, semiring)
        return SampleReturn(sampled_derivation, fd[i])  #, c_i, sampler.n_derivations()

    cdef _importance(self, Hypergraph forest, TopSortTable tsort, ValueFunction u, int batch_size):
        # does not force d0 in the sample
        return self._weighted(forest, tsort, u, batch_size, d0=None)

    cdef _cimportance(self, Hypergraph forest, TopSortTable tsort, ValueFunction u, int batch_size, tuple d0):
        # forces d0 into the sample
        assert d0 is not None, 'I need a valid previous state'
        return self._weighted(forest, tsort, u, batch_size, d0)

    cdef _exact(self, Hypergraph forest, TopSortTable tsort, ValueFunction u, int batch_size):
        """
        Draw from the slice by exact rescoring.
        :param forest: locally scored sliced forest
        :param tsort: forest's top sort table
        :param u:
        :param batch_size: number of samples
        :param d0: previous state of the Markov chain (ignore by this algorithm)
        :return: derivation (consistent with input slice), conditions, number of derivations in the slice
        """
        cdef:
            Semiring semiring = self._semiring
            EarleyRescorer rescorer
            Hypergraph rescored_forest
            AncestralSampler sampler
            tuple d_in_rescored, d_in_original
            list rules
            id_t e
            weight_t fd

        logging.debug(' [exact] Exactly rescoring the slice')
        # 1. Forest rescoring
        rescorer = EarleyRescorer(forest, self._stateless, self._stateful, semiring)
        rescored_forest = rescorer.do(tsort.root(), self._goal_rule)

        # 2. Sample a derivation
        logging.debug(' [exact] Top sorting the rescored slice')
        rescored_tsort = AcyclicTopSortTable(rescored_forest)
        logging.debug(' [exact] Sampling the next state')
        sampler = AncestralSampler(rescored_forest, rescored_tsort)
        # get a derivation from f(d|u) (the rescored forest)
        d_in_rescored = sampler.sample(1)[0]
        fd = derivation_value(rescored_forest, d_in_rescored, semiring)
        # maps it to the original forest, that is, removes annotation
        d_in_original = tuple([rescorer.maps_to(e) for e in d_in_rescored if rescorer.maps_to(e) >= 0])


        # 3. Make new conditions for slice variables
        # edges and weights must come from l
        #c_i = make_span_conditions(forest, d_from_l, semiring)
        #, c_i, sampler.n_derivations()
        return SampleReturn(d_in_original, fd)

    cdef _sample(self, Hypergraph forest, TopSortTable tsort, ValueFunction u, tuple prev_d, int batch_size, str algorithm):
        """
        Delegates sampling to the appropriate algorithm.

        :param forest: the slice
        :param tsort: slice's top sort table
        :param u: the residual value function u(d)
        :param prev_d: last state of the Markov chain
        :param batch_size: number of samples
        :param algorithm: which sampling algorithm
        :return: whatever the sampling algorithm returns.
        """
        if algorithm == 'uniform':
            return self._uniform(forest, tsort, u, batch_size, prev_d)
        elif algorithm == 'importance':
            return self._importance(forest, tsort, u, batch_size)
        elif algorithm == 'cimportance':
            return self._cimportance(forest, tsort, u, batch_size, prev_d)
        else:  # defaults to exact sampling
            return self._exact(forest, tsort, u, batch_size)

    cdef tuple _initialise(self, forest, tsort, semiring, strategy, temperature):
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
            tuple sample
            weight_t fd

        if strategy == 'uniform':
            sampler = AncestralSampler(forest,
                             tsort,
                             omega=ThresholdValueFunction(EdgeWeight(forest),
                                                          semiring,
                                                          semiring.zero,
                                                          semiring))
        else:  # local
            if temperature == 1.0:
                sampler = AncestralSampler(forest, tsort)
            else:
                sampler = AncestralSampler(forest, tsort, omega=ScaledEdgeWeight(forest, 1.0/temperature))

        sample = sampler.sample(1)[0]
        # f(d) = n(d) * l(d)
        fd = self._ffunc(forest, sample)
        return SampleReturn(sample, fd), semiring.as_real(sampler.Z)


    cpdef sample(self, args):
        cdef Hypergraph D = self._forest
        cdef TopSortTable tsort_D = self._tsort
        cdef Semiring semiring = self._semiring
        cdef object markov_chain = deque([])
        cdef:
            sv.SpanSliceVariables slicevars
            SliceReturn slice_return
            TopSortTable tsort_S  # top sort table associated with S

        # Draw the initial sample
        (d0, n_derivations) = self._initialise(D, tsort_D, semiring, args.initial, args.temperature0)
        # Prepare slice variables
        slicevars = self._make_slice_variables(make_span_conditions(D, d0.edges, semiring),
                                               args.prior[0], args.prior[1])
        # Initialise the Markov chain
        markov_chain.append(d0)

        logging.info('The forest contains %d derivations', n_derivations)
        logging.info('Initial sample f(d)=%f: %s', d0.score, bracketed_string(D, d0.edges))
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

        for _ in range(args.burn + (args.samples * args.lag)):

            # get a truncated forest weighted by l(d)
            # and a weight table corresponding to g(d)

            # Remember:
            #   D is the unpruned support
            #   markov_chain contains derivations indexed by edges in D
            #   conditions are always computed with respect to head symbols


            # 1. First we obtain the slice
            logging.debug('1a. Computing the slice')
            slice_return = slice_forest(D, tsort_D, markov_chain[-1].edges, slicevars, semiring, self._dead_rule, self._dead_terminal)

            # Slice returns contains
            #   S: the support
            #   l: the local value function l(d)
            #   u: the residual value function u(d)
            #   d0_in_S: last state of the Markov chain indexed by edges in S

            # we need to top sort nodes in the slice before continuing
            logging.debug('1b. Sorting the slice')
            tsort_S = AcyclicTopSortTable(slice_return.S)

            # TODO: count the number of derivations in S

            # 2. Then we sample from the slice and map the sample back to D
            logging.debug('2. Sampling from slice')
            d_in_S = self._sample(slice_return.S, tsort_S, slice_return.u, slice_return.d0_in_S, args.batch, args.within)
            d_in_D = SampleReturn(slice_return.back_to_D(d_in_S.edges), d_in_S.score)

            # 3. Update the Markov chain and slice variables
            logging.debug('3. Update the Markov chain and slice variables')
            # update the history
            markov_chain.append(d_in_D)
            # reset slice variables based on the sampled derivation
            slicevars.reset(make_span_conditions(D, d_in_D.edges, semiring))

            # reports
            # slices_sizes.append(n_i)
            #logging.info('Sample: tree=%s', inlinetree(make_nltk_tree(d_i)))

            #if n_i == 0:
            #    raise ValueError('I found an empty slice!')

            #self._report.report(len(markov_chain),
            #                    slice_terminals=l_slice.n_terminals(),
            #                    slice_nonterminals=l_slice.n_nonterminals(),
            #                    slice_edges=len(l_slice),
            #                    slice_derivations=n_i,
            #                    slicing_time=dt_s,
            #                    rescoring_time=dt_r,
            #                    constrained_included=muni,
            #                    unconstrained_included=mexp)

        #logging.info('Slice size: mean=%s std=%s', np.mean(slices_sizes[1:]), np.std(slices_sizes[1:]))
        # I always burn the initial derivation (because it is always byproduct of some heuristic)
        markov_chain.popleft()
        return d0, markov_chain