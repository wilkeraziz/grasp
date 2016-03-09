from grasp.alg.deduction cimport EarleyRescorer
from grasp.alg.slicing cimport SliceReturn, slice_forest
cimport grasp.alg.slicevars as sv
from grasp.alg.value cimport ValueFunction, EdgeWeight, ScaledEdgeWeight, LookupFunction, ThresholdValueFunction
from grasp.alg.inference cimport AncestralSampler


from grasp.formal.hg cimport Hypergraph
from grasp.formal.topsort cimport TopSortTable, AcyclicTopSortTable

from grasp.scoring.scorer cimport StatelessScorer, StatefulScorer
from grasp.semiring._semiring cimport Semiring
from grasp.cfg.symbol cimport Symbol, Terminal
from grasp.cfg.rule cimport Rule

from types import SimpleNamespace
import grasp.ptypes as ptypes
from grasp.ptypes cimport id_t, weight_t
cimport numpy as np
import numpy as np
import itertools
from collections import deque, defaultdict, Counter
from grasp.recipes import progressbar



cpdef Hypergraph stateless_rescoring(Hypergraph hg,
                                     StatelessScorer stateless,
                                     Semiring semiring):
    """
    Plain obvious stateless rescoring.

    :param forest: a CFG
    :param stateless: StatelessScorer
    :param semiring: provides times
    :return: the rescored CFG
    """

    cdef Hypergraph ohg = Hypergraph()
    cdef id_t e
    cdef weight_t weight
    # this is to make sure the label-node correspondence remains unchanged
    for n in range(hg.n_nodes()):
        ohg.add_node(hg.label(n))
    # this is to make sure the rule-edge correspondence remains unchanged
    for e in range(hg.n_edges()):
        weight = semiring.times(hg.weight(e), stateless.score(hg.rule(e)))
        ohg.add_xedge(hg.label(hg.head(e)),
                      tuple([hg.label(c) for c in hg.tail(e)]),
                      weight,
                      hg.rule(e),
                      hg.is_glue(e))
    return ohg


cdef AncestralSampler get_ancestral_sampler(Hypergraph forest, TopSortTable tsort, weight_t temperature):
    """
    Return an ancestral sampler for the forest.
    :param forest:
    :param tsort:
    :param temperature: if set to anything different from 1.0, it scales (in log domain) the distribution with 1/temperature.
    :return: AncestralSampler
    """
    return AncestralSampler(forest, tsort) if temperature == 1.0 else AncestralSampler(forest, tsort, omega=ScaledEdgeWeight(forest, 1.0/temperature))


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
        self._forest = forest
        self._tsort = tsort
        self._stateless = stateless
        self._stateful = stateful
        self._semiring = semiring
        self._goal_rule = goal_rule
        self._dead_rule = dead_rule
        self._dead_terminal = dead_terminal

        self._sampler0 = get_ancestral_sampler(self._forest, self._tsort, temperature0)

        #logging.info('l-forest: derivations=%d', self._sampler0.n_derivations())

        #self._make_conditions = make_span_conditions

        #self._report.init(terminals=self._forest.n_terminals(),
        #                  nonterminals=self._forest.n_nonterminals(),
        #                  edges=len(self._forest),
        #                  derivations=self._sampler0.n_derivations(),
        #                  ancestral_time=dt)

    cdef tuple sample_d0(self):
        """Draw an initial derivation from the locally weighted forest."""
        return self._sampler0.sample(1)[0]

    cdef _make_slice_variables(self, conditions, str prior_type, str prior_parameter):
        if prior_type == 'const':
            prior = sv.ConstantPrior(const=float(prior_parameter))
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

    cdef weight_t _rescore_derivation(self, list rules):
        """
        Rescores a derivation seen as a sequence of rule applications, thus abstracting away from the
        hypergraph representation.

        :param d: sequence of Rule applications
        :return: value
        """
        cdef:
            Semiring semiring = self._semiring
            weight_t w = semiring.one
            id_t e
            Rule r
        # 1. Stateless scorers
        # TODO: stateless.total_score(derivation) instead of edge by edge
        w = semiring.times.reduce([self._stateless.score(r) for r in rules])
        # 2. Stateful scorers
        w = semiring.times(w, self._stateful.score_derivation(rules))
        return w

    cdef LookupFunction _get_uniform_distribution(self, Hypergraph forest):
        """Return a value function that scores edges uniformly with respect to their head nodes."""
        cdef:
            weight_t[::1] uniform = np.zeros(forest.n_edges(), dtype=ptypes.weight)
            id_t e
        for e in range(forest.n_edges()):
            # semiring.from_real(1/|BS(head(e))|) =
            #   semiring.times.inverse(|BS(head(e))|)
            uniform[e] = self._semiring.times.inverse(self._semiring.from_real(forest.n_incoming(forest.head(e))))
        return LookupFunction(uniform)

    cdef _uniform(self, Hypergraph forest, TopSortTable tsort, LookupFunction u, int batch_size, tuple d0):
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
            weight_t[::1] empdist  # empirical distribution
            id_t e
            tuple d, sampled_derivation
            size_t i, n
            weight_t nonlocal_score, residual_score, score, denominator
            AncestralSampler sampler

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
        for i, (d, n) in enumerate(counts.items()):
            support[i] = d
            # score using the complete f(d|u)
            nonlocal_score = self._rescore_derivation([forest.rule(e) for e in d])  # this is n(d)
            residual_score = u.reduce(semiring.times, d)  # this is u(d)
            score = semiring.times(residual_score, nonlocal_score)
            # take the sample frequency into account
            empdist[i] = semiring.times(score, semiring.from_real(n))
        empdist = semiring.normalise(empdist)

        # 3. Sample a derivation
        i = semiring.plus.choice(empdist)
        sampled_derivation = support[i]
        # make conditions based on the slice variables and l(d)
        #c_i = make_span_conditions(forest, d_i, semiring)
        return sampled_derivation  #, c_i, sampler.n_derivations()

    cdef _importance(self, Hypergraph forest, TopSortTable tsort, LookupFunction u, int batch_size, tuple d0=None):
        """
        Draw from the slice by importance sampling.
        Proposal types:
            0) Standard proposal: edges are weighted 1/Exp(u; parameters)
                where u is the value of the slice variable associated with the edge's head symbol
        :param forest: locally scored sliced forest
        :param tsort: forest's top sort table
        :param u:
        :param batch_size: number of samples
        :param d0: previous state of the Markov chain (if provided, it is forced into the sample).
        :return: derivation (consistent with input slice), conditions, number of derivations in the slice
        """
        cdef:
            Semiring semiring = self._semiring
            list support
            weight_t[::1] empdist
            id_t e
            tuple d, sampled_derivation
            size_t i, n
            weight_t nonlocal_score
            # sample from u(d)
            AncestralSampler sampler = AncestralSampler(forest, tsort, u)

        # 1. Sample a number of derivations from u(d) and group them by identity
        counts = Counter(sampler.sample(batch_size))
        if d0:
            # make sure d0 is in the sample
            counts.update([d0])
        # 2. Compute the empirical (importance) distribution r(d) \propto f(d|u)/g(d|u)
        support = [None] * len(counts)
        empdist = np.zeros(len(counts))
        for i, (d, n) in enumerate(counts.items()):
            support[i] = d
            # n(d) * u(d) / u(d) = n(d)
            # this is the global part n(d)
            nonlocal_score = self._rescore_derivation([forest.rule(e) for e in d])
            # take the sample frequency into account
            empdist[i] = semiring.times(nonlocal_score, semiring.from_real(n))
        empdist = semiring.normalise(empdist)

        # 3. Sample a derivation
        i = semiring.plus.choice(empdist)
        sampled_derivation = support[i]
        # make conditions based on the slice variables and l(d)
        #c_i = make_span_conditions(forest, d_i, semiring)
        return sampled_derivation  #, c_i, sampler.n_derivations()

    cdef _exact(self, Hypergraph forest, TopSortTable tsort, LookupFunction u, int batch_size, tuple d0=None):
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
            # compute f(d|u) \propto \pi(d) g(d) exactly
            # TODO: change EarleyRescorer to accept a value function!
            # omega=g
            EarleyRescorer rescorer
            # TODO: make a goal_rule for exact rescoring
            Hypergraph rescored_forest
            AncestralSampler sampler
            tuple d_in_rescored, d_in_original
            list rules
            id_t e

        # 1. Forest rescoring
        rescorer = EarleyRescorer(forest, self._stateless, self._stateful, semiring)
        rescored_forest = rescorer.do(tsort.root(), self._goal_rule)

        # 2. Sample a derivation
        sampler = AncestralSampler(rescored_forest, TopSortTable(rescored_forest))
        # get a derivation from f(d|u) (the rescored forest)
        d_in_rescored = sampler.sample(1)[0]
        # maps it to the original forest, that is, removes annotation
        d_in_original = tuple([rescorer.maps_to(e) for e in d_in_rescored if rescorer.maps_to(e) >= 0])

        # 3. Make new conditions for slice variables
        # edges and weights must come from l
        #c_i = make_span_conditions(forest, d_from_l, semiring)

        return d_in_original  #, c_i, sampler.n_derivations()

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
            return self._importance(forest, tsort, u, batch_size, prev_d)
        else:  # defaults to exact sampling
            return self._exact(forest, tsort, u, batch_size, prev_d)


    cdef sample(self, args):
        cdef Hypergraph D = self._forest
        cdef TopSortTable tsort_D = self._tsort
        cdef Semiring semiring = self._semiring
        # 1. Initialise a Markov chain of derivations in D
        cdef object markov_chain = deque([self.sample_d0()])
        # 2. Initialise slice variables
        cdef sv.SpanSliceVariables slicevars = self._make_slice_variables(make_span_conditions(D, markov_chain[-1], semiring),
                                                                      args.prior[0], args.prior[1])
        cdef:
            SliceReturn slice_return
            TopSortTable tsort_S  # top sort table associated with S

        #logging.info('Initial sample: prob=%s tree=%s', self._sampler0.prob(d0),
        #             inlinetree(make_nltk_tree(d0)))

        # get slice variables

        # BEGIN LOG STUFF
        slices_sizes = [self._sampler0.n_derivations()]
        # mean ratio of sliced edges constrained by u either uniformly or exponentially distributed
        muni, mexp = 1.0, 1.0
        #report_size = lambda: ' |D|={:5d} uni={:5f} exp={:5f}'.format(slices_sizes[-1], muni, mexp)

        #if args.progress:
        #    bar = progressbar(range(args.burn + (args.samples * args.lag)), prefix='Sampling', dynsuffix=report_size)
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
            slice_return = slice_forest(D, tsort_D, markov_chain[-1], slicevars, semiring, self._dead_rule, self._dead_terminal)

            # Slice returns contains
            #   S: the support
            #   l: the local value function l(d)
            #   u: the residual value function u(d)
            #   d0_in_S: last state of the Markov chain indexed by edges in S

            # we need to top sort nodes in the slice before continuing
            tsort_S = AcyclicTopSortTable(slice_return.S)

            # 2. Then we sample from the slice and map the sample back to D
            d_in_S = self._sample(slice_return.S, tsort_S, slice_return.u, slice_return.d0_in_S, args.batch, args.within)
            d_in_D = slice_return.back_to_D(d_in_S)

            # update the history
            markov_chain.append(d_in_D)

            # reset slice variables based on the sampled derivation
            slicevars.reset(make_span_conditions(D, d_in_D, semiring))

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
        # markov_chain.popleft()
        return markov_chain