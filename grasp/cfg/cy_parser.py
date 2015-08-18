"""
This module is an interface for parsing.
One can choose from all available implementations.

:Authors: - Wilker Aziz
"""

import logging
import sys
import traceback
from itertools import chain
from multiprocessing import Pool
from functools import partial
from types import SimpleNamespace

from grasp.recipes import smart_wopen, timeit
from grasp.io.results import save_mcmc_derivation, save_mcmc_yields, save_markov_chain
from grasp.io.results import save_viterbi, save_kbest, save_mc_derivations, save_mc_yields
from grasp.recipes import progressbar

#from grasp.semiring import SumTimes, MaxTimes, Counting
#from grasp.inference import KBest, AncestralSampler, derivation_value, viterbi_derivation
#from grasp.inference import robust_value_recursion as compute_values
#from grasp.parsing.exact import Earley, Nederhof
from grasp.parsing.sliced.sampling import group_by_projection, group_by_identity
#from grasp.parsing.sliced.sampling import slice_sampling, apply_filters, apply_batch_filters
from grasp.parsing.sliced.slicevars import Beta, Exponential, get_prior, VectorOfPriors
#from grasp.cfg import CFG, Nonterminal
#from grasp.cfg.symbol import make_span
from grasp.cfg.projection import DerivationYield  #ItemDerivationYield
from grasp.cfg import CFG, CFGProduction
from grasp.parsing.sliced.sampling import apply_batch_filters, apply_filters

from .workspace import make_dirs
from .cmdline import argparser
from .sentence import make_sentence
from .reader import load_grammar
from .rule import get_oov_cfg_productions

from grasp.formal.topsort import RobustTopSortTable
from grasp.formal.hg import cfg_to_hg
from grasp.formal.fsa import make_dfa
from grasp.parsing.exact.deduction import Earley, Nederhof
import grasp.semiring as semiring
from grasp.cfg.symbol import Nonterminal
from grasp.inference._inference import viterbi_derivation, AncestralSampler
from grasp.inference._value import derivation_value
from collections import Counter

from grasp.semiring import SumTimes


class LazyTopSortTable(object):

    def __init__(self, forest):
        self._forest = forest
        self._tsort = None

    def do(self):
        if self._tsort is None:
            self._tsort = RobustTopSortTable(self._forest)
        return self._tsort


def group_raw(forest, raw_samples):
    raw_dist = Counter(raw_samples)
    output = [SimpleNamespace(derivation=[forest.rule(e) for e in d], count=n)
              for d, n in raw_dist.most_common()]
    return output


def make_forest(hg):
    forest = CFG()
    for e in range(hg.n_edges()):
        lhs = hg.label(hg.head(e))
        rhs = [hg.label(n) for n in hg.tail(e)]
        forest.add(CFGProduction(lhs, rhs, hg.weight(e)))
    return forest


def exact_parsing(seg: 'the input segment (e.g. a Sentence)',
                  grammars: 'list of CFGs',
                  glue_grammars: 'list of glue CFGs',
                  options: 'command line options',
                  outdir: 'where to save results'):
    """Parse the input exactly."""

    logging.debug('Building input hypergraph')
    hg = cfg_to_hg(grammars, glue_grammars)
    root = hg.fetch(Nonterminal(options.start))
    dfa = make_dfa(seg.signatures)
    if options.intersection == 'earley':
        parser = Earley(hg, dfa, semiring.inside)
    else:
        parser = Nederhof(hg, dfa, semiring.inside)
    logging.debug('Parsing')
    forest = parser.do(root, Nonterminal(options.goal))

    if not forest:
        logging.info('[%s] NO PARSE FOUND', seg.id)
        return

    tsorter = LazyTopSortTable(forest)

    # report info if necessary
    # report_forest(seg.id, forest, tsorter, outdir, options)

    # decoding strategies

    omega_d = lambda d: semiring.inside.times.reduce([r.weight for r in d])

    if options.viterbi:
        tsort = tsorter.do()
        logging.info('Viterbi...')

        d = viterbi_derivation(forest, tsort)
        score = derivation_value(forest, d, semiring.inside)
        logging.info('Viterbi derivation: %s', score)  #, DerivationYield.derivation(d))
        logging.info('Saving...')
        save_viterbi('{0}/viterbi/{1}.gz'.format(outdir, seg.id),
                     [forest.rule(e) for e in d],
                     omega_d=omega_d,
                     get_projection=DerivationYield.derivation)

    if options.samples > 0:
        tsort = tsorter.do()
        logging.info('Sampling...')
        sampler = AncestralSampler(forest, tsort)
        raw_samples = sampler.sample(options.samples)

        logging.info('Saving...')
        derivations = group_raw(forest, raw_samples)
        save_mc_derivations('{0}/ancestral/derivations/{1}.gz'.format(outdir, seg.id),
                            derivations,
                            inside=sampler.Z,
                            omega_d=omega_d)

    logging.info('[%s] Finished!', seg.id)


from grasp.parsing.sliced.slicevars import SliceVariables
from grasp.parsing.exact.deduction import reweight
from grasp.inference._value import LookupFunction
from collections import defaultdict


def make_batch_conditions(forest, raw_derivations):
    if len(raw_derivations) == 1:
        d = raw_derivations[0]
        conditions = {forest.label(forest.head(e)).underlying:
                          semiring.inside.as_real(forest.weight(e)) for e in d}
    else:
        conditions = defaultdict(set)
        for d in raw_derivations:
            [conditions[forest.label(forest.head(e)).underlying].add(semiring.as_real(forest.weight(e))) for e in d]
        conditions = {s: min(thetas) for s, thetas in conditions.items()}
    return conditions


def uninformed_conditions(hg, dfa, slicevars, root, goal, batch, algorithm):
    """
    Search for an initial set of conditions without any heuristics.

    :param grammars:
    :param glue_grammars:
    :param fsa:
    :param slicevars:
    :param root:
    :param goal:
    :param batch:
    :param generations:
    :param semiring:
    :return:
    """

    while True:

        if algorithm == 'earley':
            parser = Earley(hg, dfa, semiring.inside, slicevars)
        else:
            parser = Nederhof(hg, dfa, semiring.inside, slicevars)

        # compute a slice (a randomly pruned forest)
        logging.debug('Computing slice...')
        forest = parser.do(root, goal)
        if not forest:
            logging.debug('NO PARSE FOUND')
            slicevars.reset()  # reset the slice variables (keeping conditions unchanged if any)
            continue

        tsort = RobustTopSortTable(forest)
        values = reweight(forest, slicevars, semiring.inside)
        sampler = AncestralSampler(forest,
                                   tsort,
                                   LookupFunction(values))
        raw_derivations = sampler.sample(batch)
        return make_batch_conditions(forest, raw_derivations)


def sliced_parsing(seg: 'the input segment (e.g. a Sentence)',
                   grammars: 'a list of CFGs',
                   glue_grammars: 'a list of glue CFGs',
                   options: 'command line options',
                   outdir: 'whete to save results'):
    """Parse the input using sliced forests."""

    # Input Hypergraph
    logging.debug('Building input hypergraph')
    hg = cfg_to_hg(grammars, glue_grammars)
    root = hg.fetch(Nonterminal(options.start))
    dfa = make_dfa(seg.signatures)
    goal = Nonterminal(options.goal)

    # Slice variables
    if options.free_dist == 'beta':
        dist = Beta
        prior = VectorOfPriors(get_prior(options.prior_a[0], options.prior_a[1]),
                               get_prior(options.prior_b[0], options.prior_b[1]))
    elif options.free_dist == 'exponential':
        dist = Exponential
        prior = get_prior(options.prior_scale[0], options.prior_scale[1])

    u = SliceVariables({}, dist, prior)
    logging.debug('%s prior=%r', dist.__name__, prior)
    # make initial conditions
    # TODO: consider intialisation heuristics such as attempt_initialisation(fsa, grammars, glue_grammars, options)
    logging.info('Looking for initial set of conditions...')
    conditions = uninformed_conditions(hg, dfa, u, root, goal, options.batch, options.intersection)
    logging.info('Done')
    u.reset(conditions)

    # Sampling
    sizes = [0, 0, 0]  # number of nodes, edges and derivations (for logging purposes)
    if options.count:
        report_size = lambda: ' nodes={:5d} edges={:5d} |D|={:5d} '.format(*sizes)
    else:
        report_size = lambda: ' nodes={:5d} edges={:5d}'.format(sizes[0], sizes[1])
    if options.progress:
        bar = progressbar(range(options.burn + (options.samples * options.lag)), prefix='Sampling', dynsuffix=report_size)
    else:
        bar = range(options.burn + (options.samples * options.lag))

    # sample
    markov_chain = []
    for _ in bar:
        # create a bottom-up parser with slice variables
        parser = Nederhof(hg, dfa, semiring.inside, u)

        # compute a slice (a randomly pruned forest)
        forest = parser.do(root, goal)
        if not forest:
            raise ValueError('A slice can never be emtpy.')

        # sample from the slice
        tsort = RobustTopSortTable(forest)
        residual = reweight(forest, u, semiring.inside)
        sampler = AncestralSampler(forest, tsort, LookupFunction(residual))
        raw_derivations = sampler.sample(options.batch)
        # update the slice variables and the state of the Markov chain
        u.reset(make_batch_conditions(forest, raw_derivations))

        markov_chain.append([tuple(forest.rule(e) for e in d) for d in raw_derivations])

        # update logging information
        sizes[0], sizes[1] = forest.n_nodes(), forest.n_edges()
        if options.count:  # reporting counts
            sizes[2] = sampler.n_derivations()

    # apply MCMC filters to reduce hopefully auto-correlation
    batches = apply_filters(markov_chain,
                            burn=options.burn,
                            lag=options.lag)
    samples = apply_batch_filters(batches, resample=options.resample)

    # group by derivation
    derivations = group_by_identity(samples)
    # group by trees (free of nonterminal annotation)
    #trees = group_by_projection(samples, DerivationYield.tree)
    # save everything
    omega_d = lambda d: semiring.inside.times.reduce([r.weight for r in d])
    save_mcmc_derivation('{0}/slice/derivations/{1}.gz'.format(outdir, seg.id),
                         derivations,
                         omega_d=omega_d)
    #save_mcmc_yields('{0}/slice/trees/{1}.gz'.format(outdir, seg.id), trees)
    if options.save_chain:
        save_markov_chain('{0}/slice/chain/{1}.gz'.format(outdir, seg.id),
                          markov_chain,
                          omega_d=omega,
                          flat=False)


def do(seg, grammars, glue_grammars, options, outdir):
    if options.framework == 'exact':
        exact_parsing(seg, grammars, glue_grammars, options, outdir)
    elif options.framework == 'slice':
        sliced_parsing(seg, grammars, glue_grammars, options, outdir)
    else:
        raise NotImplementedError(
            'I do not yet know how to perform inference in this framework: %s' % options.framework)


@timeit
def t_do(seg, grammars, glue_grammars, options, outdir):
    """This is a timed version of the method `do`"""
    return do(seg, grammars, glue_grammars, options, outdir)


def core(job, args, outdir):
    """
    The main pipeline.

    :param job: a tuple containing an id and an input string
    :param args: the command line options
    :param outdir: where to save results
    """

    # Load main grammars
    logging.info('Loading main grammar...')
    cfg = load_grammar(args.grammar, args.grammarfmt, args.log)
    logging.info('Main grammar: terminals=%d nonterminals=%d productions=%d', cfg.n_terminals(),
                 cfg.n_nonterminals(),
                 len(cfg))

    # Load additional grammars
    main_grammars = [cfg]
    if args.extra_grammar:
        for grammar_path in args.extra_grammar:
            logging.info('Loading additional grammar: %s', grammar_path)
            grammar = load_grammar(grammar_path, args.grammarfmt, args.log)
            logging.info('Additional grammar: terminals=%d nonterminals=%d productions=%d', grammar.n_terminals(),
                         grammar.n_nonterminals(), len(grammar))
            main_grammars.append(grammar)

    # Load glue grammars
    glue_grammars = []
    if args.glue_grammar:
        for glue_path in args.glue_grammar:
            logging.info('Loading glue grammar: %s', glue_path)
            glue = load_grammar(glue_path, args.grammarfmt, args.log)
            logging.info('Glue grammar: terminals=%d nonterminals=%d productions=%d', glue.n_terminals(),
                         glue.n_nonterminals(), len(glue))
            glue_grammars.append(glue)

    # Report information about the main grammar
    # report_info(cfg, args)

    # Make surface lexicon
    surface_lexicon = set()
    for grammar in chain(main_grammars, glue_grammars):
        surface_lexicon.update(t.surface for t in grammar.iterterminals())

    i, input_str = job
    seg = make_sentence(i, input_str, SumTimes, surface_lexicon, args.unkmodel)

    grammars = list(main_grammars)

    if args.unkmodel == 'passthrough':
        grammars.append(CFG(get_oov_cfg_productions(seg.oovs, args.unklhs, SumTimes.one)))

    logging.info('[%d] Parsing %d words: %s', seg.id, len(seg), seg)

    dt, _ = t_do(seg, grammars, glue_grammars, args, outdir)

    logging.info('[%d] parsing time: %s', seg.id, dt)

    return dt


def traced_core(job, args, outdir):
    """
    This method simply wraps core and trace exceptions.
    This is convenient when using multiprocessing.Pool
    """
    try:
        print('[%d] parsing...' % job[0], file=sys.stdout)
        dt = core(job, args, outdir)
        print('[%d] parsing time: %s' % (job[0], dt), file=sys.stdout)
    except:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))


def configure():
    """
    Parse command line arguments, configures the main logger.
    :returns: command line arguments
    """

    args = argparser().parse_args()

    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

    return args


def main():
    """
    Configures the parser by parsing command line arguments and calling the core code.
    It might also profile the run if the user chose to do so.
    """

    args = configure()

    # Prepare output directories
    outdir, _ = make_dirs(args)
    logging.info('Writing files to: %s', outdir)
    print('Writing files to: %s' % outdir)

    # read input
    jobs = [(i, input_str.strip()) for i, input_str in enumerate(sys.stdin)]

    if args.profile:
        import cProfile

        pr = cProfile.Profile()
        pr.enable()
        for job in jobs:
            core(job, args, outdir)
        pr.disable()
        pr.dump_stats(args.profile)
    else:
        pool = Pool(args.cpus if args.cpus > 0 else None)
        # TODO: load grammars only once
        pool.map(partial(traced_core,
                         args=args,
                         outdir=outdir), jobs)

    print('Check output files in: %s' % outdir)


if __name__ == '__main__':
    main()
