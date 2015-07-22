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

from grasp.recipes import smart_wopen, timeit
from grasp.io.results import save_mcmc_derivation, save_mcmc_yields, save_markov_chain
from grasp.io.results import save_viterbi, save_kbest, save_mc_derivations, save_mc_yields
from grasp.semiring import SumTimes, MaxTimes, Counting
from grasp.inference import KBest, AncestralSampler, derivation_value, viterbi_derivation
from grasp.inference import robust_value_recursion as compute_values
from grasp.parsing.exact import Earley, Nederhof
from grasp.parsing.sliced.sampling import group_by_projection, group_by_identity
from grasp.parsing.sliced.sampling import slice_sampling, apply_filters, apply_batch_filters
from grasp.parsing.sliced.slicevars import Beta, Exponential, get_prior, VectorOfPriors
from grasp.cfg import TopSortTable, LazyTopSortTable, CFG, Nonterminal
from grasp.cfg.symbol import make_span
from grasp.cfg.projection import ItemDerivationYield, DerivationYield

from .workspace import make_dirs
from .cmdline import argparser
from .sentence import make_sentence
from .reader import load_grammar
from .rule import get_oov_cfg_productions


def report_info(cfg: CFG, args):
    """
    Report information about the CFG.
    :param cfg: CFG
    :param args: command line arguments
    """
    if args.report_top or args.report_tsort or args.report_cycles:
        tsort = TopSortTable(cfg)
        if args.report_top:
            logging.info('TOP symbols={0} buckets={1}'.format(tsort.n_top_symbols(), tsort.n_top_buckets()))
            for bucket in tsort.itertopbuckets():
                print(' '.join(str(s) for s in bucket))
            sys.exit(0)
        if args.report_tsort:
            logging.info('TOPSORT levels=%d' % tsort.n_levels())
            print(str(tsort))
            sys.exit(0)
        if args.report_cycles:
            loopy = []
            for i, level in enumerate(tsort.iterlevels(skip=1)):
                loopy.append(set())
                for bucket in filter(lambda b: len(b) > 1, level):
                    loopy[-1].add(bucket)
            logging.info('CYCLES symbols=%d cycles=%d' % (tsort.n_loopy_symbols(), tsort.n_cycles()))
            for i, buckets in enumerate(loopy, 1):
                if not buckets:
                    continue
                print('level=%d' % i)
                for bucket in buckets:
                    print(' bucket-size=%d' % len(bucket))
                    print('\n'.join('  {0}'.format(s) for s in bucket))
                print()
            sys.exit(0)


def report_forest(uid: int, forest: CFG, tsorter: LazyTopSortTable, outdir: str, options):
    """
    Report information about the forest.

    :param uid: segment id
    :param forest: a CFG
    :param tsorter: a LazyTopSortTable built for the given forest
    :param outdir: where to save info
    :param options: the command line options
    """

    # count the number of derivations if necessary
    n_derivations = None
    if options.count:
        tsort = tsorter.do()
        values = compute_values(forest, tsort, Counting, omega=lambda e: Counting.convert(e.weight, SumTimes))
        n_derivations = values[tsort.root()]
        logging.info('Forest: edges=%d nodes=%d paths=%d', len(forest), forest.n_nonterminals(), n_derivations)
        with smart_wopen('{0}/count/{1}.gz'.format(outdir, uid)) as fo:
            print('terminals=%d nonterminals=%d edges=%d paths=%d' % (forest.n_terminals(),
                                                                      forest.n_nonterminals(),
                                                                      len(forest),
                                                                      n_derivations),
                  file=fo)
    else:
        logging.info('Forest: edges=%d nodes=%d', len(forest), forest.n_nonterminals())

    # write forest down as a CFG
    if options.forest:
        with smart_wopen('{0}/forest/{1}.gz'.format(outdir, uid)) as fo:
            if n_derivations is None:
                print('# FOREST terminals=%d nonterminals=%d edges=%d' % (forest.n_terminals(),
                                                                          forest.n_nonterminals(),
                                                                          len(forest)),
                      file=fo)
            else:
                print('# FOREST terminals=%d nonterminals=%d edges=%d paths=%d' % (forest.n_terminals(),
                                                                                   forest.n_nonterminals(),
                                                                                   len(forest),
                                                                                   n_derivations),
                      file=fo)
            print(forest, file=fo)


def exact_parsing(seg: 'the input segment (e.g. a Sentence)',
                  grammars: 'list of CFGs',
                  glue_grammars: 'list of glue CFGs',
                  options: 'command line options',
                  outdir: 'where to save results'):
    """Parse the input exactly."""

    semiring = SumTimes
    if options.intersection == 'earley':
        parser = Earley(grammars, seg.fsa,
                        glue_grammars=glue_grammars,
                        semiring=semiring)
    elif options.intersection == 'nederhof':
        parser = Nederhof(grammars, seg.fsa,
                          glue_grammars=glue_grammars,
                          semiring=semiring)
    else:
        raise NotImplementedError("I don't know this intersection algorithm: %s" % options.intersection)

    # make a forest
    logging.info('Parsing...')
    forest = parser.do(root=Nonterminal(options.start), goal=Nonterminal(options.goal))
    if not forest:
        logging.info('[%s] NO PARSE FOUND', seg.id)
        return

    tsorter = LazyTopSortTable(forest)

    # report info if necessary
    report_forest(seg.id, forest, tsorter, outdir, options)

    # decoding strategies

    if options.viterbi:
        tsort = tsorter.do()
        logging.info('Viterbi...')
        d = viterbi_derivation(forest, tsort)
        logging.info('Viterbi derivation: %s %s', derivation_value(d), DerivationYield.derivation(d))
        save_viterbi('{0}/viterbi/{1}.gz'.format(outdir, seg.id),
                     d,
                     omega_d=derivation_value,
                     get_projection=DerivationYield.tree)

    if options.kbest > 0:
        root = make_span(Nonterminal(options.goal), None, None)  # this is the root after intersection
        logging.info('K-best...')
        kbestparser = KBest(forest,
                            root,
                            options.kbest,
                            MaxTimes,
                            traversal=ItemDerivationYield.string,
                            uniqueness=False).do()
        logging.info('Done!')
        derivations = list(kbestparser.iterderivations())
        save_kbest('{0}/kbest/{1}.gz'.format(outdir, seg.id),
                   derivations,
                   omega_d=derivation_value,
                   get_projection=DerivationYield.tree)

    if options.samples > 0:
        logging.info('Sampling...')
        sampler = AncestralSampler(forest, tsorter.do(), generations=options.generations)
        samples = list(sampler.sample(options.samples))
        # group samples by derivation and yield
        derivations = group_by_identity(samples)
        trees = group_by_projection(samples, get_projection=DerivationYield.tree)
        # save the empirical distribution over derivations
        save_mc_derivations('{0}/ancestral/derivations/{1}.gz'.format(outdir, seg.id),
                            derivations,
                            inside=sampler.Z,
                            omega_d=derivation_value)
        # save the empirical distribution over strings
        save_mc_yields('{0}/ancestral/trees/{1}.gz'.format(outdir, seg.id),
                       trees)

    logging.info('[%s] Finished!', seg.id)


def sliced_parsing(seg: 'the input segment (e.g. a Sentence)',
                   grammars: 'a list of CFGs',
                   glue_grammars: 'a list of glue CFGs',
                   options: 'command line options',
                   outdir: 'whete to save results'):
    """Parse the input using sliced forests."""

    if options.free_dist == 'beta':
        dist = Beta
        prior = VectorOfPriors(get_prior(options.prior_a[0], options.prior_a[1]),
                               get_prior(options.prior_b[0], options.prior_b[1]))
    elif options.free_dist == 'exponential':
        dist = Exponential
        prior = get_prior(options.prior_scale[0], options.prior_scale[1])

    markov_chain = slice_sampling(seg.fsa,
                                  grammars,
                                  glue_grammars,
                                  root=Nonterminal(options.start),
                                  N=options.samples,
                                  lag=options.lag,
                                  burn=options.burn,
                                  batch=options.batch,
                                  report_counts=options.count,
                                  goal=Nonterminal(options.goal),
                                  generations=options.generations,
                                  free_dist=dist,
                                  free_dist_prior=prior,
                                  progress=options.progress)

    # apply MCMC filters to reduce hopefully auto-correlation
    batches = apply_filters(markov_chain,
                            burn=options.burn,
                            lag=options.lag)
    samples = apply_batch_filters(batches, resample=options.resample)

    # group by derivation
    derivations = group_by_identity(samples)
    # group by trees (free of nonterminal annotation)
    trees = group_by_projection(samples, DerivationYield.tree)
    # save everything
    save_mcmc_derivation('{0}/slice/derivations/{1}.gz'.format(outdir, seg.id),
                         derivations,
                         omega_d=derivation_value)
    save_mcmc_yields('{0}/slice/trees/{1}.gz'.format(outdir, seg.id), trees)
    if options.save_chain:
        save_markov_chain('{0}/slice/chain/{1}.gz'.format(outdir, seg.id),
                          markov_chain,
                          omega_d=derivation_value,
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

    semiring = SumTimes

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
    report_info(cfg, args)

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
