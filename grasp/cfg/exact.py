"""
Pipeline for exact decoding.

:Authors: - Wilker Aziz
"""

import logging


from grasp.semiring import SumTimes, MaxTimes, Counting
from grasp.inference import KBest, AncestralSampler, derivation_value, viterbi_derivation
from grasp.inference import robust_value_recursion as compute_values
from grasp.parsing.exact import Earley, Nederhof
from grasp.io.results import save_viterbi, save_kbest, save_mc_derivations, save_mc_yields
from grasp.parsing.sliced.sampling import group_by_projection, group_by_identity
from grasp.cfg import LazyTopSortTable
from grasp.cfg.symbol import Nonterminal, make_span
from grasp.cfg.projection import ItemDerivationYield, DerivationYield
from grasp.recipes import smart_wopen


def report_forest(uid, forest, tsorter, outdir, options):
    """
    Report information about the forest
    :param state: ParserState
    :return:
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


def exact(uid, input, grammars, glue_grammars, options, outdir):
    semiring = SumTimes
    if options.intersection == 'earley':
        parser = Earley(grammars, input.fsa,
                        glue_grammars=glue_grammars,
                        semiring=semiring)
    elif options.intersection == 'nederhof':
        parser = Nederhof(grammars, input.fsa,
                          glue_grammars=glue_grammars,
                          semiring=semiring)
    else:
        raise NotImplementedError("I don't know this intersection algorithm: %s" % options.intersection)

    # make a forest
    logging.info('Parsing...')
    forest = parser.do(root=Nonterminal(options.start), goal=Nonterminal(options.goal))
    if not forest:
        logging.info('[%s] NO PARSE FOUND', uid)
        return

    tsorter = LazyTopSortTable(forest)

    # report info if necessary
    report_forest(uid, forest, tsorter, outdir, options)

    # decoding strategies

    if options.viterbi:
        tsort = tsorter.do()
        logging.info('Viterbi...')
        d = viterbi_derivation(forest, tsort, generations=options.generations)
        logging.info('Viterbi derivation: %s %s', derivation_value(d), DerivationYield.derivation(d))
        save_viterbi('{0}/viterbi/{1}.gz'.format(outdir, uid),
                     d,
                     omega_d=derivation_value,
                     get_projection=DerivationYield.tree)

    if options.kbest > 0:
        root = make_span(Nonterminal(options.goal))  # this is the root after intersection
        logging.info('K-best...')
        kbestparser = KBest(forest,
                            root,
                            options.kbest,
                            MaxTimes,
                            traversal=ItemDerivationYield.string,
                            uniqueness=False).do()
        logging.info('Done!')
        derivations = list(kbestparser.iterderivations())
        save_kbest('{0}/kbest/{1}.gz'.format(outdir, uid),
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
        save_mc_derivations('{0}/ancestral/derivations/{1}.gz'.format(outdir, uid),
                            derivations,
                            inside=sampler.Z,
                            omega_d=derivation_value)
        # save the empirical distribution over strings
        save_mc_yields('{0}/ancestral/trees/{1}.gz'.format(outdir, uid),
                       trees)

    logging.info('[%s] Finished!', uid)