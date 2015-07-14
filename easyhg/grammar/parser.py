"""
This module is an interface for parsing as intersection.
One can choose from all available implementations.

:Authors: - Wilker Aziz
"""

import logging
import os
import sys
from itertools import chain



from easyhg.alg.sliced.sampling import slice_sampling, make_result
from easyhg.alg.sliced.slicevars import Beta, Exponential, get_prior, VectorOfPriors

from .cmdline import argparser
from .sentence import make_sentence
from .semiring import SumTimes
from .reader import load_grammar
from .cfg import CFG, TopSortTable, Nonterminal
from .rule import get_oov_cfg_productions
from .utils import make_unique_directory
from .report import save_kbest, save_viterbi, save_mc, save_mcmc, save_sample_history
from .exact import exact




def report_info(cfg, args):
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


def do(uid, input, grammars, glue_grammars, options, outdir):

    if options.framework == 'exact':



        results_by_method = exact(uid, input, grammars, glue_grammars, options, outdir)

        if 'viterbi' in results_by_method:
            save_viterbi('{0}/viterbi/{1}.gz'.format(outdir, uid), results_by_method['viterbi'])

        if 'kbest' in results_by_method:
            save_kbest('{0}/kbest/{1}.gz'.format(outdir, uid), results_by_method['kbest'])

        if 'ancestral' in results_by_method:
            save_mc('{0}/ancestral/{1}.gz'.format(outdir, uid), results_by_method['ancestral'])

    elif options.framework == 'slice':

        if options.free_dist == 'beta':
            dist = Beta
            prior = VectorOfPriors(get_prior(options.prior_a[0], options.prior_a[1]),
                                   get_prior(options.prior_b[0], options.prior_b[1]))
        elif options.free_dist == 'exponential':
            dist = Exponential
            prior = get_prior(options.prior_scale[0], options.prior_scale[1])

        history = slice_sampling(input.fsa,
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
                                 free_dist_prior=prior)

        results = make_result(history, lag=options.lag, burn=options.burn, resample=options.resample)
        save_mcmc('{0}/slice/{1}.gz'.format(outdir, uid), results)
        if options.history:
            save_sample_history('{0}/history/{1}.gz'.format(outdir, uid), history)
    else:
        raise NotImplementedError('I do not yet know how to perform inference in this framework: %s' % options.framework)


def make_dirs(args):
    """
    Make output directories and saves the command line arguments for documentation purpose.

    :param args: command line arguments
    :return: main output directory within workspace (prefix is a timestamp and suffix is a unique random string)
    """

    # create the workspace if missing
    logging.info('Workspace: %s', args.workspace)
    if not os.path.exists(args.workspace):
        os.makedirs(args.workspace)

    # create a unique experiment area or reuse a given one
    if not args.experiment:
        outdir = make_unique_directory(args.workspace)
    else:
        outdir = '{0}/{1}'.format(args.workspace, args.experiment)
    logging.info('Writing files to: %s', outdir)

    # create output directories for the several inference algorithms
    if args.viterbi:
        os.makedirs('{0}/viterbi'.format(outdir))
    if args.kbest > 0:
        os.makedirs('{0}/kbest'.format(outdir))
    if args.samples > 0:
        if args.framework == 'exact':
            os.makedirs('{0}/ancestral'.format(outdir))
        elif args.framework == 'slice':
            os.makedirs('{0}/slice'.format(outdir))
        elif args.framework == 'gibbs':
            os.makedirs('{0}/gibbs'.format(outdir))
    if args.forest:
        os.makedirs('{0}/forest'.format(outdir))
    if args.count:
        os.makedirs('{0}/count'.format(outdir))
    if args.history:
        os.makedirs('{0}/history'.format(outdir))

    # write the command line arguments to an ini file
    args_ini = '{0}/args.ini'.format(outdir)
    logging.info('Writing command line arguments to: %s', args_ini)
    with open(args_ini, 'w') as fo:
        for k, v in sorted(vars(args).items()):
            print('{0}={1}'.format(k,repr(v)),file=fo)

    return outdir


def core(args):

    semiring = SumTimes

    # Load main grammars
    logging.info('Loading main grammar...')
    cfg = load_grammar(args.grammar, args.grammarfmt, args.log)
    logging.info('Main grammar: terminals=%d nonterminals=%d productions=%d', cfg.n_terminals(), cfg.n_nonterminals(), len(cfg))

    # Load additional grammars
    main_grammars = [cfg]
    if args.extra_grammar:
        for grammar_path in args.extra_grammar:
            logging.info('Loading additional grammar: %s', grammar_path)
            grammar = load_grammar(grammar_path, args.grammarfmt, args.log)
            logging.info('Additional grammar: terminals=%d nonterminals=%d productions=%d', grammar.n_terminals(), grammar.n_nonterminals(), len(grammar))
            main_grammars.append(grammar)

    # Load glue grammars
    glue_grammars = []
    if args.glue_grammar:
        for glue_path in args.glue_grammar:
            logging.info('Loading glue grammar: %s', glue_path)
            glue = load_grammar(glue_path, args.grammarfmt, args.log)
            logging.info('Glue grammar: terminals=%d nonterminals=%d productions=%d', glue.n_terminals(), glue.n_nonterminals(), len(glue))
            glue_grammars.append(glue)

    # Report information about the main grammar
    report_info(cfg, args)

    # Make surface lexicon
    surface_lexicon = set()
    for grammar in chain(main_grammars, glue_grammars):
        surface_lexicon.update(t.surface for t in grammar.iterterminals())

    # Prepare output directories
    outdir = make_dirs(args)

    # Parse sentence by sentence
    viterbi_solutions = []
    for i, input_str in enumerate(args.input):
        # get an input automaton
        sentence = make_sentence(input_str, semiring, surface_lexicon, args.unkmodel)
        grammars = list(main_grammars)

        if args.unkmodel == 'passthrough':
            grammars.append(CFG(get_oov_cfg_productions(sentence.oovs, args.unklhs, semiring.one)))

        logging.info('Parsing %d words: %s', len(sentence), sentence)

        do(i, sentence, grammars, glue_grammars, args, outdir)

    logging.info('Check output files in: %s', outdir)


def configure():
    """
    Parse command line arguments, configures the main logger.
    :returns: command line arguments
    """

    args = argparser().parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

    return args


def main():
    """
    Configures the parser by parsing command line arguments and calling the core code.
    It might also profile the run if the user chose to do so.
    """
    args = configure()

    if args.profile:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
        core(args)
        pr.disable()
        pr.dump_stats(args.profile)
    else:
        core(args)




if __name__ == '__main__':
    main()
