"""
Command line interface for hierarchical MT decoding.

:Authors: - Wilker Aziz
"""

import logging
import sys
import random
from itertools import chain
from multiprocessing import Pool
from functools import partial
import traceback
from types import SimpleNamespace

from grasp.mt.pipeline import read_segments_from_stream
from grasp.recipes import traceit

from grasp.io.report import EmptyReport, IterationReport
from grasp.io.results import save_kbest, save_viterbi
from grasp.io.results import save_mc_derivations, save_mc_yields
from grasp.io.results import save_mcmc_yields, save_mcmc_derivations, save_markov_chain

from grasp.mt.cmdline import argparser
from grasp.mt.workspace import make_dirs
from grasp.mt.util import GoalRuleMaker
from grasp.mt.util import make_dead_oview
import grasp.mt.pipeline as pipeline
from grasp.recipes import dummyfunc

from grasp.recipes import timeit, smart_wopen

from grasp.scoring.scorer import StatefulScorer, StatelessScorer, TableLookupScorer
from grasp.scoring.util import make_models
from grasp.scoring.util import read_weights
from grasp.scoring.util import construct_extractors
from grasp.scoring.frepr import FComponents

import grasp.semiring as semiring

from grasp.formal.topsort import AcyclicTopSortTable
from grasp.formal.traversal import bracketed_string, yield_string

from grasp.alg.rescoring import stateless_rescoring
from grasp.alg.deduction import EarleyRescorer
from grasp.alg.inference import viterbi_derivation, AncestralSampler
from grasp.alg.chain import group_by_projection, group_by_identity, apply_batch_filters, apply_filters
from grasp.formal.wfunc import derivation_weight, HypergraphLookupFunction


def save_forest(hg, outdir, sid, name):
    with smart_wopen('{0}/forest/{1}.{2}.gz'.format(outdir, name, sid)) as fo:
            print('# FOREST', file=fo)
            print(hg, file=fo)


def viterbi(sid, hg, tsort, outdir, round):
    edges = viterbi_derivation(hg, tsort)
    score = derivation_weight(hg, edges, semiring.inside)
    logging.info('Viterbi derivation: %s', score)
    save_viterbi('{0}/viterbi/{1}.{2}.gz'.format(outdir, round, sid),
                 SimpleNamespace(derivation=edges),
                 valuefunc=lambda d: derivation_weight(hg, d, semiring.inside),
                 get_projection=lambda d: yield_string(hg, d),
                 derivation2str=lambda d: bracketed_string(hg, d))


@traceit
def decode(seg, args, model, outdir):
    """
    """

    # pass0
    src_forest = pipeline.pass0(seg,
                                extra_grammar_paths=args.extra_grammar,
                                glue_grammar_paths=args.glue_grammar,
                                pass_through=args.pass_through,
                                default_symbol=args.default_symbol,
                                goal_str=args.goal,
                                start_str=args.start,
                                max_span=args.max_span,
                                n_goal=0, log=logging.info)
    tgt_forest = pipeline.make_target_forest(src_forest, TableLookupScorer(model.lookup))
    tsort = AcyclicTopSortTable(tgt_forest)

    if args.viterbi:
        viterbi(seg.id, tgt_forest, tsort, outdir, "pass0")

    # pass1
    if model.stateless:
        tgt_forest = stateless_rescoring(tgt_forest, StatelessScorer(model.stateless), semiring.inside)
        if args.viterbi:
            viterbi(seg.id, tgt_forest, tsort, outdir, "pass1")

    samples = []

    if args.framework == 'exact' or not model.stateful:  # exact scoring or no stateful scoring
        # we have access to Viterbi, k-best, sampling
        if model.stateful:
            goal_maker = GoalRuleMaker(goal_str=args.goal, start_str=args.start, n=1)

            rescorer = EarleyRescorer(tgt_forest,
                                          TableLookupScorer(model.dummy),
                                          StatelessScorer(model.dummy),
                                          StatefulScorer(model.stateful),
                                          semiring.inside)

            tgt_forest = rescorer.do(tsort.root(), goal_maker.get_oview())
            tsort = AcyclicTopSortTable(tgt_forest)

        # Do everything: viterbi, map, consensus, etc...
        if args.viterbi:
            viterbi(seg.id, tgt_forest, tsort, outdir, "pass2")

        if args.kbest > 0:
            # TODO: call kbest code
            pass
        if args.samples > 0:
            sampler = AncestralSampler(tgt_forest, tsort)
            samples = sampler.sample(args.samples)
            derivations = group_by_identity(samples)
            save_mc_derivations('{0}/exact/derivations/{1}.gz'.format(outdir, seg.id),
                                derivations, sampler.Z,
                                valuefunc=lambda d: derivation_weight(tgt_forest, d, semiring.inside),
                                derivation2str=lambda d: bracketed_string(tgt_forest, d))
            projections = group_by_projection(samples, lambda d: yield_string(tgt_forest, d))
            save_mc_yields('{0}/exact/yields/{1}.gz'.format(outdir, seg.id), projections)

            # TODO: fix this hack
            # it's here just so I can reuse pipeline.consensus
            # the fix involves moving SampleReturn to a more general module
            # and making AncestralSampler use it
            from grasp.alg.rescoring import SampleReturn
            samples = [SampleReturn(s, 0.0, FComponents([]))for s in samples]

    else:  # for sliced scoring, we only have access to sampling

        logging.info('Sliced rescoring...')
        from grasp.alg.rescoring2 import SlicedRescoring
        goal_maker = GoalRuleMaker(goal_str=args.goal, start_str=args.start, n=1)
        rescorer = SlicedRescoring(tgt_forest,
                                   HypergraphLookupFunction(tgt_forest),
                                   tsort,
                                   TableLookupScorer(model.dummy),
                                   StatelessScorer(model.dummy),
                                   StatefulScorer(model.stateful),
                                   semiring.inside,
                                   goal_maker.get_oview(),
                                   make_dead_oview(args.default_symbol))

        if args.gamma_shape > 0:
            gamma_shape = args.gamma_shape
        else:
            gamma_shape = len(model)  # number of local components
        gamma_scale_type = args.gamma_scale[0]
        gamma_scale_parameter = float(args.gamma_scale[1])

        # here samples are represented as sequences of edge ids
        d0, markov_chain = rescorer.sample(n_samples=args.samples, batch_size=args.batch, within=args.within,
                                           initial=args.initial,
                                           gamma_shape=gamma_shape,
                                           gamma_scale_type=gamma_scale_type,
                                           gamma_scale_parameter=gamma_scale_parameter,
                                           burn=args.burn, lag=args.lag,
                                           temperature0=args.temperature0)

        # apply usual MCMC heuristics (e.g. burn-in, lag)
        samples = apply_filters(markov_chain,
                                burn=args.burn,
                                lag=args.lag)

        # group by derivation (now a sample is represented by a Derivation object)
        derivations = group_by_identity(samples)
        save_mcmc_derivations('{0}/slice/derivations/{1}.gz'.format(outdir, seg.id),
                              derivations,
                              valuefunc=lambda d: d.score,
                              derivation2str=lambda d: bracketed_string(tgt_forest, d.edges))
        projections = group_by_projection(samples, lambda d: yield_string(tgt_forest, d.edges))
        save_mcmc_yields('{0}/slice/yields/{1}.gz'.format(outdir, seg.id),
                         projections)

        if args.save_chain:
            markov_chain.appendleft(d0)
            save_markov_chain('{0}/slice/chain/{1}.gz'.format(outdir, seg.id),
                              markov_chain,
                              flat=True,
                              valuefunc=lambda d: d.score,
                              derivation2str=lambda d: bracketed_string(tgt_forest, d.edges))

    if samples:
        # decision rule
        decisions = pipeline.consensus(seg, tgt_forest, samples)
        return decisions[0]


#@traceit
#def decode(seg, args, n_samples, decisiondir, model, redo, log=dummyfunc):
#    saving = {
#        'decisions': '{0}/{1}.decisions.gz'.format(decisiondir, seg.id)
#    }
#    return pipeline.decode(seg, args, n_samples, model, saving, redo, log)


def core(args):
    """
    The main pipeline including multiprocessing.
    """

    outdir = make_dirs(args)

     # Load feature extractors
    #extractors = pipeline.load_feature_extractors(rt=args.rt, wp=args.wp, ap=args.ap, slm=args.slm, lm=args.lm)
    extractors = construct_extractors(args.model)
    # Load the model
    model = make_models(read_weights(args.weights,
                                     default=args.default,
                                     temperature=args.temperature),
                        extractors)

    # Load the segments
    segments = read_segments_from_stream(args.input, grammar_dir=args.grammars, shuffle=args.shuffle)

    args.input = None  # necessary because we cannot pickle the input stream (TODO: get rid of this ugly thing!)

    with Pool(args.cpus) as workers:
        results = workers.map(partial(decode,
                                      args=args,
                                      model=model,
                                      outdir=outdir), segments)

    with open('{0}/translations'.format(outdir), 'w') as ft:
        for seg, result in zip(segments, results):
            if result is not None:
                l, p, y = result
                print(y, file=ft)
            else:
                print(file=ft)

    # TODO: reports, MAP decision rule
    # separate functions: sampling (MC, MCMC) from search (viterbi, kbest, cube pruning?)
    #    print(y)
    # time report
    #time_report = IterationReport('{0}/time'.format(outdir))
    #for seg, (dt, dt_detail) in sorted(zip(segments, results), key=lambda seg_info: seg_info[0].id):
    #    time_report.report(seg.id, total=dt, **dt_detail)
    #time_report.save()

    print('Check output files in:', outdir, file=sys.stderr)


def main():
    args = argparser().parse_args()

    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

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
