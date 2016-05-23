"""
:Authors: - Wilker Aziz
"""
import argparse
import sys
import logging
from multiprocessing import Pool
from functools import partial
import numpy as np

import grasp.ptypes as ptypes
from grasp.recipes import smart_ropen
from grasp.recipes import traceit
import grasp.semiring as semiring
from grasp.formal.topsort import AcyclicTopSortTable
from grasp.formal.traversal import bracketed_string
from grasp.formal.traversal import yield_string
from grasp.formal.wfunc import TableLookupFunction
from grasp.formal.wfunc import derivation_weight
from grasp.alg.rescoring import stateless_rescoring
from grasp.alg.rescoring import score_derivation
from grasp.alg.inference import AncestralSampler
from grasp.alg.chain import group_by_identity
from grasp.alg.chain import group_by_projection
from grasp.scoring.frepr import FComponents
from grasp.scoring.util import construct_extractors
from grasp.scoring.util import read_weights
from grasp.scoring.util import make_weight_map
from grasp.scoring.util import make_models
from grasp.scoring.util import InitialWeightFunction
from grasp.scoring.scorer import TableLookupScorer
from grasp.scoring.scorer import StatelessScorer
from grasp.scoring.scorer import StatefulScorer
from grasp.mt.util import GoalRuleMaker
from grasp.mt.util import make_dead_oview
from grasp.alg.impsamp import ISDerivation, ISYield
from grasp.loss.fast_bleu import DecodingBLEU


import grasp.mt.pipeline as pipeline


def cmd_grammar(group):
    group.add_argument('--start', '-S',
                       type=str, default='S',
                       metavar='LABEL',
                       help='default start symbol')
    group.add_argument('--extra-grammar',
                       action='append', default=[], metavar='PATH',
                       help="path to an additional grammar (multiple allowed)")
    group.add_argument('--glue-grammar',
                       action='append', default=[], metavar='PATH',
                       help="glue rules are only applied to initial states (multiple allowed)")
    group.add_argument('--pass-through',
                       action='store_true',
                       help="add pass-through rules for every input word (and an indicator feature for unknown words)")
    group.add_argument('--default-symbol', '-X',
                       type=str, default='X', metavar='LABEL',
                       help='default nonterminal (used for pass-through rules and automatic glue rules)')


def cmd_parser(group):
    group.add_argument('--goal',
                       type=str, default='GOAL', metavar='LABEL',
                       help='default goal symbol (root after parsing/intersection)')
    group.add_argument('--max-span',
            type=int, default=-1, metavar='N',
            help='size of the longest input path under an X nonterminal (a negative value implies no constraint)')


def cmd_sampling(group):
    group.add_argument('--temperature',
                       type=float, default=1.0,
                       help='peak (0 < t < 1.0) or flatten (t > 1.0) the distribution')
    group.add_argument('--samples',
                       type=int, default=0, metavar='N',
                       help='number of samples')


def argparser():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(prog='impsamp')

    parser.description = 'MT decoding by importance sampler'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument("proxy",
                        type=str,
                        help="path to proxy model")
    parser.add_argument("target",
                        type=str,
                        help="path to target model description")
    parser.add_argument("--proxy-weights", '-Q',
                        type=str,
                        help="path to proxy model weights")
    parser.add_argument("--target-weights", '-P',
                        type=str,
                        help="path to target model weights")
    parser.add_argument('input', nargs='?',
            type=argparse.FileType('r'), default=sys.stdin,
            help='input corpus (one sentence per line)')
    parser.add_argument('--cpus',
                        type=int, default=1,
                        help='number of cpus available')
    parser.add_argument("--grammars",
                        type=str,
                        help="where to find grammars (grammar files are expected to be named grammar.$i.sgm, "
                             "with $i 0-based)")
    parser.add_argument('--verbose', '-v',
                        action='count', default=0,
                        help='increase the verbosity level')

    cmd_grammar(parser.add_argument_group('Grammar'))
    cmd_parser(parser.add_argument_group('Parser'))
    cmd_sampling(parser.add_argument_group('Sampling'))

    return parser




def decide(samples, n_samples, proxy, target):
    # support
    Y = [None] * len(samples)
    # posterior
    Q = np.zeros(len(samples), dtype=ptypes.weight)
    P = np.zeros(len(samples), dtype=ptypes.weight)
    # compute posterior
    for i, sample in enumerate(samples):
        Y[i] = sample.y.split()
        D = sample.D
        qy = 0.0
        py = 0.0
        #py = semiring.inside.zero
        for d in D:
            f = target.score(d.p_comps)
            g = proxy.score(d.q_comps)  # TODO: consider normalising g exactly
            w = semiring.inside.divide(f, g)
            qy += float(d.count) / n_samples
            py += d.count * semiring.inside.as_real(w)
            #py = semiring.inside.plus(semiring.inside.times(semiring.inside.from_real(d.count), w), py)
        #P[i] = semiring.inside.as_real(py)
        Q[i] = qy
        P[i] = py
    P /= P.sum()
    # compute consensus loss
    bleu = DecodingBLEU(Y, P, max_order=4, smoothing=0.1)
    L = [bleu.loss(y) for y in Y]
    ranking = sorted(range(len(Y)), key=lambda i: (L[i], -P[i], -Q[i]))

    return [(L[i], P[i], Q[i], samples[i].y)for i in ranking]


@traceit
def decode(seg, args, proxy, target):
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

    if not proxy.stateful:
        tgt_forest, lookup_comps, stateless_comps = pipeline.pass1(seg,
                                                                   src_forest,
                                                                   proxy,
                                                                   saving={},
                                                                   redo=True,
                                                                   log=logging.info)
        q_components = [FComponents([comp1, comp2]) for comp1, comp2 in zip(lookup_comps, stateless_comps)]
    else:
        tgt_forest = pipeline.make_target_forest(src_forest)
        goal_maker = GoalRuleMaker(goal_str=args.goal, start_str=args.start, n=1)
        tgt_forest, q_components = pipeline.pass2(seg,
                                                  tgt_forest,
                                                  TableLookupScorer(proxy.lookup),
                                                  StatelessScorer(proxy.stateless),
                                                  StatefulScorer(proxy.stateful),
                                                  goal_rule=goal_maker.get_oview(), omega=None,
                                                  saving={}, redo=True, log=logging.info)
    # TODO: save tgt_forest and q_components
    # Make unnormalised q(d)
    q_func = TableLookupFunction(np.array([proxy.score(comps) for comps in q_components], dtype=ptypes.weight))

    logging.info('[%d] Forest: nodes=%d edges=%d', seg.id, tgt_forest.n_nodes(), tgt_forest.n_edges())
    tsort = AcyclicTopSortTable(tgt_forest)

    sampler = AncestralSampler(tgt_forest, tsort, omega=q_func)
    samples = sampler.sample(args.samples)
    n_samples = len(samples)

    d_groups = group_by_identity(samples)
    y_groups = group_by_projection(d_groups, lambda group: yield_string(tgt_forest, group.key))

    is_yields = []
    for y_group in y_groups:
        y = y_group.key
        is_derivations = []
        for d_group in y_group.values:
            edges = d_group.key
            # reduce q weights through inside.times
            q_score = derivation_weight(tgt_forest, edges, semiring.inside, omega=q_func)
            # reduce q components through inside.times
            q_comps = proxy.constant(semiring.inside.one)
            for e in edges:
                q_comps = q_comps.hadamard(q_components[e], semiring.inside.times)
            # compute p components and p score
            p_comps, p_score = score_derivation(tgt_forest, edges, semiring.inside,
                                                TableLookupScorer(target.lookup),
                                                StatelessScorer(target.stateless),
                                                StatefulScorer(target.stateful))
            # TODO: save {y => {edges: (q_comps, p_comps, count)}}
            is_derivations.append(ISDerivation(edges, q_comps, p_comps, d_group.count))
        is_yields.append(ISYield(y, is_derivations, y_group.count))
    # TODO: pickle pickling
    return decide(is_yields, n_samples, proxy, target)



def core(args):
    proxy_extractors = construct_extractors(args.proxy)
    if args.proxy_weights:
        proxy_wmap = read_weights(args.proxy_weights)
    else:
        proxy_wmap = make_weight_map(proxy_extractors, InitialWeightFunction.constant(2.0))

    target_extractors = construct_extractors(args.target)
    if args.target_weights:
        target_wmap = read_weights(args.target_weights)
    else:
        target_wmap = make_weight_map(target_extractors, InitialWeightFunction.constant(2.0))

    proxy = make_models(proxy_wmap, proxy_extractors)
    target = make_models(target_wmap, target_extractors)

    segments = pipeline.read_segments_from_stream(args.input, grammar_dir=args.grammars)
    args.input = None  # necessary because we cannot pickle the input stream (TODO: get rid of this ugly thing!)

    with Pool(args.cpus) as workers:
        results = workers.map(partial(decode,
                                      args=args,
                                      proxy=proxy,
                                      target=target), segments)

    for seg, result in zip(segments, results):
        print(seg.id)
        for l, p, q, y in result:
            print('{0} ||| {1} ||| {2} ||| {3}'.format(l, p, q, y))
        print()


def main():
    args = argparser().parse_args()
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    core(args)


if __name__ == '__main__':
    main()