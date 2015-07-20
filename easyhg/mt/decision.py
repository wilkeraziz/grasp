"""
:Authors: - Wilker Aziz
"""

import argparse
import logging
import numpy as np
import os
import sys
import random
from functools import partial
from multiprocessing import Pool
from easyhg.mteval.bleu import BLEU
from easyhg.recipes import list_numbered_files, smart_ropen
import traceback


def _make_dirs(args, exist_ok=True):
    os.makedirs(args.decisions, exist_ok=exist_ok)
    if args.kbest > 1:
        if args.map:
            outdir = '{0}/MAP'.format(args.decisions)
            os.makedirs(outdir, exist_ok=exist_ok)
        if args.mbr:
            outdir = '{0}/MBR-{1}'.format(args.decisions, args.metric)
            os.makedirs(outdir, exist_ok=exist_ok)
        if args.consensus:
            outdir = '{0}/consensus-{1}'.format(args.decisions, args.metric)
            os.makedirs(outdir, exist_ok=exist_ok)
    return outdir


def read_empirical_distribution(path):
    """
    Return the empirical distribution (a numpy array) and the support (tuples).
    :param path: path of distribution over projections
    :return:
    """
    Y = []
    P = []
    with smart_ropen(path) as fi:
        lines = fi.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            line = line.strip()
            if not line:
                continue
            fields = line.split('\t')
            if len(fields) < 2:
                raise ValueError('Bad format: I expected the first column to be an estimate and the last to be the solution.')
            estimate = float(fields[0])
            projection = tuple(fields[-1].split())
            Y.append(projection)
            P.append(estimate)
    return np.array(P), tuple(Y)


def MAP(P, Y, metric=None):
    """
    Sort solutions according to the MAP decision rule.
    :param P: probabilities
    :param Y: support
    :return: sorted list of triplets (loss, probability, solution)
    """
    return [1 - p for p in P]


def MBR(P, Y, metric):
    """
    :param empdist: for now MBR assumes Yh == Ye (see above)
    :param metric:
    :param normalise:
    :return:
    """
    M = len(P)
    losses = np.zeros(M)
    for hid, hyp in enumerate(Y):
        for rid, (p_ref, ref) in enumerate(zip(P, Y)):
            score = metric.loss(c=hid, r=rid)
            losses[hid] += score * p_ref
    return losses


def consensus(P, Y, metric):
    return [metric.coloss(c=i) for i, y in enumerate(Y)]


def traced_decide(job, rule, metric):
    try:
        segid, path = job
        P, Y = read_empirical_distribution(path)
        metric.prepare_decoding(Y, P)
        if rule == 'map':
            losses = MAP(P, Y, metric)
        elif rule == 'mbr':
            losses = MBR(P, Y, metric)
        elif rule == 'consensus':
            losses = consensus(P, Y, metric)
        ranking = sorted(zip(losses, P, Y))
        metric.cleanup()
        return segid, ranking
    except:
        raise Exception(''.join(traceback.format_exception(*sys.exc_info())))


def main(args):
    if args.metric == 'bleu':
        metric = BLEU()
    else:
        raise NotImplementedError('I do not yet know an implementation of %s' % args.metric)

    jobs = list_numbered_files(args.samples, '.gz', sort=True)
    if args.shuffle:
        random.shuffle(jobs)

    pool = Pool(args.cpus)
    results = pool.map(partial(traced_decide,
                               rule=args.rule,
                               metric=metric), jobs)

    with open('{0}.output'.format(args.decisions), 'w') as fo:
        with open('{0}.log'.format(args.decisions), 'w') as fe:
            for segid, solutions in sorted(results):
                loss, prob, projection = solutions[0]
                proj_str = ' '.join(projection)
                print(proj_str, file=fo)
                print('# segment\tloss\tprobability\tyield', file=fe)
                for solution in solutions:
                    print('{0}\t{1}\t{2}\t{3}'.format(segid, solution[0], solution[1], ' '.join(solution[2])), file=fe)
                print(file=fe)


def configure():
    parser = argparse.ArgumentParser(description='Applies a decision rule to a sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("samples",
                        type=str,
                        help="where samples can be found")
    parser.add_argument("decisions",
                        type=str, default=None,
                        help="where decisions should be stored")
    parser.add_argument("--rule",
                        type=str, default='map', choices=['map', 'mbr', 'consensus'],
                        help="decision rule")
    parser.add_argument("--metric", '-m',
                        type=str, default='bleu', choices=['bleu'],
                        help="similarity function")
    #parser.add_argument("--kbest", '-k',
    #                    type=int, default=1,
    #                    help="number of solutions")
    parser.add_argument("--cpus",
                        type=int, default=2,
                        help="number of cpus available")
    parser.add_argument("--shuffle",
                        action='store_true',
                        help="shuffle input segments")
    parser.add_argument('--verbose', '-v',
                        action='count', default=0,
                        help='increases verbosity')

    args = parser.parse_args()

    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
    elif args.verbose > 1:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

    return args

if __name__ == '__main__':
    main(configure())