"""
A simple command line tool for computing IBM BLEU.

:Authors: - Wilker Aziz
"""
import argparse
import sys
import os
from grasp.loss.fast_bleu import stream_doc_bleu
from grasp.recipes import smart_ropen
import numpy as np


def argparser():
    """parse command line arguments"""

    parser = argparse.ArgumentParser(prog='bleu')

    parser.description = 'BLEU (Papineni et al, 2002)'
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument("references",
                        type=str,
                        help="path to reference sets (one set per line, each reference is separated by 3 bars |||)"
                             " - may be gzipped")
    parser.add_argument('hypotheses', nargs='?', type=str, default=None,
                        help='input corpus (by default read from stdin) - may be gzipped')
    parser.add_argument('--order',
                        type=int, default=4,
                        help='longest n-gram')
    parser.add_argument("--smoothing",
                        type=float, default=0.0,
                        help="stupid add-n smoothing")

    return parser


def main():
    args = argparser().parse_args()
    if args.hypotheses:
        if not os.path.exists(args.hypotheses):
            raise FileNotFoundError('Hypotheses file not found: %s' % args.hypotheses)
        hstream = smart_ropen(args.hypotheses).readlines()
    else:
        hstream = sys.stdin.readlines()
    if not os.path.exists(args.references):
        raise FileNotFoundError('Reference file not found: %s' % args.references)
    rstream = smart_ropen(args.references).readlines()

    # compute bleu
    bleu, pn, bp = stream_doc_bleu(hstream, rstream, args.order, args.smoothing)
    print(bleu)

    # log brevity penalty, n-gram precisions, and BLEU-1 to BLEU-order
    print('grasp.mt.bleu loaded %d segments' % len(hstream),
          file=sys.stderr)
    bleus = []
    for max_order in range(1, args.order + 1):
        bleus.append((bp * np.exp(1.0 / max_order * np.sum(np.log(pn[0:max_order])))))
    print('bp=%.4f ||| %s ||| %s' % (bp,
                                     ' '.join('p%d=%.4f' % (i, x) for i, x in enumerate(pn, 1)),
                                     ' '.join('bleu-%d=%.4f' % (i, x) for i, x in enumerate(bleus, 1))
                                     ),
          file=sys.stderr)


if __name__ == '__main__':
    main()
