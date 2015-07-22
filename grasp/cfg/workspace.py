"""
Create the workspace tree for the CFG parser.

:Authors: - Wilker Aziz
"""

import os
from grasp.recipes import make_unique_directory


def make_dirs(args, exist_ok=True):
    """
    Make output directories and saves the command line arguments for documentation purpose.

    :param args: command line arguments
    :return: main output directory within workspace (prefix is a timestamp and suffix is a unique random string)
    """

    # create the workspace if missing
    if not os.path.exists(args.workspace):
        os.makedirs(args.workspace)

    # create a unique experiment area or reuse a given one
    if not args.experiment:
        outdir = make_unique_directory(args.workspace)
    else:
        outdir = '{0}/{1}'.format(args.workspace, args.experiment)

    # create output directories for the several inference algorithms
    if args.viterbi:
        os.makedirs('{0}/viterbi'.format(outdir), exist_ok=exist_ok)
    if args.kbest > 0:
        os.makedirs('{0}/kbest'.format(outdir), exist_ok=exist_ok)
    if args.samples > 0:
        if args.framework == 'exact':
            os.makedirs('{0}/ancestral'.format(outdir), exist_ok=exist_ok)
            os.makedirs('{0}/ancestral/derivations'.format(outdir), exist_ok=exist_ok)
            os.makedirs('{0}/ancestral/trees'.format(outdir), exist_ok=exist_ok)
        elif args.framework == 'slice':
            os.makedirs('{0}/slice'.format(outdir), exist_ok=exist_ok)
            os.makedirs('{0}/slice/derivations'.format(outdir), exist_ok=exist_ok)
            os.makedirs('{0}/slice/trees'.format(outdir), exist_ok=exist_ok)
            if args.save_chain:
                os.makedirs('{0}/slice/chain'.format(outdir), exist_ok=exist_ok)
        elif args.framework == 'gibbs':
            os.makedirs('{0}/gibbs'.format(outdir), exist_ok=exist_ok)
    if args.forest:
        os.makedirs('{0}/forest'.format(outdir), exist_ok=exist_ok)
    if args.count:
        os.makedirs('{0}/count'.format(outdir), exist_ok=exist_ok)

    # write the command line arguments to an ini file
    args_ini = '{0}/args.ini'.format(outdir)

    with open(args_ini, 'w') as fo:
        for k, v in sorted(vars(args).items()):
            print('{0}={1}'.format(k,repr(v)),file=fo)

    return outdir, args_ini