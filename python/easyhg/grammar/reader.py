"""

:Authors: Wilker Aziz
"""


import numpy as np
import ply_cfg
import discodopfmt


def load_grammar(path, grammarfmt, log_transform):
    """
    Read a CFG from files encoded in a specific format.

    :param path: path to a grammar.
        If grammarfmt == 'bar', this is the exact path.
        If grammarfmt == 'discodop', this is a prefix to files containing the rules and the lexicon.
    :param grammarfmt: a known grammar format.
    :param log_transform: whether or not we should apply the log transform.
    :return: a CFG
    """

    if log_transform:
        transform = np.log
    else:
        transform = float
    if grammarfmt == 'bar':
        cfg = ply_cfg.read_grammar(path, transform=transform)
    elif grammarfmt == 'discodop':
        cfg = discodopfmt.read_grammar('{0}.rules.gz'.format(path),
                                       '{0}.lex.gz'.format(path),
                                       transform=transform)
    else:
        raise NotImplementedError("I don't know this grammar format: %s" % grammarfmt)
    return cfg
