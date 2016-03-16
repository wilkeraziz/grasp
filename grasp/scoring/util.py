"""
:Authors: - Wilker Aziz
"""
from collections import defaultdict
from grasp.recipes import smart_ropen
import numpy as np

def cdec_basic():
    return dict(EgivenFCoherent=1.0,
                SampleCountF=1.0,
                CountEF=1.0,
                MaxLexFgivenE=1.0,
                MaxLexEgivenF=1.0,
                IsSingletonF=1.0,
                IsSingletonFE=1.0,
                Glue=1.0)


def read_weights(path, default=None, random=False, temperature=1.0, u=0, std=0.01):
    """
    Read a sequence of key-value pairs.
    :param path: file where to read sequence from
    :param default: if set, overwrites the values read from file
    :param random: if set, sample values from N(u, std)
    :param temperature: scales the final weight: weight/T
    :param u: mean of normal
    :param std: standard deviation
    :return:
    """
    wmap = {}
    with smart_ropen(path) as fi:
        for line in fi.readlines():
            fields = line.split()
            if len(fields) != 2:
                continue
            w = float(fields[1])
            if default is not None:
                w = default
            elif random:
                w = np.random.normal(u, std)
            w /= temperature
            wmap[fields[0]] = w
    return wmap

