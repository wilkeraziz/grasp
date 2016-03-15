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


def read_weights(path, temperature=1.0, default=None, random=False, u=0, std=0.1):
    wmap = {}
    with smart_ropen(path) as fi:
        for line in fi.readlines():
            fields = line.split()
            if len(fields) != 2:
                continue
            if not random:
                if default is not None:
                    wmap[fields[0]] = default
                else:
                    wmap[fields[0]] = float(fields[1]) / temperature
            else:
                wmap[fields[0]] = np.random.normal(u, std)
    return wmap

