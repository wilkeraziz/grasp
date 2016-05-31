"""
:Authors: - Wilker Aziz
"""
import numpy as np
import importlib
from grasp.recipes import smart_ropen
from grasp.recipes import smart_wopen
from grasp.recipes import re_sub
from grasp.recipes import re_key_value
from grasp.scoring.extractor import TableLookup, Stateless, Stateful
from grasp.scoring.model import ModelContainer, ModelView


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


def save_weights(path: str, fnames: list, fvalues: list):
    with smart_wopen(path) as fw:
        for fname, fvalue in zip(fnames, fvalues):
            print('{0} {1}'.format(fname, repr(fvalue)), file=fw)


def get_extractor_implementation(cls, pkg=None):
    """
    Get the implementation of an Extractor.
    :param cls: class name (str)
    :param pkg: package where implementation is found (str)
    :return: class implementation
    """
    # known extractors
    from grasp.scoring.lookup import NamedFeature
    from grasp.scoring.lookup import RuleTable
    from grasp.scoring.stateless import WordPenalty
    from grasp.scoring.stateless import ArityPenalty
    from grasp.scoring.lm import StatelessLM
    from grasp.scoring.lm import KenLM
    from grasp.scoring.lm import ConstantLM
    # others must be fully specified through pkg.cls
    if pkg:
        try:
            module = importlib.import_module(pkg)
            impl = getattr(module, cls)
        except:
            raise ValueError('Could not load feature definitions from file %s', pkg)
    else:
        impl = eval(cls)
    return impl


def construct_extractors(path):
    """
    Read a configuration file and construct the extractors specified in each line.
    :param path: path to configuration file
    :return: list of extractors (in the order they were listed in the configuration file)
    """
    extractors = []
    names = set()
    with smart_ropen(path) as fi:
        for i, line in enumerate(fi, 1):
            if line.startswith('#'):
                continue
            line = line.strip()
            if not line:
                continue

            try:
                cfg, [cls] = re_sub('^([^ ]+)', '', line)
            except:
                raise ValueError('In line %d, expected class name: %s' % (i, line))
            cfg, name = re_key_value('name', cfg)
            if not name:
                name = cls
            if name in names:
                raise ValueError('In line %d, duplicate name (%s), rename your extractor with name=<CustomName>' % (i, name))
            names.add(name)
            cfg, pkg = re_key_value('pkg', cfg)
            impl = get_extractor_implementation(cls, pkg)
            extractor = impl.construct(len(extractors), name, cfg)
            extractors.append(extractor)
    return extractors


class InitialWeightFunction:
    """
    Groups a number of weight functions useful for initialisation.
    """

    @staticmethod
    def constant(constant):
        return lambda n: constant

    @staticmethod
    def normal(u=0, std=0.01):
        return lambda n: np.random.normal(u, std)

    @staticmethod
    def uniform(den):
        p = 1.0/den
        return lambda n: p/n


def make_weight_map(extractors, weightfunc):
    wmap = {}
    for ext in extractors:
        features = ext.features()
        for ff in features:
            wmap[ff] = weightfunc(len(features))
    return wmap


def make_models(wmap, extractors: 'list[Extractor]', uniform_weights=False) -> ModelContainer:
    wmap = dict(wmap)
    # all scorers sorted by id
    extractors = tuple(sorted(extractors, key=lambda x: x.id))

    if uniform_weights:
        for extractor in extractors:
            fnames = extractor.fnames(wmap.keys())
            for fname in fnames:
                wmap[fname] = 1.0 / len(extractors) / len(fnames)

    return ModelContainer(wmap, extractors)
