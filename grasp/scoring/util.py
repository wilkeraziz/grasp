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
from collections import defaultdict


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


def save_model(model: ModelView, path):
    with open(path, 'w') as fo:
        for extractor in model.extractors():
            print(extractor.cfg(), file=fo)


def compare_models(model: ModelView, path):
    my_cfg = set()
    for extractor in model.extractors():
        my_cfg.add(extractor.cfg())
    other_cfg = set()
    with open(path, 'r') as fi:
        for line in fi:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            other_cfg.add(line)
    return my_cfg == other_cfg


def save_factorisation(joint_model: ModelView, conditional_model: ModelView, path):
    with open(path, 'w') as fo:
        print('[joint]', file=fo)
        print('local=%s' % ' '.join(extractor.name for extractor in joint_model.local_model().extractors()), file=fo)
        print('nonlocal=%s' % ' '.join(extractor.name for extractor in joint_model.nonlocal_model().extractors()),
              file=fo)
        print(file=fo)
        print('[conditional]', file=fo)
        print('local=%s' % ' '.join(extractor.name for extractor in conditional_model.local_model().extractors()),
              file=fo)
        print('nonlocal=%s' % ' '.join(extractor.name for extractor in conditional_model.nonlocal_model().extractors()),
              file=fo)


def read_factorisation(path):
    """
    Return a joint and a conditional factorisation of the model.
    :param path: path to a file with the complete factorisation of a model
    """
    joint_cfg = defaultdict(set)
    conditional_cfg = defaultdict(set)
    if path:
        with smart_ropen(path) as fi:
            changes = None
            for line in fi:
                line = line.strip()
                if not line or line.startswith('#'):  # ignore comments and empty lines
                    continue
                if line == '[joint]':
                    changes = joint_cfg
                elif line == '[conditional]':
                    changes = conditional_cfg
                elif changes is None:
                    raise ValueError('Syntax error in factorisation file')
                elif line.startswith('local='):
                    names = line.replace('local=', '', 1)
                    changes['local'].update(names.split())
                elif line.startswith('nonlocal='):
                    names = line.replace('nonlocal=', '', 1)
                    changes['nonlocal'].update(names.split())

    return joint_cfg, conditional_cfg


def compare_factorisations(joint_model: ModelView, conditional_model: ModelView, path):
    joint_local = set(extractor.name for extractor in joint_model.local_model().extractors())
    joint_nonlocal = set(extractor.name for extractor in joint_model.nonlocal_model().extractors())
    conditional_local = set(extractor.name for extractor in conditional_model.local_model().extractors())
    conditional_nonlocal = set(extractor.name for extractor in conditional_model.nonlocal_model().extractors())

    other_joint, other_conditional = read_factorisation(path)

    joint_status = joint_local == other_joint['local'] and joint_nonlocal == other_joint['nonlocal']
    conditional_status = conditional_local == other_conditional['local'] and conditional_nonlocal == other_conditional[
        'nonlocal']
    return joint_status and conditional_status