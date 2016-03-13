"""
:Authors: - Wilker Aziz
"""

import time
import sys
import tempfile
import datetime
import gzip
import warnings
import pickle
from io import TextIOWrapper
from functools import wraps
from glob import glob
from os.path import basename, splitext


def pickle_it(path, obj):
    """Dump a pickled representation of the object to disk"""
    with open(path, 'wb') as fo:
        pickle.dump(obj, fo)


def unpickle_it(path):
    """Load an object from a pickled representation stored on disk"""
    with open(path, 'rb') as fo:
        return pickle.load(fo)


def smart_ropen(path):
    """Opens files directly or through gzip depending on extension."""
    if path.endswith('.gz'):
        return TextIOWrapper(gzip.open(path, 'rb'))
    else:
        return open(path, 'r')


def smart_wopen(path):
    """Opens files directly or through gzip depending on extension."""
    if path.endswith('.gz'):
        return TextIOWrapper(gzip.open(path, 'wb'))
    else:
        return open(path, 'w')


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    Recipe from:
        https://wiki.python.org/moin/PythonDecoratorLibrary#Generating_Deprecation_Warnings
    """

    @wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn_explicit(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            filename=func.__code__.co_filename,
            lineno=func.__code__.co_firstlineno + 1
        )
        return func(*args, **kwargs)
    return new_func


def timeit(func):
    @wraps(func)
    def newfunc(*args, **kwargs):
        t0 = time.time()
        r = func(*args, **kwargs)
        delta = time.time() - t0
        return delta, r
    return newfunc


def progressbar(it, count=None, prefix='', dynsuffix=lambda: '', size=60, file=sys.stderr):
    """
    A recipe for a progressbar.

        http://code.activestate.com/recipes/576986-progress-bar-for-console-programs-as-iterator/
    :param it:
    :param count:
    :param prefix:
    :param size:
    :param file:
    :return:
    """

    if count is None:
        count = len(it)

    def _show(_i):
        x = int(size * _i / count)
        file.write('%s[%s%s] %i/%i%s\r' % (prefix, '#'*x, '.'*(size - x), _i, count, dynsuffix()))
        file.flush()

    _show(0)
    for i, item in enumerate(it):
        yield item
        _show(i+1)
    file.write('\n')
    file.flush()


def make_unique_directory(dir=None):
    return tempfile.mkdtemp(prefix=datetime.datetime.now().strftime("%y%m%d_%H%M%S_"), dir=dir)


def list_numbered_files(basedir, suffix='', sort=True, reverse=False):
    paths = glob('{0}/[0-9]*{1}'.format(basedir, suffix))
    ids = [int(splitext(basename(path))[0]) for path in paths]
    if not sort:
        return zip(ids, paths)
    else:
        return sorted(zip(ids, paths), reverse=reverse)