"""
:Authors: - Wilker Aziz
"""

import time
import sys
import tempfile
import gzip
import warnings
import pickle
from io import TextIOWrapper
from functools import wraps
from glob import glob
from os.path import basename, splitext
import logging
from datetime import datetime
import traceback
import re
import os


def symlink(path_from, path_to):
    """
    Create a (forced) symlink between abspath(from) and abspath(to).
    """
    i_path = os.path.abspath(path_from)
    o_path = os.path.abspath(path_to)
    if os.path.exists(o_path):
        os.remove(o_path)
    os.symlink(i_path, o_path)


def dummyfunc(*args, **kwargs):
    pass


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
    return tempfile.mkdtemp(prefix=datetime.now().strftime("%y%m%d_%H%M%S_"), dir=dir)


def list_numbered_files(basedir, suffix='', sort=True, reverse=False):
    paths = glob('{0}/[0-9]*{1}'.format(basedir, suffix))
    ids = [int(splitext(basename(path))[0]) for path in paths]
    if not sort:
        return zip(ids, paths)
    else:
        return sorted(zip(ids, paths), reverse=reverse)


def traceit(func):
    @wraps(func)
    def newfunc(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            raise Exception(''.join(traceback.format_exception(*sys.exc_info())))
    return newfunc


def re_collect_groups(re_match, groups=[], repl=''):
    """
    Collect groups for re.sub
    :param re_match: re match object
    :param groups: a list used to return the groups matched
    :param repl: replacement string
    :return: repl string
    """
    groups.extend(re_match.groups())
    return repl


def re_sub(pattern, repl, string):
    """
    Wraps a call to re.sub in order to return both the resulting string and the matched groups.
    :param pattern:
    :param repl: a replacement string
    :param string:
    :return: the resulting string, matched groups
    """
    groups = []
    result = re.sub(pattern, lambda m: re_collect_groups(m, groups, repl), string)
    return result, groups


def re_key_value(key, string, separator='=', repl='', optional=True):
    """
    Matches a key-value pair and replaces it by a given string.
    :param key:
    :param string:
    :param separator: separator of the key-value pair
    :param optional: if False, raises an exception in case no matches are found
    :return: resulting string, value (or None)
    """
    result, groups = re_sub(r'{0}{1}([^ ]+)'.format(key, separator), '', string)
    if not optional and not groups:
        raise ValueError('Expected a key-value pair of the kind {0}{1}<value>: {2}'.format(key, separator, string))
    if groups:
        return result, groups[0]
    else:
        return result, None

