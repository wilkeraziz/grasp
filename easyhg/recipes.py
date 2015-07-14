"""
:Authors: - Wilker Aziz
"""

from functools import wraps
import time
import sys


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