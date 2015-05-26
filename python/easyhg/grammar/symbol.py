"""
Contains class definitions for symbols (e.g. Terminal and Nonterminal) and other utilitary functions involving them.

:Authors: - Wilker Aziz
"""

from weakref import WeakValueDictionary

class Terminal(object):
    """
    Implements a terminal symbol. References to terminal symbols are managed by the Terminal class.
    We use WeakValueDictionary for builtin reference counting.

    >>> t1 = Terminal(1)
    >>> t2 = Terminal(1)
    >>> t1 == t2
    True
    >>> t1 is t2
    True
    >>> t3 = Terminal(2)
    >>> t1 != t3
    True
    >>> id(t1) == id(t2) != id(t3)
    True
    >>> hash(t1) == hash(t2) != hash(t3)
    True
    >>> Terminal(10)
    Terminal(10)
    >>> Terminal('x')
    Terminal('x')
    """

    _vocabulary = WeakValueDictionary()

    def __new__(cls, surface):
        """The surface has to be hashable."""

        obj = Terminal._vocabulary.get(surface, None)
        if not obj:
            obj = object.__new__(cls)
            Terminal._vocabulary[surface] = obj
            obj._surface = surface
        return obj

    @property
    def surface(self):
        """The underlying object that uniquely represent the symbol."""
        return self._surface

    def __repr__(self):
        return '%s(%s)' % (Terminal.__name__, repr(self._surface))

    def __str__(self):
        return "'{0}'".format(self._surface)
    
    def flatten(self):
        try:
            if isinstance(self._surface[0], Terminal):
                return self._surface[0]
        except:
            pass
        return self


class Nonterminal(object):
    """
    Implements a nonterminal symbol. References to nonterminal symbols are managed by the Nonterminal class.
    We use WeakValueDictionary for builtin reference counting.

    >>> n1 = Nonterminal('S')
    >>> n2 = Nonterminal('S')
    >>> n3 = Nonterminal('X')
    >>> n1 == n2 != n3
    True
    >>> n1 is n2 is not n3
    True
    >>> Nonterminal(('NP', 1, 2))  # a noun phrase spanning from 1 to 2
    Nonterminal(('NP', 1, 2))
    """

    _categories = WeakValueDictionary()

    def __new__(cls, label):
        """The label has to be hashable."""
        obj = Nonterminal._categories.get(label, None)
        if not obj:
            obj = object.__new__(cls)
            Nonterminal._categories[label] = obj
            obj._label = label
        return obj

    @property
    def label(self):
        """The underlying object that uniquely represents the symbol."""
        return self._label

    def __repr__(self):
        return '%s(%s)' % (Nonterminal.__name__, repr(self._label))

    def __str__(self):
        return '[{0}]'.format(self._label)

    def flatten(self):
        try:
            if isinstance(self._label[0], Nonterminal):
                return self._label[0]
        except:
            pass
        return self


def make_flat_symbol(base_symbol, sfrom, sto):
    """Return a symbol of same type (Terminal or Nonterminal) as `base_symbol`.
    If Nonterminal, the symbol is annotated with the span [sfrom, sto].

    >>> t = Terminal('a')
    >>> n = Nonterminal('X')
    >>> make_flat_symbol(t, 0, 1)
    Terminal('a')
    >>> make_flat_symbol(n, 1, 2)
    Nonterminal('X:1-2')
    """

    if sfrom is None and sto is None:
        return base_symbol
    return base_symbol if isinstance(base_symbol, Terminal) else Nonterminal('%s:%s-%s' % (base_symbol.label, sfrom, sto))

def make_recursive_symbol(base_symbol, sfrom, sto):
    """
    Return a symbol of same type (Terminal or Nonterminal) as `base_symbol`.
    If nonterminal, the label will be the tuple (base_symbol, sfrom, sto).

    >>> t = Terminal('a')
    >>> n = Nonterminal('X')
    >>> make_recursive_symbol(t, 0, 1)
    Terminal('a')
    >>> make_recursive_symbol(n, 1, 2)
    Nonterminal((Nonterminal('X'), 1, 2))
    """

    #if sfrom is None and sto is None:
    #    return base_symbol
    return base_symbol if isinstance(base_symbol, Terminal) else Nonterminal((base_symbol, sfrom, sto))

