"""
Contains class definitions for symbols (e.g. Terminal and Nonterminal) and other utilitary functions involving them.

:Authors: - Wilker Aziz
"""

from weakref import WeakValueDictionary
from collections import defaultdict


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
    #_vocabulary = defaultdict(None)

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

    @property
    def underlying(self):
        return self._surface

    def __repr__(self):
        return '%s(%s)' % (Terminal.__name__, repr(self._surface))

    def __str__(self):
        """Return the string associated with the underlying object wrapped with single quotes."""
        return "'{0}'".format(self._surface)

    def underlying_str(self):
        """Return the string associated with the underlying object."""
        return str(self._surface)
    
    def flatten(self):
        """Return itself"""
        return self


class _Nonterminal(object):
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
    #_categories = defaultdict(None)

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

    @property
    def underlying(self):
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


class Nonterminal(object):
    """

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

    @property
    def underlying(self):
        """Return the underlying object."""
        return self._label

    def __repr__(self):
        return '%s(%s)' % (Nonterminal.__name__, repr(self._label))

    def __str__(self):
        """Return the string associated with the underlying object wrapped with squared brackets."""
        return '[{0}]'.format(self._str())

    def underlying_str(self):
        """Return the string associated with the underlying object."""
        return str(self._label)

    def flatten(self):
        """Return itself."""
        return self


class Span(Nonterminal):
    """
    A Span is a Nonterminal rewriting from `start` to `end`.
    """

    def __new__(cls, nonterminal, start, end):
        assert(isinstance(nonterminal, Nonterminal)), 'In a span, the base symbol must be a Nonterminal.'
        return super(Span, cls).__new__(cls, (nonterminal, start, end))

    def __str__(self):
        """Underlying string representation wrapped with squared brackets."""
        return '[{0}]'.format(self.underlying_str())

    def __repr__(self):
        return '%s(%s)' % (Span.__name__, repr(self._label))

    def underlying_str(self):
        """Construct a string representing the span (except for the squared brackets)."""
        return '{0}:{1}-{2}'.format(self.underlying[0].underlying_str(),
                                    self.underlying[1] if self.underlying[1] is not None else '',
                                    self.underlying[2] if self.underlying[2] is not None else '')

    @property
    def base(self):
        return self._label[0]

    @property
    def start(self):
        return self._label[1]

    @property
    def end(self):
        return self._label[2]

    def flatten(self):
        """Return a basic nonterminal using the underlying string representation of the span(without squared brackets)."""
        return Nonterminal(self.underlying_str())


def _make_flat_symbol(base_symbol, sfrom, sto):
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

def _make_recursive_symbol(base_symbol, sfrom, sto):
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

def make_span(symbol, sfrom, sto):
    return symbol if isinstance(symbol, Terminal) else Span(symbol, sfrom, sto)

def flatten_symbol(symbol):
    if isinstance(symbol, Terminal):
        return symbol
    elif isinstance(symbol.label, tuple):
        if len(symbol.label) == 3:
            if symbol.label[1] is not None and symbol.label[2] is not None:
                return Nonterminal('%s:%s-%s' % (symbol.label[0].label, symbol.label[1], symbol.label[2]))
        return Nonterminal(symbol.label[0].label)
    else:
        return symbol.label