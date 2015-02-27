"""

A basic symbol is actualy an integer
"""

from collections import deque, defaultdict
from array import array


class Terminal:

    def __init__(self, surface):
        """
        >>> Terminal('the')
        Terminal('the')
        >>> print(Terminal('the'))
        the
        """
        self.surface_ = surface
    
    def __repr__(self):
        return 'Terminal(%r)' % self.surface_

    def __str__(self):
        return str(self.surface_)

    def __eq__(self, other):
        return self.surface_ == other.surface_

    def __hash__(self):
        return hash(self.surface_)

    def is_terminal(self):
        return true
        

class Nonterminal:

    def __init__(self, category):
        """
        >>> Nonterminal('X')
        Nonterminal('X')
        >>> print(Nonterminal('X'))
        [X]
        """
        self.category_ = category
    
    def __repr__(self):
        return 'Nonterminal(%r)' % self.category_

    def __str__(self):
        return '[%s]' % self.category_

    def __eq__(self, other):
        return self.category_ == other.category_

    def __hash__(self):
        return hash(self.category_)

    def is_terminal(self):
        return false
    

class RHS(tuple):

    def __new__(cls, *symbols):  
        """
        >>> RHS(Terminal('the'))
        (Terminal('the'),)
        >>> RHS(Nonterminal('X'))
        (Nonterminal('X'),)
        >>> RHS(Terminal('the'), Nonterminal('X'), Terminal('dog'))
        (Terminal('the'), Nonterminal('X'), Terminal('dog'))
        >>> RHS([Terminal('the'), Nonterminal('X'), Terminal('dog')])
        (Terminal('the'), Nonterminal('X'), Terminal('dog'))
        >>> RHS(x for x in [Terminal('the'), Nonterminal('X'), Terminal('dog')] if type(x) is Terminal)
        (Terminal('the'), Terminal('dog'))
        >>> RHS(x for x in [Terminal('the'), Nonterminal('X'), Terminal('dog')] if type(x) is Nonterminal)
        (Nonterminal('X'),)
        """
        if len(symbols) == 0:
            raise ValueError('RHS cannot be empty! However, you can check about epsilon strings ;)')
        if len(symbols) > 1:
            return super(RHS, cls).__new__(cls, tuple(symbols))
        elif isinstance(symbols[0], Terminal) or isinstance(symbols[0], Nonterminal):
            return super(RHS, cls).__new__(cls, (symbols[0],))
        else:
            return super(RHS, cls).__new__(cls, symbols[0])

    def __str__(self):
        return ' '.join(str(s) for s in self)

class Production(tuple):

    def __new__(cls, lhs, rhs):
        """
        >>> the, black, dog = Terminal('the'), Terminal('black'), Terminal('dog')
        >>> X, S = Nonterminal('X'), Nonterminal('S')
        >>> Production(X, RHS(the, black, dog))
        (Nonterminal('X'), (Terminal('the'), Terminal('black'), Terminal('dog')))
        >>> Production(S, RHS(X))
        (Nonterminal('S'), (Nonterminal('X'),))
        >>> Production(S, RHS(S, X))
        (Nonterminal('S'), (Nonterminal('S'), Nonterminal('X')))
        """
        return super(Production, cls).__new__(cls, (lhs, rhs))

    def lhs(self):
        return self[0]

    def rhs(self):
        return self[1]

    def __str__(self):
        return '{0} ||| {1}'.format(self[0], self[1])

class CFG:
    """
    >>> the, black, dog = Terminal('the'), Terminal('black'), Terminal('dog')
    >>> X, S = Nonterminal('X'), Nonterminal('S')
    >>> r1 = Production(X, RHS(the, black, dog))
    >>> r2 = Production(S, RHS([X]))
    >>> r3 = Production(S, RHS([S, X]))
    >>> cfg = CFG()
    >>> cfg.append(r1)
    >>> cfg.append(r2)
    >>> cfg.append(r3)
    >>> print(cfg)
    [X] ||| the black dog
    [S] ||| [X]
    [S] ||| [S] [X]

    """

    def __init__(self):
        self.rules_ = []
        self.by_lhs_ = defaultdict(list)

    def append(self, rule):
        i = len(self.rules_)
        self.rules_.append(rule)
        self.by_lhs_[rule.lhs].append(i)

    def __str__(self):
        return '\n'.join(str(r) for r in self.rules_)



