"""
@author wilkeraziz
"""

from numpy import log
from itertools import ifilter
import ply.lex as lex
import ply.yacc as yacc

from symbol import Terminal, Nonterminal
from rule import CFGProduction
from cfg import CFG 


_EXAMPLE_GRAMMAR_ = """
[S] ||| '<s>' [X] '<s>' ||| 1.0
[X] ||| [X] [X] ||| 0.5
[X] ||| '1' ||| 0.25
[X] ||| '2' ||| 0.25
"""

class CFGLex(object):

    # Token definitions
    tokens = (
            'TERMINAL',
            'NONTERMINAL',
            #'FNAME', 
            'FVALUE',
            'BAR', 
            #'EQUALS'.
            )

    t_BAR    = r'[\|]{3}'
    #t_EQUALS = r'='
    #t_FNAME  = r"[^ '=\[\]\|]+"

    def t_TERMINAL(self, t):
        r"'[^ ]*'"
        t.value = Terminal(t.value[1:-1])
        return t

    def t_NONTERMINAL(self, t):
        r"\[([^ \[\]]+)\]"
        t.value = Nonterminal(t.value[1:-1])
        return t

    def t_FVALUE(self, t):
        r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'
        t.value = float(t.value)
        return t

    # Ignored characters
    t_ignore = " \t"

    def t_newline(self, t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")

    def t_error(self, t):
        print("Illegal character '%s'" % t.value[0])
        t.lexer.skip(1)

    def build(self, **kwargs):
        self.lexer = lex.lex(module=self, **kwargs)

    def tokenize(self, data):
        self.lexer.input(data)
        for tok in self.lexer:
            yield tok


class CFGYacc(object):
    """
    Parse a weighted CFG from a text file formatted as the _EXAMPLE_GRAMMAR_ above.

    >>> parser = CFGYacc()
    >>> _ = parser.build(debug=False, write_tables=False)
    >>> G = [r for r in parser.parse(_EXAMPLE_GRAMMAR_.splitlines())]
    >>> len(G)
    4
    >>> G[0]
    CFGProduction(Nonterminal('S'), (Terminal('<s>'), Nonterminal('X'), Terminal('<s>')), 1.0)
    >>> str(G[0])
    "[S] ||| '<s>' [X] '<s>' ||| 1.0"
    >>> G[3]
    CFGProduction(Nonterminal('X'), (Terminal('2'),), 0.25)
    >>> str(G[3])
    "[X] ||| '2' ||| 0.25"
    """

    def __init__(self, cfg_lexer=None, transform=None):
        if cfg_lexer is None:
            cfg_lexer = CFGLex()
            cfg_lexer.build(debug=False, nowarn=True)
        CFGYacc.tokens = cfg_lexer.tokens
        CFGYacc.lexer = cfg_lexer.lexer
        self.transform_ = transform

    def p_rule_and_weights(self, p):
        'rule : NONTERMINAL BAR rhs BAR weight'
        p[0] = CFGProduction(p[1], p[3], p[5])

    def p_rhs(self, p):
        '''rhs : TERMINAL
               | NONTERMINAL'''
        p[0] = [p[1]]

    def p_rhs_recursion(self, p):
        '''rhs : rhs TERMINAL
               | rhs NONTERMINAL'''
        p[0] = p[1] + [p[2]]

    def p_weight(self, p):
        'weight : FVALUE'
        if self.transform_:
            p[0] = self.transform_(p[1])
        else:
            p[0] = p[1]

    def p_error(self, p):
        print("Syntax error at '%s'" % p)
    
    def build(self, **kwargs):
        self.parser = yacc.yacc(module=self, **kwargs)
        return self

    def parse(self, lines):
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            production = self.parser.parse(line, lexer=self.lexer)
            yield production


def read_grammar(istream, transform=None):
    """Read a grammar parsed with CFGYacc from an input stream"""
    parser = CFGYacc(transform=transform)
    parser.build(debug=False, write_tables=False)
    return CFG(parser.parse(istream.readlines()))

if __name__ == '__main__':
    import sys
    print read_grammar(sys.stdin)

