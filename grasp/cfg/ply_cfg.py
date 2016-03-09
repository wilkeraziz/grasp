"""
:Authors: - Wilker Aziz
"""

import ply.lex as lex
import ply.yacc as yacc

from .symbol import Terminal, Nonterminal
from .cfg import CFG
from grasp.recipes import smart_ropen
from grasp.cfg.rule import NewCFGProduction as CFGProduction


_EXAMPLE_GRAMMAR_ = """
[S] ||| '<s>' [X] '<s>' ||| 1.0
[X] ||| [X] [X] ||| 0.5
[X] ||| '1' ||| 0.25
[X] ||| '2' ||| 0.25
[NP-SBJ|<UCP,,>] ||| 'x' ||| 1.0
"""

class CFGLex(object):

    # Token definitions
    tokens = (
            'TERMINAL',
            'NONTERMINAL',
            'FNAME',
            'FVALUE',
            'BAR', 
            'EQUALS'
            )

    t_BAR    = r'[\|]{3}'
    t_EQUALS = r'='
    t_FNAME  = r"[^ '=\[\]\|]+"

    def t_TERMINAL(self, t):
        r"'[^ ]*'"
        t.value = Terminal(t.value[1:-1])
        return t

    def t_NONTERMINAL(self, t):
        r"\[([^ ]+)\]"
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
        print("Illegal character '%s'" % t.value[0], file=sys.stderr)
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
    """

    def __init__(self, cfg_lexer=None, transform=None, fprefix='UnnamedFeature'):
        if cfg_lexer is None:
            cfg_lexer = CFGLex()
            cfg_lexer.build(debug=False, nowarn=True, optimize=True, lextab='cfg_lextab')
        CFGYacc.tokens = cfg_lexer.tokens
        CFGYacc.lexer = cfg_lexer.lexer
        self.transform_ = transform
        self.fprefix_ = fprefix

    def p_rule_and_weights(self, p):
        'rule : NONTERMINAL BAR rhs BAR fpairs'

        # convert fpairs to fmap
        i = 0
        fmap = {}
        for k, v in p[5]:
            if k == '':
                fmap['{0}{1}'.format(self.fprefix_, i)] = v
                i += 1
            else:
                fmap[k] = v
        # construct a scfg production
        p[0] = CFGProduction(p[1], p[3], fmap)

    def p_rhs(self, p):
        '''rhs : TERMINAL
               | NONTERMINAL'''
        p[0] = [p[1]]

    def p_rhs_recursion(self, p):
        '''rhs : rhs TERMINAL
               | rhs NONTERMINAL'''
        p[0] = p[1] + [p[2]]

    def p_fpairs(self, p):
        """fpairs : fpair
                  | fvalue"""
        p[0] = [p[1]]

    def p_fpairs_recursion(self, p):
        """fpairs : fpairs fpair
                  | fpairs fvalue"""
        p[0] = p[1] + [p[2]]

    def p_fpair(self, p):
        """fpair : FNAME EQUALS FVALUE"""
        p[0] = (p[1], self.transform_(p[3]))

    def p_fvalue(self, p):
        """fvalue : FVALUE"""
        p[0] = ('', self.transform_(p[1]))

    def p_error(self, p):
        print("Syntax error at '%s'" % p, file=sys.stderr)
    
    def build(self, **kwargs):
        self.parser = yacc.yacc(module=self, **kwargs)
        return self

    def parse(self, lines):
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            production = self.parser.parse(line, lexer=self.lexer)
            yield production


def read_basic(istream, transform, fprefix='UnnamedFeature'):
    for line in istream:
        if line.startswith('#'):
            continue
        line = line.strip()
        if not line:
            continue
        fields = line.split(' ||| ')
        if len(fields) != 3:
            raise ValueError('I expected 3 fields, got %d: %s' % (len(fields), fields))
        if not (fields[0].startswith('[') and fields[0].endswith(']')):
            raise ValueError('Expected a nonterminal LHS, got something else: %s' % fields[0])
        lhs = Nonterminal(fields[0][1:-1])  # ignore brackets
        rhs = tuple(Nonterminal(x[1:-1]) if x.startswith('[') and x.endswith(']') else Terminal(x[1:-1]) for x in fields[1].split())
        fmap = {}
        unnamed = 0
        for pair in fields[2].split():
            parts = pair.split('=')
            if len(parts) == 2:
                try:
                    fmap[parts[0]] = float(parts[1])
                except ValueError:
                    raise ValueError('Feature values must be real numbers: %s' % parts[1])
            elif len(parts) == 1:
                try:
                    fmap['{0}{1}'.format(fprefix, unnamed)] = float(parts[0])
                except ValueError:
                    raise ValueError('Feature values must be real numbers: %s' % parts[0])
                unnamed += 1
            else:
                raise ValueError("Wrong feature-value format (perhaps you have '=' as part of the feature name?): %s" % pair)
        yield CFGProduction(lhs, rhs, fmap)


def cdec_adaptor(istream):
    for line in istream:
        if line.startswith('#') or not line.strip():
            continue
        fields = line.split('|||')
        fields[1] = ' '.join(s if s.startswith('[') and s.endswith(']') else "'%s'" % s for s in fields[1].split())
        yield ' ||| '.join(fields)


def read_grammar(istream, transform=float, cdec_adapt=False, fprefix='UnnamedFeature', ply_based=True):
    """
    Read a grammar from an input stream.
    :param istream: an input stream or a path to grammar file.
    :param transform: a transformation (e.g. log).
    :param cdec_adapt: wehter or not the input grammar is in cdec format
    :param fprefix: prefix used in naming unnamed features
    :param ply_based: whether or not to use a lex-yacc parser
    :return: a CFG
    """

    if type(istream) is str:
        istream = smart_ropen(istream)
    if cdec_adapt:
        istream = cdec_adaptor(istream)
    if ply_based:
        parser = CFGYacc(transform=transform, fprefix=fprefix)
        parser.build(debug=False, optimize=True, write_tables=True, tabmodule='cfg_yacctab')
        return CFG(parser.parse(istream))
    else:
        return CFG(read_basic(istream, transform))


if __name__ == '__main__':
    import sys
    cdec_adapt = False
    if len(sys.argv) == 2:
        if sys.argv[1] == 'cdec':
            cdec_adapt = True
    print(read_grammar(sys.stdin, cdec_adapt=cdec_adapt, ply_based=True))

