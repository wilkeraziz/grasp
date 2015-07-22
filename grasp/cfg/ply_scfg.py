"""
:Authors: - Wilker Aziz
"""

import ply.yacc as yacc
from .ply_cfg import CFGLex
from .rule import SCFGProduction
from .scfg import SCFG


_EXAMPLE_GRAMMAR_ = """
[S] ||| [X,1] ||| [1] ||| 1.0
[X] ||| [X,1] [X,2] ||| [1] [2] ||| 0.25
[X] ||| [X,1] [X,2] ||| [2] [1] ||| 0.25
[X] ||| '1' ||| '1' ||| 0.25
[X] ||| '2' ||| '2' ||| 0.25
"""


class SCFGYacc(object):

    def __init__(self, cfg_lexer=None, transform=None):
        if cfg_lexer is None:
            cfg_lexer = CFGLex()
            cfg_lexer.build(debug=False, nowarn=True)
        SCFGYacc.tokens = cfg_lexer.tokens
        SCFGYacc.lexer = cfg_lexer.lexer
        self.transform_ = transform

    def p_rule_and_weights(self, p):
        'rule : NONTERMINAL BAR srhs BAR trhs BAR weight'
        p[0] = SCFGProduction.create(p[1], p[3], p[5], p[7])

    def p_srhs(self, p):
        'srhs : rhs'
        p[0] = p[1]

    def p_trhs(self, p):
        'trhs : rhs'
        p[0] = p[1]

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

    def parse(self, istream):
        for line in istream:
            if line.startswith('#') or not line.strip():
                continue
            production = self.parser.parse(line, lexer=self.lexer)
            yield production

def cdec_adaptor(istream):
    for line in istream:
        if line.startswith('#') or not line.strip():
            continue
        fields = line.split('|||')
        lhs = fields[0]
        f_rhs = ' '.join(s if s.startswith('[') and s.endswith(']') else "'%s'" % s for s in fields[1].split())
        e_rhs = ' '.join(s if s.startswith('[') and s.endswith(']') else "'%s'" % s for s in fields[2].split())
        weight = '1.0'
        yield ' ||| '.join((lhs, f_rhs, e_rhs, weight))

def read_grammar(istream, transform=None, cdec_adapt=False):
    """Read a grammar parsed with CFGYacc from an input stream"""
    parser = SCFGYacc(transform=transform)
    parser.build(debug=False, write_tables=False)
    if cdec_adapt:
        return SCFG(parser.parse(cdec_adaptor(istream)))
    else:
        return SCFG(parser.parse(istream))

if __name__ == '__main__':
    import sys
    import logging
    FORMAT = '%(asctime)-15s %(message)s'
    G = read_grammar(sys.stdin)
    print('SCFG')
    print(G)
    print('F-CFG')
    print(G.f_projection())
    print('E-CFG')
    print(G.e_projection())

