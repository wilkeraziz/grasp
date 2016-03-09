"""
:Authors: - Wilker Aziz
"""

import ply.yacc as yacc
from .ply_cfg import CFGLex
from .srule import SCFGProduction
from .scfg import SCFG


_EXAMPLE_GRAMMAR_ = """
[S] ||| [X,1] ||| [1] ||| 1.0
[X] ||| [X,1] [X,2] ||| [1] [2] ||| 0.25
[X] ||| [X,1] [X,2] ||| [2] [1] ||| 0.25
[X] ||| '1' ||| '1' ||| 0.25
[X] ||| '2' ||| '2' ||| 0.25
"""


class SCFGYacc(object):

    def __init__(self, cfg_lexer=None, transform=None, fprefix='UnnamedFeature'):
        if cfg_lexer is None:
            cfg_lexer = CFGLex()
            cfg_lexer.build(debug=False, nowarn=True, optimize=True, lextab='scfg_lextab')
        SCFGYacc.tokens = cfg_lexer.tokens
        SCFGYacc.lexer = cfg_lexer.lexer
        self.transform_ = transform
        self.fprefix_ = fprefix

    def p_rule_and_weights(self, p):
        'rule : NONTERMINAL BAR srhs BAR trhs BAR fpairs'

        # convert fpairs to fmap
        i = 0
        fmap = {}
        for k, v in p[7]:
            if k == '':
                fmap['{0}{1}'.format(self.fprefix_, i)] = v
                i += 1
            else:
                fmap[k] = v
        # construct a scfg production
        p[0] = SCFGProduction.create(p[1], p[3], p[5], fmap)

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
        fields[1] = ' '.join(s if s.startswith('[') and s.endswith(']') else "'%s'" % s for s in fields[1].split())
        fields[2] = ' '.join(s if s.startswith('[') and s.endswith(']') else "'%s'" % s for s in fields[2].split())
        yield ' ||| '.join(fields)


def read_grammar(istream, transform=float, cdec_adapt=False, fprefix='UnnamedFeature'):
    """
    Read a grammar parsed with CFGYacc from an input stream

    :param istream: an input stream or a path to grammar file.
    :param transform: a transformation (e.g. log).
    :param cdec_adapt: if True the grammar is seen as in cdec-format
    :param fprefix: prefix used in naming unnamed features
    :return: an SCFG
    """
    parser = SCFGYacc(transform=transform, fprefix=fprefix)
    parser.build(debug=False, optimize=True, write_tables=True, tabmodule='scfg_yacctab')
    if cdec_adapt:
        return SCFG(parser.parse(cdec_adaptor(istream)))
    else:
        return SCFG(parser.parse(istream))

if __name__ == '__main__':
    import sys
    cdec_format = False
    if len(sys.argv) == 2:
        if sys.argv[1] == 'cdec':
            cdec_format = True
    G = read_grammar(sys.stdin, cdec_adapt=cdec_format)
    print('SCFG')
    print(G)
    #print('F-CFG')
    #print(G.f_projection())
    #print('E-CFG')
    #print(G.e_projection())

