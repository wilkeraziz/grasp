from math import log
import ply.lex as lex
import ply.yacc as yacc
from ply_cfg import CFGLex

class SCFGYacc(object):

    def __init__(self, cfg_lexer=None):
        if cfg_lexer is None:
            cfg_lexer = CFGLex()
            cfg_lexer.build(debug=False, nowarn=True)
        SCFGYacc.tokens = cfg_lexer.tokens
        SCFGYacc.lexer = cfg_lexer.lexer

    def p_rule(self, p):
        'rule : NONTERMINAL BAR srhs BAR trhs'
        p[0] = (p[1], p[3], p[5], 0.0)

    def p_rule_and_weights(self, p):
        'rule : NONTERMINAL BAR srhs BAR trhs BAR prob'
        p[0] = (p[1], p[3], p[5], log(p[7]))

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

    def p_prob(self, p):
        'prob : FVALUE'
        p[0] = p[1]

    def p_default_prob(self, p):
        'prob : empty'
        p[0] = 1.0

    def p_empty(self, p):
        'empty :'
        pass

    def p_error(self, p):
        print("Syntax error at '%s'" % p)
    
    def build(self, **kwargs):
        self.parser = yacc.yacc(module=self, **kwargs)
        return self

    def parse(self, lines):
        for line in lines:
            production = self.parser.parse(line, lexer=self.lexer)
            yield production

    
if __name__ == '__main__':
    import sys
    parser = SCFGYacc()
    parser.build(debug=False, write_tables=False)
    for prod in parser.parse(sys.stdin.readlines()):
        print prod

