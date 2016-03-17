"""
:Authors: - Wilker Aziz
"""

from grasp.cfg.fsa import WDFSA
from grasp.cfg.symbol import Terminal, Nonterminal
from grasp.cfg.scfg import SCFG
from grasp.cfg.srule import SCFGProduction


def make_pass_grammar(seg, grammars, semiring, unk_lhs):
    """
    Make an input fsa for an input segment as well as its pass-through grammar.
    :param seg: a Segment object.
    :param grammars: a sequence of SCFGs.
    :param semiring: must provide `one`.
    :return: the input WDFSA, the pass-through grammar
    """
    fsa = WDFSA()
    pass_grammar = SCFG()
    unk = Nonterminal(unk_lhs)
    tokens = seg.src.split()
    for i, token in enumerate(tokens):
        word = Terminal(token)
        if any(g.in_ivocab(word) for g in grammars):
            pass_grammar.add(SCFGProduction.create(unk,
                                                   [word],
                                                   [word],
                                                   {'PassThrough': 1.0}))
        else:
            pass_grammar.add(SCFGProduction.create(unk,
                                                   [word],
                                                   [word],
                                                   {'PassThrough': 1.0,
                                                    'Unknown': 1.0}))
        fsa.add_arc(i, i + 1, word, semiring.one)
    fsa.make_initial(0)
    fsa.make_final(len(tokens))
    return fsa, pass_grammar