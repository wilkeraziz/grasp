"""
:Authors: Wilker Aziz
"""

import unknownmodel
from fsa import make_linear_fsa
from symbol import Nonterminal, Terminal
from rule import CFGProduction
from fsa import WDFSA
import logging

PASSTHROUGH = 'passthrough'
STFDBASE = 'stfdbase'
STFD4 = 'stfd4'
STFD6 = 'stfd6'
DEFAULT_SYMBOL = 'X'

class Sentence(object):

    def __init__(self, words, signatures, fsa):
        self._words = tuple(words)
        self._signatures = tuple(signatures)
        self._fsa = fsa

    @property
    def words(self):
        return self._words

    @property
    def signatures(self):
        return self._signatures

    @property
    def fsa(self):
        return self._fsa

    def __str__(self):
        return ' '.join(self.words)


def make_sentence(input_str, semiring, lexicon, unkmodel=None, default_symbol=DEFAULT_SYMBOL):
    words = [Terminal(t) for t in input_str.split()]
    signatures = list(words)
    extra_rules = []
    str_lexicon = None
    for i, word in enumerate(words):
        if word not in lexicon and unkmodel is not None:
            # special treatment for unknown words
                if unkmodel == PASSTHROUGH:
                    extra_rules.append(CFGProduction(Nonterminal(default_symbol), [word], semiring.one))
                    logging.debug('Passthrough rule for %s: %s', word, extra_rules[-1])
                else:
                    if unkmodel == STFDBASE:
                        get_signature = unknownmodel.unknownwordbase
                    elif unkmodel == STFD4:
                        get_signature = unknownmodel.unknownword4
                    elif unkmodel == STFD6:
                        get_signature = unknownmodel.unknownword6
                    else:
                        raise NotImplementedError('I do not know this model: %s' % unkmodel)
                    if str_lexicon is None:
                        str_lexicon = set(t.surface for t in lexicon)
                    signatures[i] = Terminal(get_signature(word.surface, i, str_lexicon))

    fsa = WDFSA()
    for i, word in enumerate(signatures):
        fsa.add_arc(i, i + 1, word, semiring.one)
    fsa.make_initial(0)
    fsa.make_final(len(signatures))
    return Sentence(words, signatures, fsa), extra_rules