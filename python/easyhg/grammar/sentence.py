"""
:Authors: - Wilker Aziz
"""

from . import unknownmodel
from .symbol import Nonterminal, Terminal
from .rule import CFGProduction
from .fsa import WDFSA
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
    """

    :param input_str:
    :param semiring:
    :param lexicon:
    :param unkmodel:
    :param default_symbol:
    :return: a Sentence object

    >>> from .semiring import Prob
    >>> from .symbol import Terminal
    >>> input_str = 'The dog barked'
    >>> lexicon = {Terminal(x) for x in 'the dog barked'.split()}
    >>> snt, extra = make_sentence(input_str, Prob, lexicon, unkmodel=STFDBASE)
    >>> snt.words
    ('The', 'dog', 'barked')
    >>> snt.signatures
    ('_UNK-C', 'dog', 'barked')
    >>> snt, extra = make_sentence(input_str, Prob, lexicon, unkmodel=STFD4)
    >>> snt.signatures
    ('_UNK-SC', 'dog', 'barked')
    >>> snt, extra = make_sentence(input_str, Prob, lexicon, unkmodel=STFD6)
    >>> snt.signatures
    ('_UNK-INITC-KNOWNLC', 'dog', 'barked')
    >>> snt, extra = make_sentence(input_str, Prob, lexicon, unkmodel=PASSTHROUGH)
    >>> snt.signatures
    ('The', 'dog', 'barked')
    >>> extra[0]
    CFGProduction(Nonterminal('X'), (Terminal('The'),), 1.0)
    """
    words = input_str.split()
    signatures = list(words)
    extra_rules = []
    str_lexicon = [t.surface for t in lexicon]
    for i, word in enumerate(words):
        if word not in str_lexicon and unkmodel is not None:
            # special treatment for unknown words
                if unkmodel == PASSTHROUGH:
                    extra_rules.append(CFGProduction(Nonterminal(default_symbol), [Terminal(word)], semiring.one))
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
                    signatures[i] = get_signature(word, i, str_lexicon)
                    logging.debug('Unknown word model (%s): i=%d word=%s signature=%s', unkmodel, i, word, signatures[i])

    fsa = WDFSA()
    for i, word in enumerate(signatures):
        fsa.add_arc(i, i + 1, Terminal(word), semiring.one)
    fsa.make_initial(0)
    fsa.make_final(len(signatures))
    return Sentence(words, signatures, fsa), extra_rules