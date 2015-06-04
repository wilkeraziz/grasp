"""
Code copied/adapted from Andreas van Cranenburgh's discodop.
"""

import re

UNK = '_UNK'

# === functions for unknown word signatures ============

HASDIGIT = re.compile(r"\d", re.UNICODE)
HASNONDIGIT = re.compile(r"\D", re.UNICODE)
# NB: includes '-', hyphen, non-breaking hyphen
# does NOT include: figure-dash, em-dash, en-dash (these are punctuation,
# not word-combining) u2012-u2015; nb: these are hex values.
HASDASH = re.compile("[-\u2010\u2011]")
# FIXME: exclude accented characters for model 6?
HASLOWER = re.compile('[a-z\xe7\xe9\xe0\xec\xf9\xe2\xea\xee\xf4\xfb\xeb'
                      '\xef\xfc\xff\u0153\xe6]')
HASUPPER = re.compile('[A-Z\xc7\xc9\xc0\xcc\xd9\xc2\xca\xce\xd4\xdb\xcb'
                      '\xcf\xdc\u0178\u0152\xc6]')
HASLETTER = re.compile('[A-Za-z\xe7\xe9\xe0\xec\xf9\xe2\xea\xee\xf4\xfb'
                       '\xeb\xef\xfc\xff\u0153\xe6\xc7\xc9\xc0\xcc\xd9\xc2\xca\xce\xd4'
                       '\xdb\xcb\xcf\xdc\u0178\u0152\xc6]')
# Cf. http://en.wikipedia.org/wiki/French_alphabet
LOWER = ('abcdefghijklmnopqrstuvwxyz\xe7\xe9\xe0\xec\xf9\xe2\xea\xee\xf4\xfb'
         '\xeb\xef\xfc\xff\u0153\xe6')
UPPER = ('ABCDEFGHIJKLMNOPQRSTUVWXYZ\xc7\xc9\xc0\xcc\xd9\xc2\xca\xce\xd4\xdb'
         '\xcb\xcf\xdc\u0178\u0152\xc6')
LOWERUPPER = LOWER + UPPER


def unknownword6(word, loc, lexicon):
    """
    Model 6 of the Stanford parser (for WSJ treebank). 

    :param word: the actual word observed
    :param loc: the 0-based position in the sentence
    :param lexicon: the known terminals
    :return: signature
   
    >>> lexicon = set('This is a short sentence .'.split())
    >>> unknownword6('this', 0, lexicon)
    '_UNK-LC'
    >>> unknownword6('this', 1, lexicon)
    '_UNK-LC'
    >>> unknownword6('Sentence', 0, lexicon)
    '_UNK-INITC-KNOWNLC'
    >>> unknownword6('Sentence', 1, lexicon)
    '_UNK-CAP'
    >>> unknownword6('annoying', 1, lexicon)
    '_UNK-LC-ing'
    """
    wlen = len(word)
    numcaps = 0
    sig = UNK
    numcaps = len(HASUPPER.findall(word))
    lowered = word.lower()
    if numcaps > 1:
        sig += "-CAPS"
    elif numcaps > 0:
        if loc == 0:
            sig += "-INITC"
            if lowered in lexicon:
                sig += "-KNOWNLC"
        else:
            sig += "-CAP"
    elif HASLOWER.search(word):
        sig += "-LC"
    if HASDIGIT.search(word):
        sig += "-NUM"
    if HASDASH.search(word):
        sig += "-DASH"
    if lowered.endswith('s') and wlen >= 3:
        if lowered[-2] not in 'siu':
            sig += '-s'
    elif wlen >= 5 and not HASDASH.search(word) and not (HASDIGIT.search(word) and numcaps > 0):
        suffixes = ('ed', 'ing', 'ion', 'er', 'est', 'ly', 'ity', 'y', 'al')
        for a in suffixes:
            if lowered.endswith(a):
                sig += "-%s" % a
                break
    return sig


def unknownword4(word, loc, lexicon=set()):
    """
    Model 4 of the Stanford parser. Relatively language agnostic.

    :param word: the actual word observed
    :param loc: the 0-based position in the sentence
    :param lexicon: the known terminals (ignored in this version)
    :return: signature

    >>> unknownword4('this', 0)
    '_UNK-L-is'
    >>> unknownword4('this', 1)
    '_UNK-L-is'
    >>> unknownword4('Sentence', 0)
    '_UNK-SC-ce'
    >>> unknownword4('Sentence', 1)
    '_UNK-C-ce'
    >>> unknownword4('annoying', 1)
    '_UNK-L-ng'
    """
    sig = UNK

    # letters
    if word and word[0] in UPPER:
        if not HASLOWER.search(word):
            sig += "-AC"
        elif loc == 0:
            sig += "-SC"
        else:
            sig += "-C"
    elif HASLOWER.search(word):
        sig += "-L"
    elif HASLETTER.search(word):
        sig += "-U"
    else:
        sig += "-S"  # no letter

    # digits
    if HASDIGIT.search(word):
        if HASNONDIGIT.search(word):
            sig += "-n"
        else:
            sig += "-N"

    # punctuation
    if "-" in word:
        sig += "-H"
    if "." in word:
        sig += "-P"
    if "," in word:
        sig += "-C"
    if len(word) > 3:
        if word[-1] in LOWERUPPER:
            sig += "-%s" % word[-2:].lower()
    return sig


def unknownwordbase(word, loc=0, lexicon=set()):
    """
    BaseUnknownWordModel of the Stanford parser.
    Relatively language agnostic.

    :param word: the actual word observed
    :param loc: the 0-based position in the sentence (ignored in this version)
    :param lexicon: the known terminals (ignored in this version)
    :return: signature

    >>> unknownwordbase('this')
    '_UNK-c-is'
    >>> unknownwordbase('this')
    '_UNK-c-is'
    >>> unknownwordbase('Sentence')
    '_UNK-C-ce'
    >>> unknownwordbase('Sentence')
    '_UNK-C-ce'
    >>> unknownwordbase('annoying')
    '_UNK-c-ng'
    """
    sig = UNK

    # letters
    if word[0] in UPPER:
        sig += "-C"
    else:
        sig += "-c"

    # digits
    if HASDIGIT.search(word):
        if HASNONDIGIT.search(word):
            sig += "-n"
        else:
            sig += "-N"

    # punctuation
    if "-" in word:
        sig += "-H"
    if word == ".":
        sig += "-P"
    if word == ",":
        sig += "-C"
    if len(word) > 3:
        if word[-1] in LOWERUPPER:
            sig += "-%s" % word[-2:].lower()
    return sig
