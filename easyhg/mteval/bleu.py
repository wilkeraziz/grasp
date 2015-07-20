"""
:Authors: - Wilker Aziz
"""
from ._bleu import DecodingBLEU


class BLEU(object):

    def __init__(self):
        self._bleu_config = {'max_order': 4, 'smoothing': 'p1'}
        self._decoding_bleu_wrapper = None

    def prepare_decoding(self, support, distribution):
        """
        Compute sufficient statistics for BLEU in decoding mode
        """
        self._decoding_bleu_wrapper = DecodingBLEU(support, distribution, **self._bleu_config)

    def loss(self, c, r):
        return 1 - self._decoding_bleu_wrapper.bleu(c, r)

    def coloss(self, c):
        return 1 - self._decoding_bleu_wrapper.cobleu(c)

    def cleanup(self):
        self._decoding_bleu_wrapper = None