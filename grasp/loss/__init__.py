"""
In this package we provide loss functions that can be used in risk minimisation.

:Authors: - Wilker Aziz
"""


class Loss(object):

    def prepare_decoding(self, support, distribution):
        """
        Compute sufficient statistics based on an empirical distribution.

        :param support: a tuple of valid solutions
        :param distribution: a probability distribution over the support
        """
        pass

    def loss(self, c, r):
        """
        Compute the loss.

        :param c: 0-based index of the candidate solution
        :param r: 0-based index of the reference
        :return: loss
        """
        pass

    def coloss(self, c):
        """
        Compute the consensus-loss, or loss with respected to expected features.

        :param c: 0-based index of the candidate solution
        :return: coloss
        """
        pass

    def cleanup(self):
        """
        Cleanup sufficient statistics.
        """
        pass