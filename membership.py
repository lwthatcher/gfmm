"""Fuzzy Membership Functions for FMMs"""
import numpy as np


class FuzzyMembershipFunction:
    """The standard Fuzzy Membership Function, introduced by Simpson"""

    def __init__(self, parent, gamma=1):
        """
        :param parent: the GFMM reference for accessing variables
        :param gamma: sensitivity parameter
        """
        self.parent = parent
        self.gamma = gamma

    def degree(self, a):
        """
        Returns the degree-of-membership for a given hyperbox
        :param a: The h'th input vector to consider
        :return: the degree of membership for all hyperboxes has for input Ah
        """
        dw = a.reshape(len(a), 1) - self.W
        dw[dw > 1] = 1  # min(1, a-w)
        dw *= self.gamma
        dw[dw < 0] = 0  # max(0, γ*min)
        dw = 1 - dw
        dw[dw < 0] = 0  # max(0, 1-max)

        dv = self.V - a.reshape(len(a), 1)
        dv[dv > 1] = 1  # min(1, v-a)
        dv *= self.gamma
        dv[dv < 0] = 0  # max(0, γ*min)
        dv = 1 - dv
        dv[dv < 0] = 0  # max(0, 1-max)

        wv = np.add(dw, dv)
        total = np.sum(wv, axis=0) * (1 / (2 * self.n))
        return total

    def __call__(self, *args, **kwargs):
        return self.degree(*args)

    @property
    def V(self):
        return self.parent.V

    @property
    def W(self):
        return self.parent.W

    @property
    def n(self):
        """ number of dimensions. """
        return self.parent.n or self.parent.V.shape[0]


class Clustering(FuzzyMembershipFunction):
    pass


class General(FuzzyMembershipFunction):
    pass
