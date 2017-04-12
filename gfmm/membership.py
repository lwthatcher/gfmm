"""Fuzzy Membership Functions for FMMs"""
import numpy as np


class FuzzyMembershipFunction:
    """The standard Fuzzy Membership Function, introduced by Simpson"""

    def __init__(self, parent, gamma=1):
        """
        :type gamma: int or array-like
        :param parent: the GFMM reference for accessing variables
        :param gamma: int or array-like, size=[n_dimensions]
            sensitivity parameter
        """
        self.parent = parent
        if not np.isscalar(gamma):
            if len(gamma.shape) == 1:
                gamma = gamma.reshape(len(gamma), 1)
        self.gamma = gamma

    def degree(self, al, au):
        """
        Returns the degree-of-membership for a given hyperbox
        :param al: The min value of the h'th input vector to consider
        :param au: The max value of the h'th input vector to consider
        :return: the degree of membership for all hyperboxes has for input Ah
        """
        # get A-W part
        dw = au.reshape(len(au), 1) - self.W
        dw = self._inner_maxmins(dw)
        # get V-A part
        dv = self.V - al.reshape(len(al), 1)
        dv = self._inner_maxmins(dv)
        # add these two parts together
        wv = np.add(dw, dv)
        # reduce to vector (1/2n * sum)
        total = np.sum(wv, axis=0) * (1 / (2 * self.n))
        return total
    
    def _inner_maxmins(self, q):
        q[q > 1] = 1  # min(1, q)
        q *= self.gamma
        q[q < 0] = 0  # max(0, γ*min(1,q))
        q = 1 - q
        q[q < 0] = 0  # max(0, 1-max(0, γ*min(1,q)))
        return q

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


def ramp(r, γ):
    rγ = r * γ
    rγ[rγ > 1] = 1
    rγ[rγ < 0] = 0
    return rγ


class Clustering(FuzzyMembershipFunction):

    # noinspection PyTypeChecker
    def degree(self, al, au):
        dw = ramp(au.reshape(len(au), 1) - self.W, self.gamma)
        dv = ramp(self.V - au.reshape(len(au), 1), self.gamma)
        inner_score = 1 - dw - dv
        total = np.sum(inner_score, axis=0) * (1 / self.n)
        return total


class General(FuzzyMembershipFunction):

    def degree(self, al, au):
        dw = 1 - ramp(au.reshape(len(au), 1) - self.W, self.gamma)
        dv = 1 - ramp(self.V - au.reshape(len(au), 1), self.gamma)
        return np.min(np.min([dw, dv], axis=0), axis=0)


def get_membership_function(name):
    if name == "standard":
        return FuzzyMembershipFunction
    elif name == "cluster":
        return Clustering
    elif name == "general":
        return General
