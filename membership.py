"""Fuzzy Membership Functions for FMMs"""
import numpy as np


class FuzzyMembershipFunction:

    def __init__(self, parent, gamma=1):
        """
        :param parent: the GFMM reference for accessing variables
        :param gamma: sensitivity parameter
        """
        self.parent = parent
        self.gamma = gamma

    def degree(self, x):
        """
        Returns the degree-of-membership for a given hyperbox
        :param x: The h'th input vector to consider
        :return: the degree of membership hyperbox Bj has for the input Ah
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.degree(*args)

    @property
    def V(self):
        return self.parent.V


class Classification(FuzzyMembershipFunction):

    def degree(self, a):
        print('V', self.parent.V, self.V, a)


class Clustering(FuzzyMembershipFunction):
    pass


class General(FuzzyMembershipFunction):
    pass
