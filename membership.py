"""Fuzzy Membership Functions for FMMs"""


class FuzzyMembershipFunction:

    def __init__(self, parent, gamma):
        """
        :param parent: the GFMM reference for accessing variables
        :param gamma: sensitivity parameter
        """
        self.parent = parent
        self.gamma = gamma

    def degree(self, Bj, Ah):
        """
        Returns the degree-of-membership for a given hyperbox
        :param Bj: The j'th hyperbox to consider
        :param Ah: The h'th input vector to consider
        :return: the degree of membership hyperbox Bj has for the input Ah
        """
        pass


class Classification(FuzzyMembershipFunction):
    pass


class Clustering(FuzzyMembershipFunction):
    pass


class General(FuzzyMembershipFunction):
    pass
