import numpy as np
import membership


class GFMM:

    def __init__(self, membership_func=membership.General):
        print('Hello GFMM')
        pass

    def fit(self, X, Y):
        """
        :param X: array-like, size=[n_samples, n_features]
            Training Data
        :param Y: array, dtype=float64, size=[n_samples]
            Target Values
            note that d=0 corresponds to an unlabeled item
        """
        pass

    def predict(self, X):
        pass

    def _expansion(self):
        pass

    def _overlap_test(self):
        pass

    def _contraction(self):
        pass

    def _initialize(self, n):
        """
        Initializes the V and W matrices
        :param n: the number of input dimensions
        """
        self.V = np.zeros((n, 0))
        self.W = np.zeros((n, 0))