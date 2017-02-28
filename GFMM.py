import numpy as np


class GFMM:

    def __init__(self, n):
        self.V = np.zeros((n, 0))
        self.W = np.zeros((n, 0))

    def fit(self, X):
        pass

    def predict(self):
        pass

    def _expansion(self):
        pass

    def _overlap_test(self):
        pass

    def _contraction(self):
        pass
