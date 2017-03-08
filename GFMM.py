import numpy as np
import membership


class GFMM:

    def __init__(self, membership_func=None):
        if membership_func is None:
            membership_func = membership.FuzzyMembershipFunction
        self.X_l = np.zeros((0, 0))
        self.X_u = np.zeros((0, 0))
        self.n = 0
        self.num_hboxes = 0
        self.B_cls = []
        self.mfunc = membership_func(self)
        self.ϴ = 0.1
        self.φ = 0.9

    def fit(self, X, Y):
        """
        :param X: array-like, size=[n_samples, n_features]
            Training Data
        :param Y: array, dtype=float64, size=[n_samples]
            Target Values
            note that d=0 corresponds to an unlabeled item
        """
        input_length = X.shape[0]
        self._initialize(X)

        for h in range(input_length):
            xl = self.X_l[h, :]
            xu = self.X_u[h, :]
            d = Y[h]
            self._expansion(xl, xu, d)
            Δ, l = self._overlap_test()
            self._contraction(Δ, l)

    def predict(self, X):
        pass

    def _expansion(self, xl, xu, d):
        """
        Does the expansion step for the given input pattern.
        For consistency with notation, this is assumed to be the h'th input pattern.
        :param xl: array-like, size=[n_features]
            The min value for the h'th input pattern
        :param xu: array-like, size=[n_features]
            The max value for the h'th input pattern
        :param d: the h'th label
            d=0 means unlabeled
        """
        if self.num_hboxes == 0:
            self._add_hyperbox(xl, xu, d)
            return
        degree = self.mfunc(xl, xu)
        print(degree)

    def _overlap_test(self):
        """
        Checks if any hyperboxes are overlapping, and if so which case it is.
        If Δ = -1, then the contraction step can be skipped
        :return: tuple (Δ, l)
            Δ: the index of the overlapping dimension, returns -1 if no overlap
            l: the overlap case where l ϵ {1, 2, 3, 4}
        """
        return -1, None

    def _contraction(self, Δ, l):
        if Δ == -1:
            return
        pass

    def _initialize(self, X):
        """
        Initializes internal values and matrices from the input matrix
        :param X: the entire input data
        """
        # input matrices: Xl, Xu
        if len(X.shape) >= 3 and X.shape[2] >= 2:
            self.X_l = X[:, :, 0]
            self.X_u = X[:, :, 1]
        else:
            self.X_l = X
            self.X_u = np.copy(X)
        # set num dimensions
        self.n = X.shape[1]
        # initially no hyperboxes
        self.num_hboxes = 0
        # initialize hyperbox matrices
        self.V = np.zeros((self.n, 0))
        self.W = np.zeros((self.n, 0))

    def _add_hyperbox(self, xl, xu, cls):
        """
        Add a new hyperbox and set its initial min and max value.
        This corresponds to adding a new column in both V and W.
        :param xl: array-like, size = [n_dimensions]
            The lower bound of the input vector to set as the initial min values.
        :param xu: array-like, size = [n_dimensions]
            The upper bound of the input vector to set as the initial max values.
        :param cls: int, The class of the new hyperbox
        """
        # add column to V
        dV = np.zeros((self.n, self.num_hboxes+1))
        dV[:, :-1] = self.V
        if xl is not None:
            dV[:, -1] = xl
        self.V = dV
        # add column to W
        dW = np.zeros((self.n, self.num_hboxes+1))
        dW[:, :-1] = self.W
        if xu is not None:
            dW[:, -1] = xu
        self.W = dW
        # set class of new hyperbox
        self.B_cls.append(cls)
        # increment number-of-hyperboxes counter
        self.num_hboxes += 1
