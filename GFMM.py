import numpy as np
import membership


class GFMM:

    def __init__(self, membership_func=None):
        """

        :param membership_func:
        """
        # membership function
        if membership_func is None:
            membership_func = membership.FuzzyMembershipFunction
        self.mfunc = membership_func(self)
        # initial input min/max arrays
        self.X_l = np.zeros((0, 0))
        self.X_u = np.zeros((0, 0))
        # number of dimensions
        self.n = 0
        # number of hyperboxes
        self.hboxes = 0
        # classes of hyperboxes
        self.B_cls = []
        # max size of hyperboxes
        self.ϴ = 0.1
        # speed of decrease of ϴ
        self.φ = 0.9
        # K-nearest neighbors to retrieve for expansion
        self.Kn = 10

    # region Public Methods
    def fit(self, X, Y):
        """
        :param X: array-like, size=[n_samples, n_features]
            Training Data
        :param Y: array-like, dtype=float64, size=[n_samples]
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
    # endregion

    # region Pipeline
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
        if self.hboxes == 0:
            self._add_hyperbox(xl, xu, d)
            return
        degree = self.mfunc(xl, xu)
        k_idx = self.k_best(degree, self.Kn)
        print(degree[k_idx])

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
    # endregion

    # region Helper Methods
    def _initialize(self, X):
        """
        Initializes internal values and matrices from the input matrix
        This is typically called from the .fit( ) method
        :param X: array-like, size=[n_samples, n_features]
            The training data
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
        self.hboxes = 0
        self.B_cls = []
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
        :param cls: int
            The class of the new hyperbox
        """
        # add column to V
        dV = np.zeros((self.n, self.hboxes + 1))
        dV[:, :-1] = self.V
        if xl is not None:
            dV[:, -1] = xl
        self.V = dV
        # add column to W
        dW = np.zeros((self.n, self.hboxes + 1))
        dW[:, :-1] = self.W
        if xu is not None:
            dW[:, -1] = xu
        self.W = dW
        # set class of new hyperbox
        self.B_cls.append(cls)
        # increment number-of-hyperboxes counter
        self.hboxes += 1

    def _valid_class(self, idx, d):
        pass

    def _can_expand(self, xl, xu, idx):
        W_max = np.maximum(self.W, xu.reshape(len(xu), 1))
        V_min = np.minimum(self.V, xl.reshape(len(xl), 1))
        dim_sizes = W_max - V_min
        print("diff", dim_sizes)


    @staticmethod
    def k_best(d, k):
        """
        Gets the indices for the k highest values from the specified array, in sorted order
        :param d: array-like, size=[n_hyperboxes]
            The degree array
        :param k: int
            The number of values to look for.
        :return: array-like, size=[min(k, n_hyperboxes)]
            The k indices for the k highest values in d
        """
        k = min(k, len(d))
        if k == 1:
            return np.argmax(d)
        idx = np.argpartition(d, -k)[::-1]  # indices for first k which are top k, not necessarily in order
        s_idx = np.argsort(d[idx[:k]])[::-1]  # the k sorted indices for M, relative to idx
        return idx[s_idx]  # returns indices for top k, in sorted order
    # endregion

    if __name__ == "__main__":
        print("GFMM coming soon")
        pass
