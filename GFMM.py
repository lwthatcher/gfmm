import numpy as np
import membership


class GFMM:

    def __init__(self, membership_func=None):
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
        self.B_cls = np.array([])
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
        # TODO: initialize only once option?
        self._initialize(X)
        out = []

        for h in range(input_length):
            xl = self.X_l[h, :]
            xu = self.X_u[h, :]
            d = Y[h]
            j, ď, exp = self._expansion(xl, xu, d)
            out.append(ď)
            if exp:
                Δ, l = self._overlap_test(j, ď)
                self._contraction(Δ, l)
        return out

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
        :return: tuple (j, ď, exp)
            j: int
                The index of the expanded or containing hyperbox.
                If a new hyperbox was created, then this should be -1.
            ď: int
                The classification value assigned.
            exp: boolean
                True if expansion occurred, False otherwise.
        """
        if self.hboxes == 0:
            ď = self._add_hyperbox(xl, xu, d)
            return -1, ď, False
        degree = self.mfunc(xl, xu)
        # idx: ordered list of indices corresponding to candidate hyperboxes to expand
        idx = self.k_best(degree, self.Kn)
        if len(idx) > 0:
            idx = self._can_expand(idx, xl, xu)
            if len(idx) > 0:
                if d == 0:
                    j = idx[0]
                    exp = self._expand(j, xl, xu)
                    return j, self.B_cls[j], exp
                idx = self._valid_class(idx, d)
                if len(idx) > 0:
                    j = idx[0]
                    exp = self._expand(j, xl, xu)
                    if self.B_cls[j] == 0:
                        self.B_cls[j] = d
                    return j, d, exp
        ď = self._add_hyperbox(xl, xu, d)
        return -1, ď, False

    def _overlap_test(self, j, d):
        """
        Checks if any hyperboxes are overlapping, and if so which case it is.
        If Δ = -1, then the contraction step can be skipped
        :param j: int
            The index of the expanded hyperbox to check for overlap with.
        :param d: int
            The output classification of the expanded hyperbox.
            Note that we can ignore overlap for all other hyperboxes of the same class.
        :return: tuple (Δ, l)
            Δ: the index of the overlapping dimension, returns -1 if no overlap.
            l: the overlap case where l ϵ {1, 2, 3, 4}
        """
        Δ = -1
        l = None
        # get candidate boxes
        if self.B_cls[j] == 0:
            # if d == 0, check for overlap with all other hyperboxes
            idx = self.B_cls >= 0
        else:
            # otherwise don't check for overlap within the same class
            idx = self.B_cls != d
        # if no candidates for overlap, Δ = -1
        if len(idx) > 0:
            # alias for convenience
            V = self.V[idx]
            W = self.W[idx]
            # hyperbox Bj
            Vj = V[:,j].reshape(self.n, 1)
            Wj = W[:,j].reshape(self.n, 1)
            # store some other useful variables
            # TODO: extract below to separate function



        return Δ, l

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
        self.B_cls = np.array([])
        # initialize hyperbox matrices
        self.V = np.zeros((self.n, 0))
        self.W = np.zeros((self.n, 0))

    def _expand(self, j, xl, xu):
        """
        Expands the j'th hyperbox to fit the provided data
        :param j: int
            The index of the hyperbox to expand
        :param xl: array-like, size=[n_dimensions]
            The lower bound of the input vector to cover
        :param xu: array-like, size=[n_dimensions]
            The upper bound of the input vector to cover
        :return: boolean
            True if expansion occurred, False otherwise.
        """
        # check if completely contained within hyperbox j
        if np.all([self.V[:,j] < xl, self.W[:,j] > xu]):
            return False
        self.V[:,j] = np.minimum(self.V[:,j], xl)
        self.W[:,j] = np.maximum(self.W[:,j], xu)
        return True

    def _add_hyperbox(self, xl, xu, cls):
        """
        Add a new hyperbox and set its initial min and max value.
        This corresponds to adding a new column in both V and W.
        :param xl: array-like, size = [n_dimensions]
            The lower bound of the input vector to set as the initial min values.
        :param xu: array-like, size = [n_dimensions]
            The upper bound of the input vector to set as the initial max values.
        :param cls: int
            The classification of the new hyperbox
        :return: The assigned classification
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
        # TODO: add clustering support, where if d==0, B_cls[-1] = p+1
        self.B_cls = np.append(self.B_cls, cls)
        # increment number-of-hyperboxes counter
        self.hboxes += 1
        # return classification
        return cls

    def _valid_class(self, idx, d):
        """
        Reduces a list of candidate indices based on whether the corresponding
        hyperbox is of a valid class.
        i.e.:
        Hyperbox Bj has a valid class if class(Bj) == 0 or class(Bj) == d.
        :param idx: array-like, size<=[min(Kn, n_hyperboxes)]
            List of candidate indices already being considered.
        :param d: int
            The corresponding output class.
            d == 0 represents unlabeled data.
        :return: The filtered list of candidate indices.
        """
        # gets all hyperboxes that have class 0 or class d
        c0 = self.B_cls[idx] == 0
        cd = self.B_cls[idx] == d
        result = np.any([c0, cd], 0)
        return idx[result]

    def _can_expand(self, idx, xl, xu):
        """
        Checks whether the hyperbox can expand to include the specified point
        based on the inequality:

        ∀i[max(Wji, xu_i) - min(Vji, xl_i)] ≤ ϴ

         -where i is the input dimension
        :param idx: array-like, size<=[min(Kn, n_hyperboxes)]
            List of candidate indices already being considered.
        :param xl: array-like, size = [n_dimensions]
            The lower bound of the input vector
        :param xu: array-like, size = [n_dimensions]
            The upper bound of the input vector to cover
        :return: The filtered list of candidate indices.
        """
        W_max = np.maximum(self.W[:, idx], xu.reshape(len(xu), 1))
        V_min = np.minimum(self.V[:, idx], xl.reshape(len(xl), 1))
        dim_sizes = W_max - V_min
        result = np.all(dim_sizes <= self.ϴ, 0)
        return idx[result]

    @staticmethod
    def min_overlap_adjustment(V, W, Vj, Wj):
        """
        Finds the dimension, case, and hyperbox index of the minimum overlap adjustment.
        Here it compares the j'th hyperbox against all other candidate hyperboxes
        :param V: array-like, size=[n_dimension, n_candidates]
            The filtered min matrix
        :param W: array-like , size=[n_dimension, n_candidates]
            The filtered max matrix
        :param Vj: array-like, size=[n_dimensions]
            The min values for the j'th hyperbox.
        :param Wj: array-like, size=[n_dimensions]
            The max values for the j'th hyperbox.
        :return: tuple (Δ, l, k)
            Δ: the index of the least overlapping dimension, returns -1 if no overlap.
            l: the overlap case where l ϵ {1, 2, 3, 4}
            k: the index of the other hyperbox to adjust
        """
        # TODO: Is 1 the best choice?
        FILL = 1
        # only compute these values once
        vjv = Vj < V
        wjw = Wj < W
        vvj = V < Vj
        wwj = W < Wj
        # get indices where each case is true
        case_1 = np.all([vjv, V < Wj, wjw], 0)
        case_2 = np.all([vvj, Vj < W, wwj], 0)
        case_3 = np.all([vjv, V <= W, wwj], 0)
        case_4 = np.all([vvj, wjw], 0)
        # get the respective overlap matrices
        c1 = Wj-V
        c2 = W-Vj
        c3 = np.minimum(c2, c1)
        c4 = np.minimum(c1, c2)
        # mask non-overlapping values
        c1[case_1 != True] = FILL
        c2[case_2 != True] = FILL
        c3[case_3 != True] = FILL
        c4[case_4 != True] = FILL
        # pull together for convenience
        diff = np.array([c1, c2, c3, c4])
        l, Δ, k = np.unravel_index(diff.argmin(), diff.shape)
        l += 1  # convert from zero index so l ϵ {1, 2, 3, 4}
        # TODO: convert from filtered array indices, to actual indices?
        return Δ, l, k

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
            # needs to be returned as an array, to no break things elsewhere
            return np.array([np.argmax(d)])
        idx = np.argpartition(d, -k)[::-1]  # indices for first k which are top k, not necessarily in order
        s_idx = np.argsort(d[idx[:k]])[::-1]  # the k sorted indices for M, relative to idx
        return idx[s_idx]  # returns indices for top k, in sorted order
    # endregion

    if __name__ == "__main__":
        print("GFMM coming soon")
        pass
