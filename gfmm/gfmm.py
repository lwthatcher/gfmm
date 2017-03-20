import numpy as np

from gfmm.membership import FuzzyMembershipFunction


class GFMM:

    def __init__(self, m_func=None, gamma=1, n=None, p=None, Kn=10, theta=0.3, theta_min=0.03, phi=0.9):
        # TODO: add argument parsing
        # membership function
        if m_func is None:
            m_func = FuzzyMembershipFunction
        self.m_func = m_func(self, gamma)
        # number of dimensions
        self.n = n
        # number of hyperboxes
        self.m = 0
        # classes of hyperboxes
        self.B_cls = np.array([])
        self.V = None
        self.W = None
        # max size of hyperboxes
        self.ϴ = theta
        self.ϴ_min = theta_min
        # speed of decrease of ϴ (should be between 0 and 1)
        self.φ = phi
        # K-nearest neighbors to retrieve for expansion
        self.Kn = Kn
        # number of output classifications
        self.p = p

    # region Public Methods
    def fit(self, X, Y=None, wipe=False):
        """
        :param X: array-like, size=[n_samples, n_features]
            Training Data
        :param Y: array-like, dtype=float64, size=[n_samples]
            Target Values
            note that d=0 corresponds to an unlabeled item
        :param wipe: boolean
            If true, erases previous learned data. Default = False.
        :return: array, size=[n_samples]
            Returns the predicted output for each item in the input data.
        """
        input_length = X.shape[0]
        # TODO: if Y is not set, default to clustering
        X_l, X_u = self._initialize(X, Y, wipe)
        out = []
        # TODO: add multi-epoch support
        for h in range(input_length):
            xl = X_l[h, :]
            xu = X_u[h, :]
            d = Y[h]
            j, ď, exp = self._expansion(xl, xu, d)
            out.append(ď)
            if exp:
                Δ, l, k = self._overlap_test(j, ď)
                self._contraction(Δ, l, j, k)
        return self
        # TODO: add stopping criteria
        # TODO: add φ*ϴ update

    def predict(self, X):
        """
        Predicts the classification from the supplied samples.
        :param X: array-like, size=[n_samples, n_features]
            The feature vectors for the samples to predict.
        :return: array-like, size=[n_samples]
            The predicted classification results of the given array.
        """
        X_l, X_u = self.splice_matrix(X)
        input_length = X.shape[0]
        U = self.U
        out = []
        for h in range(input_length):
            xl = X_l[h, :]
            xu = X_u[h, :]
            Bh = self.m_func(xl, xu)
            totals = Bh.dot(U)
            cls = np.argmax(totals)
            out.append(cls)
        return np.array(out)
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
        if self.m == 0:
            ď = self._add_hyperbox(xl, xu, d)
            return -1, ď, False
        degree = self.m_func(xl, xu)
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
        :return: tuple (Δ, l, k)
            Δ: the index of the overlapping dimension, returns -1 if no overlap.
            l: the overlap case where l ϵ {1, 2, 3, 4}
            k: the index of the other hyperbox to adjust.
        """
        Δ = -1
        l = k = None
        # get candidate boxes
        if self.B_cls[j] == 0:
            # if d == 0, check for overlap with all other hyperboxes
            idx = self.B_cls >= 0   # all True
            idx[j] = False      # exclude j'th hyperbox
        else:
            # otherwise don't check for overlap within the same class
            idx = self.B_cls != d
        # if no candidates for overlap, Δ = -1
        if len(idx) > 0:
            # alias for convenience
            V = self.V[:, idx]
            W = self.W[:, idx]
            if V.shape == (1, 2):   # make sure V and W are column vectors
                V = V.reshape(self.n, 1)
                W = W.reshape(self.n, 1)
            # save inverse mask
            i_mask = np.where(idx)[0]
            # store hyperbox Bj
            Vj = self.V[:,j].reshape(self.n, 1)
            Wj = self.W[:,j].reshape(self.n, 1)
            # store some other useful variables
            Δ, l, k = self.min_overlap_adjustment(V, W, Vj, Wj)
            # convert k back from relative index
            k = i_mask[k]
        return Δ, l, k

    def _contraction(self, Δ, l, j, k):
        """
        If overlap occurred, contracts the specified hyperboxes to eliminate overlap.
        If no overlap has occurred (Δ == -1), then no contraction takes place.
        :param Δ: the index of the overlapping dimension, -1 if no overlap
        :param l: the overlap case where l ϵ {1, 2, 3, 4}
        :param j: index for recently expanded hyperbox Bj
        :param k: index for the hyperbox Bk that overlaps with Bj
        """
        # if no overlap, no expansion
        if Δ == -1:
            return
        # case 1
        elif l == 1:
            di = (self.V[Δ, k] + self.W[Δ, j]) / 2
            self.V[Δ, k] = di
            self.W[Δ, j] = di
        # case 2
        elif l == 2:
            di = (self.V[Δ, j] + self.W[Δ, k]) / 2
            self.V[Δ, j] = self.W[Δ, k] = di
        # case 3
        elif l == 3:
            if self.W[Δ,k]-self.V[Δ,j] < self.W[Δ,j]-self.V[Δ,k]:
                self.V[Δ, j] = self.W[Δ, k]
            else:
                self.W[Δ, j] = self.V[Δ, k]
        # case 4
        elif l == 4:
            if self.W[Δ,k]-self.V[Δ,j] < self.W[Δ,j]-self.V[Δ,k]:
                self.W[Δ, k] = self.V[Δ, j]
            else:
                self.V[Δ, k] = self.W[Δ, j]
    # endregion

    # region Initialization Methods
    def _initialize(self, X, Y=None, wipe=False):
        """
        Initializes internal values and matrices from the input matrix
        This is typically called from the .fit( ) method
        :param X: array-like, size=[n_samples, n_features]
            The training data
        :param Y: array-like or None, size=[n_samples]
            The classification labels corresponding to the input.
            Note that y[i] == 0 corresponds to no label.
            If y is None, it is initialized to all zeros.
        :return: tuple (X_l, X_u)
            X_l: The min value for each training instance.
            X_u: The max value for each training instance.
        """
        # input matrices: Xl, Xu
        X_l, X_u = self.splice_matrix(X)
        # if no output classes, initialize to all zeros
        if Y is None:
            n_samples = X.shape[0]
            Y = np.zeros(n_samples)
        # make sure p is set
        if wipe or self.p is None:
            print("inferring p: number of output classes")
            self.p = Y.max()
        # set num dimensions
        if wipe or self.n is None:
            self.n = X.shape[1]
        # if wipe is set, set # of hyperboxes to zero
        if wipe:
            self.m = 0
            self.B_cls = np.array([])
        # initialize or reset hyperbox matrices
        if wipe or self.V is None:
            self.V = np.zeros((self.n, 0))
            self.W = np.zeros((self.n, 0))
        return X_l, X_u
    # endregion

    # region Helper Methods
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
        dV = np.zeros((self.n, self.m + 1))
        dV[:, :-1] = self.V
        if xl is not None:
            dV[:, -1] = xl
        self.V = dV
        # add column to W
        dW = np.zeros((self.n, self.m + 1))
        dW[:, :-1] = self.W
        if xu is not None:
            dW[:, -1] = xu
        self.W = dW
        # set class of new hyperbox
        # TODO: add clustering support, where if d==0, B_cls[-1] = p+1
        self.B_cls = np.append(self.B_cls, cls)
        # increment number-of-hyperboxes counter
        self.m += 1
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
    # endregion

    # region Static Methods
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
            k: the index of the other hyperbox to adjust, relative to V and W.
        """
        FILL = np.nan
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
        # if no overlap, Δ = -1
        if np.all(np.isnan(diff)):
            return -1, None, None
        l, Δ, k = np.unravel_index(np.nanargmin(diff), diff.shape)
        l += 1  # convert from zero index so l ϵ {1, 2, 3, 4}
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

    @staticmethod
    def splice_matrix(X, depth=2):
        """
        Splits the matrix X into X_l and X_u,
        where X_l is the lower bound and X_u is the upper bound.
        If the input matrix is of shape=[n_samples, n_features, depth]
        then it is assumed that X[:,:,0] = X_l, and X[:,:,1] = X_u.
        :param X: array-like, size=[n_samples, n_features(, depth)]
        :param depth: int
            Default: 2 (not actually expected to be used yet...)
        :return: X_l, X_u
        """
        if len(X.shape) >= 3 and X.shape[2] >= depth:
            X_l = X[:, :, 0]
            X_u = X[:, :, 1]
        else:
            X_l = X
            X_u = np.copy(X)
        return X_l, X_u
    # endregion

    # region Properties
    @property
    def U(self):
        u = np.zeros((self.m, self.p + 1))   # m*p boolean matrix
        Bi = np.where(self.B_cls)[0]    # get the indices for B_cls
        u[Bi, self.B_cls.astype(int)] = 1   # if Bj is a hyperbox for class Ci, then Uij = 1
        return u
    # endregion

if __name__ == "__main__":
    print("GFMM coming soon")
    pass
