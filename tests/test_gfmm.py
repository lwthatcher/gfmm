from unittest import TestCase

import numpy as np

from gfmm import GFMM


class TestGFMM(TestCase):
    # Test input matrices
    X1 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
    X2 = np.array([[.1, .1],
                   [.7, .7],
                   [.5, .5],
                   [.4, .3]])
    X3 = np.array([[[.1, .15], [.1, .15]],
                   [[.7, .75], [.7, .75]],
                   [[.5, .55], [.5, .55]],
                   [[.4, .45], [.3, .35]]])
    d2 = np.array([1, 2, 1, 2])

    # region Examples
    @property
    def CASE_STUDY_I(self):
        """
        The example illustrated in Fig. 5,6 from the KnFMM paper
        """
        class _CSI:
            def __init__(self):
                # initial state
                self.Va = np.array([[.1, .55, .2],
                                    [.2, .2, .6]])
                self.Wa = np.array([[.4, .75, .25],
                                    [.5, .4, .7]])
                # Fig 6.b
                self.Vb = np.array([[.1, .55, .2],
                                    [.2, .2, .53]])
                self.Wb = np.array([[.4, .75, .3],
                                    [.5, .4, .7]])
                # Fig 5.b
                self.Vb_k1 = np.array([[.1, .55, .2, .3],
                                       [.2, .2, .6, .53]])
                self.Wb_k1 = np.array([[.4, .75, .25, .3],
                                       [.5, .4, .7, .53]])
                self.gfmm = GFMM()
                self.gfmm.n = 2
                self.gfmm.hboxes = 3
                self.gfmm.p = 2
                self.gfmm.V = np.copy(self.Va)
                self.gfmm.W = np.copy(self.Wa)
                self.gfmm.B_cls = np.array([1, 2, 1])
                self.gfmm.ϴ = .301
                self.a1 = np.array([.3, .53])
                self.a2 = np.array([.38, .55])
                self.a3 = np.array([.1, .59])
        return _CSI()

    @property
    def EX_1(self):
        """
        The example illustrated in Fig 4 from the KnFMM paper
        """
        class _EX1:
            def __init__(self):
                # hyperbox classifications
                self.B_cls = np.array([1, 2])  # default
                # some default gfmm values
                self.gfmm = GFMM()
                self.gfmm.n = 2
                self.gfmm.hboxes = 2
                self.gfmm.p = 2
                self.gfmm.ϴ = .4
                self.gfmm.B_cls = self.B_cls
                # Fig 4.a
                self.Va = np.array([[.1], [.1]])
                self.Wa = np.array([[.1], [.1]])
                # Fig 4.b
                self.Vb = np.array([[.1, .7], [.1, .7]])
                self.Wb = np.array([[.1, .7], [.1, .7]])
                # Fig 4.c
                self.Vc = np.array([[.1, .7], [.1, .7]])
                self.Wc = np.array([[.5, .7], [.5, .7]])
                # Fig 4.d
                self.Vd = np.array([[.1, .4], [.1, .3]])
                self.Wd = np.array([[.5, .7], [.5, .7]])
                # Fig 4.e
                self.Ve = np.array([[.1, .45], [.1, .3]])
                self.We = np.array([[.45, .7], [.5, .7]])
                # input values
                self.a1 = np.array([.1, .1])
                self.a2 = np.array([.7, .7])
                self.a3 = np.array([.5, .5])
                self.a4 = np.array([.4, .3])
                self.X = np.array([self.a1, self.a2, self.a3, self.a4])
                self.d = np.array([1, 2, 1, 2])
                # expected U matrix
                self.U = np.array([[0, 1, 0],
                                   [0, 0, 1]])
                # sample prediction values
                self.p1 = np.array([.3, .3])
                self.p2 = np.array([.6, .6])
                self.p3 = np.array([.5, .4])
                self.p4 = np.array([.1, .6])
                self.p5 = np.array([0, 0])
                self.p6 = np.array([.9, .9])
                self.P = np.array([self.p1, self.p2, self.p3, self.p4, self.p5, self.p6])
                self.Z = np.array([1, 2, 2, 1, 1, 2])
        return _EX1()

    @property
    def EX_2(self):
        """
        Custom example with two overlaying boxes
        """
        class _EX2:
            def __init__(self):
                self.V = np.array([[.1,.05],[.1,.2]])
                self.W = np.array([[.3,.35],[.3,.4]])
        return _EX2()

    @property
    def EX_3(self):
        """
        Custom example with box-within-a-box
        """
        class _EX2:
            def __init__(self):
                self.V = np.array([[.1, .3], [.1, .05]])
                self.W = np.array([[.4, .7], [.4, .35]])
                self.Vj = np.array([[.25], [.29]])
                self.Wj = np.array([[.35], [.38]])
        return _EX2()

    @property
    def EX_4(self):
        """
        A variation on the example illustrated in Fig 4.d from the KnFMM paper
        Here several hyperboxes have been added, but the original overlap issue
        should be the same.
        """
        class _EX4:
            def __init__(self):
                # hyperbox classifications
                self.B_cls = np.array([2, 1, 1, 2, 1, 3, 2])
                # Fig 4.d --with added hyperboxes
                self.V = np.array([[0., 0., .1, .4, .9, .4, .3],
                                   [.9, 0., .1, .3, .9, .8, .65]])
                self.W = np.array([[.1, .2, .5, .7, 1., .6, .65],
                                   [1., .2, .5, .7, 1.1, 1., .7]])
                self.Vj = np.array([[.4], [.3]])
                self.Wj = np.array([[.7], [.7]])
                # Filtered out j, and all hyperboxes with class 2
                self.V_f = np.array([[0., .1, .9, .4],
                                     [0., .1, .9, .8]])
                self.W_f = np.array([[.2, .5, 1., .6],
                                     [.2, .5, 1.1, 1]])
                # Corrected Overlap Result (based on Fig 4.e)
                self.Ve = np.array([[.3, 0., .1, .45, .9, .4],
                                    [.6, 0., .1, .3, .9, .8]])
                self.We = np.array([[.6, .2, .45, .7, 1., .6],
                                    [.7, .2, .5, .7, 1.1, 1]])
                # Expected U-matrix
                self.U = np.array([[0, 0, 1, 0],
                                   [0, 1, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, 1, 0]])
                # some default gfmm values
                self.gfmm = GFMM()
                self.gfmm.n = 2
                self.gfmm.p = 3
                self.gfmm.hboxes = 7
                self.gfmm.B_cls = self.B_cls
                self.gfmm.V = self.V
                self.gfmm.W = self.W
        return _EX4()
    # endregion

    # region Public Methods
    def test_fit(self):
        ex = self.EX_1
        gfmm = GFMM(theta=.4)
        gfmm.fit(ex.X, ex.d)
        self.assertEqual(gfmm.V.shape, (2, 2))
        self.assertEqual(gfmm.W.shape, (2, 2))
        self.assertEqual(gfmm.hboxes, 2)
        np.testing.assert_array_equal(gfmm.V, ex.Ve)
        np.testing.assert_array_equal(gfmm.W, ex.We)

    def test_predict(self):
        ex = self.EX_1
        gfmm = ex.gfmm
        gfmm.V = ex.Ve
        gfmm.W = ex.We
        z = gfmm.predict(ex.P)
        z = np.array(z)
        self.assertEqual(len(z), 6)  # should classify six examples
        np.testing.assert_array_equal(z, ex.Z)  # verify it matches expected outputs
    # endregion

    # region Expansion
    def test_expansion(self):
        self.gfmm._initialize(self.X2)
        self.gfmm.ϴ = .4
        # first input
        self.gfmm._expansion(self.X2[0, :], self.X2[0, :], self.d2[0])
        self.assertEqual(self.gfmm.hboxes, 1)
        self.assertEqual(self.gfmm.V.shape, (2, 1))
        np.testing.assert_array_equal(self.gfmm.V, self.gfmm.W)
        np.testing.assert_array_equal(self.gfmm.V, np.array([[.1], [.1]]))
        # second input
        self.gfmm._expansion(self.X2[1, :], self.X2[1, :], self.d2[1])
        self.assertEqual(self.gfmm.hboxes, 2)
        self.assertEqual(self.gfmm.V.shape, (2, 2))
        np.testing.assert_array_equal(self.gfmm.V, np.array([[.1, .7], [.1, .7]]))
        np.testing.assert_array_equal(self.gfmm.W, np.array([[.1, .7], [.1, .7]]))
        # third input
        self.gfmm._expansion(self.X2[2, :], self.X2[2, :], self.d2[2])
        self.assertEqual(self.gfmm.hboxes, 2)
        self.assertEqual(self.gfmm.V.shape, (2, 2))
        np.testing.assert_array_equal(self.gfmm.V, np.array([[.1, .7], [.1, .7]]))
        np.testing.assert_array_equal(self.gfmm.W, np.array([[.5, .7], [.5, .7]]))

    def test_expansion_Kn(self):
        # ----- Kn == 1 -----
        s = self.CASE_STUDY_I
        s.gfmm.Kn = 1
        s.gfmm._expansion(s.a1, s.a1, 1)
        self.assertEqual(s.gfmm.hboxes, 4)
        self.assertEqual(s.gfmm.V.shape, (2, 4))
        np.testing.assert_array_equal(s.gfmm.V, s.Vb_k1)
        np.testing.assert_array_equal(s.gfmm.W, s.Wb_k1)
        # ----- Kn == 3 -----
        s = self.CASE_STUDY_I
        s.gfmm.Kn = 3
        s.gfmm._expansion(s.a1, s.a1, 1)
        self.assertEqual(s.gfmm.hboxes, 3)
        self.assertEqual(s.gfmm.V.shape, (2, 3))
        np.testing.assert_array_equal(s.gfmm.V, s.Vb)
        np.testing.assert_array_equal(s.gfmm.W, s.Wb)
        # ----- Kn > n_hyperboxes -----
        s = self.CASE_STUDY_I
        s.gfmm.Kn = 10
        s.gfmm._expansion(s.a1, s.a1, 1)
        self.assertEqual(s.gfmm.hboxes, 3)
        self.assertEqual(s.gfmm.V.shape, (2, 3))
        np.testing.assert_array_equal(s.gfmm.V, s.Vb)
        np.testing.assert_array_equal(s.gfmm.W, s.Wb)

    def test_expansion__within_existing_hyperbox(self):
        # same class
        s = self.CASE_STUDY_I
        a0 = np.array([.3, .3])
        j, d, exp = s.gfmm._expansion(a0, a0, 1)
        self.assertEqual(s.gfmm.hboxes, 3)
        self.assertEqual(j, 0)
        self.assertEqual(d, 1)
        self.assertEqual(exp, False)
        np.testing.assert_array_equal(s.gfmm.V, s.Va)
        np.testing.assert_array_equal(s.gfmm.W, s.Wa)
        # unclassified
        s = self.CASE_STUDY_I
        a0 = np.array([.3, .3])
        j, d, exp = s.gfmm._expansion(a0, a0, 0)
        self.assertEqual(s.gfmm.hboxes, 3)
        self.assertEqual(s.gfmm.V.shape, (2, 3))
        np.testing.assert_array_equal(s.gfmm.V, s.Va)
        np.testing.assert_array_equal(s.gfmm.W, s.Wa)
        self.assertEqual(j, 0)
        self.assertEqual(d, 1)
        self.assertEqual(exp, False)
    # endregion

    # region Overlap Test
    def test_overlap_test(self):
        ex = self.EX_1
        gfmm = ex.gfmm
        # KnFMM Fig 4.c (no overlap)
        gfmm.V = ex.Vc
        gfmm.W = ex.Wc
        dim, l, k = gfmm._overlap_test(1, 2)
        self.assertEqual(dim, -1)
        # KnFMM Fig 4.d (overlap)
        gfmm.V = ex.Vd
        gfmm.W = ex.Wd
        dim, l, k = gfmm._overlap_test(1, 2)
        self.assertEqual(dim, 0)
        self.assertEqual(l, 2)

    def test_overlap_test__index_k_relative_to_original(self):
        ex = self.EX_4
        gfmm = ex.gfmm
        # remove the overlapping box of the same class for now
        gfmm.V = ex.V[:, :-1]
        gfmm.W = ex.W[:, :-1]
        gfmm.B_cls = ex.B_cls[:-1]
        gfmm.hboxes = 6
        # basic tests to make sure we have the right setup
        self.assertEqual(gfmm.V.shape, (2, 6))
        self.assertEqual(len(gfmm.B_cls), 6)
        # perform overlap check
        k = 2  # index of B2
        j = 3  # index of B3
        d = 2  # class(B3)
        dim, l, idx = gfmm._overlap_test(j, d)
        # verify results
        self.assertEqual(dim, 0)  # dimension 0 (i.e. x)
        self.assertEqual(l, 2)    # case 2
        self.assertEqual(idx, k)  # overlaps with hyperbox B2

    def test_overlap_test__ignore_same_class(self):
        ex = self.EX_4
        gfmm = ex.gfmm
        # set class(B0) to be 3, so it is not filtered out when passed to min_overlap_adjustment()
        gfmm.B_cls[0] = 3
        # perform overlap check
        k = 2  # index of B2
        j = 3  # index of B3
        d = 2  # class(B3)
        dim, l, idx = gfmm._overlap_test(j, d)
        # verify results
        self.assertEqual(dim, 0)  # dimension 0 (i.e. x)
        self.assertEqual(l, 2)    # case 2
        self.assertEqual(idx, k)  # overlaps with B2, ignore B6
    # endregion

    # region Contraction
    def test_contraction(self):
        ex = self.EX_1
        gfmm = ex.gfmm
        gfmm.V = ex.Vd
        gfmm.W = ex.Wd
        # KnFMM Fig 4.d -> Fig 4.e
        gfmm._contraction(0, 2, 1, 0)
        np.testing.assert_array_equal(gfmm.V, ex.Ve)
        np.testing.assert_array_equal(gfmm.W, ex.We)

    def test_contraction__no_overlap(self):
        ex = self.EX_1
        gfmm = ex.gfmm
        gfmm.V = ex.Vc
        gfmm.W = ex.Wc
        gfmm._contraction(0, 2, 1, 0)
        # no changes
        np.testing.assert_array_equal(gfmm.V, ex.Vc)
        np.testing.assert_array_equal(gfmm.W, ex.Wc)
    # endregion

    # region Initialization Methods
    def test__initialize(self):
        # initially 0 dimensions and no hyperboxes
        self.assertIsNone(self.gfmm.n)
        self.assertEqual(self.gfmm.hboxes, 0)
        # 3 dims, 3 examples
        X_l, X_u = self.gfmm._initialize(self.X1)
        self.assertEqual(self.gfmm.V.shape, (3, 0))
        self.assertEqual(self.gfmm.W.shape, (3, 0))
        self.assertEqual(self.gfmm.n, 3)
        self.assertEqual(self.gfmm.hboxes, 0)
        self.assertEqual(X_l[0, 0], 1)
        self.assertEqual(X_u[0, 0], 1)
        # 2 dims, 4 examples
        X_l, X_u = self.gfmm._initialize(self.X2, wipe=True)
        self.assertEqual(self.gfmm.V.shape, (2, 0))
        self.assertEqual(self.gfmm.W.shape, (2, 0))
        self.assertEqual(self.gfmm.n, 2)
        self.assertEqual(self.gfmm.hboxes, 0)
        self.assertEqual(X_l[0, 0], .1)
        self.assertEqual(X_u[0, 0], .1)
        # set with different min/max values
        X_l, X_u = self.gfmm._initialize(self.X3, wipe=True)
        self.assertEqual(self.gfmm.V.shape, (2, 0))
        self.assertEqual(self.gfmm.W.shape, (2, 0))
        self.assertEqual(self.gfmm.n, 2)
        self.assertEqual(self.gfmm.hboxes, 0)
        self.assertEqual(X_l[0, 0], .1)
        self.assertEqual(X_u[0, 0], .15)
    # endregion

    # region Helper Methods
    def test__add_hyperbox(self):
        # initially none
        self.gfmm._initialize(self.X2)
        self.assertEqual(self.gfmm.hboxes, 0)
        self.assertEqual(len(self.gfmm.B_cls), 0)
        # 1 hyperbox
        xl = np.array([.1, .1])
        xu = np.array([.15, .15])
        self.gfmm._add_hyperbox(xl, xu, 1)
        self.assertEqual(self.gfmm.hboxes, 1)
        self.assertEqual(self.gfmm.V[0, 0], .1)
        self.assertEqual(self.gfmm.W[0, 0], .15)
        self.assertEqual(self.gfmm.V.shape, (2, 1))
        self.assertEqual(len(self.gfmm.B_cls), 1)
        self.assertEqual(self.gfmm.B_cls[0], 1)
        # 2 hyperboxes
        xl = np.array([.7, .7])
        xu = np.array([.75, .75])
        self.gfmm._add_hyperbox(xl, xu, 2)
        self.assertEqual(self.gfmm.hboxes, 2)
        self.assertEqual(self.gfmm.V[0, 1], .7)
        self.assertEqual(self.gfmm.V.shape, (2, 2))
        self.assertEqual(len(self.gfmm.B_cls), 2)
        self.assertEqual(self.gfmm.B_cls[1], 2)

    def test__k_best(self):
        # large list
        B = np.array([0.56107481,0.86329273,0.97936701,0.35803764,0.57650837,0.00782822,0.39214498,0.78773393,0.11900468,0.20468056, 0.55545065,0.53322588,0.8644637 ,0.73928695,0.35036804, 0.97295121,0.38587608,0.14557602,0.46625496,0.57082576, 0.90507391,0.71348483,0.97306754,0.34191463,0.08013307, 0.64183503,0.59941625,0.69198802,0.19599409,0.04848452, 0.06788202,0.57459323,0.12906092,0.11598434,0.66394427, 0.86173515,0.60943442,0.95808242,0.63095585,0.75520642, 0.4304246 ,0.68292622,0.22150663,0.97191045,0.40548108, 0.33391995,0.91687308,0.42245342,0.24562244,0.38428225, 0.83314525,0.43079164,0.29827926,0.67329566,0.2441916,0.9239678,0.53756163,0.03056051,0.27519125,0.19571112, 0.9574535 ,0.09105649,0.2649977 ,0.28807847,0.00957413, 0.41519489,0.047152,0.35971428,0.00921316,0.37833271,0.01505729,0.99262542,0.86774579,0.2119966,0.45947339,0.32850559,0.15212822,0.44581891,0.42037057,0.16470616, 0.89845261,0.76883223,0.65530583,0.64080171,0.28412596,0.97319349,0.43705441,0.43858264,0.83801105,0.34161406, 0.20119325,0.4895211,0.96748549,0.3362789,0.44988399,0.1799015,0.26523983,0.2015935,0.34865404,0.12577012])
        order = np.argsort(B)[::-1]
        k_best = GFMM.k_best(B, 10)
        np.testing.assert_array_equal(order[:10], k_best)

    def test__can_expand(self):
        # 1 result, with index reordering
        s = self.CASE_STUDY_I
        idx = s.gfmm._can_expand(np.array([0, 2, 1]), s.a1, s.a1)
        np.testing.assert_array_equal(idx, np.array([2]))
        # 1 result, with index trimming
        s = self.CASE_STUDY_I
        idx = s.gfmm._can_expand(np.array([0, 2]), s.a1, s.a1)
        np.testing.assert_array_equal(idx, np.array([2]))
        # 2 results, with index trimming
        s = self.CASE_STUDY_I
        s.gfmm.ϴ = .33
        idx = s.gfmm._can_expand(np.array([0, 2]), s.a1, s.a1)
        np.testing.assert_array_equal(idx, np.array([0, 2]))
        # 1 result, with single index
        s = self.CASE_STUDY_I
        idx = s.gfmm._can_expand(np.array([2]), s.a1, s.a1)
        np.testing.assert_array_equal(idx, np.array([2]))
        # 0 results, with single index
        s = self.CASE_STUDY_I
        idx = s.gfmm._can_expand(np.array([0]), s.a1, s.a1)
        self.assertEqual(len(idx), 0)

    def test__valid_class(self):
        s = self.CASE_STUDY_I
        idx = s.gfmm._valid_class(np.array([0, 2, 1]), 1)
        np.testing.assert_array_equal(idx, np.array([0, 2]))

    def test__expand(self):
        ex = self.EX_1
        gfmm = ex.gfmm
        gfmm.V = ex.Vb
        gfmm.W = ex.Wb
        exp = gfmm._expand(0, ex.a3, ex.a3)
        np.testing.assert_array_equal(gfmm.V, ex.Vc)
        np.testing.assert_array_equal(gfmm.W, ex.Wc)
        self.assertTrue(exp)

    def test__expand__no_expansion(self):
        ex = self.EX_1
        gfmm = ex.gfmm
        gfmm.V = ex.Vc
        gfmm.W = ex.Wc
        # point within hyperbox
        a = np.array([.3, .3])
        exp = gfmm._expand(0, a, a)
        np.testing.assert_array_equal(gfmm.V, ex.Vc)
        np.testing.assert_array_equal(gfmm.W, ex.Wc)
        self.assertFalse(exp)
        # min/max value within hyperbox
        b = np.array([.35, .4])
        exp = gfmm._expand(0, a, b)
        np.testing.assert_array_equal(gfmm.V, ex.Vc)
        np.testing.assert_array_equal(gfmm.W, ex.Wc)
        self.assertFalse(exp)

    def test__min_overlap_adjustment(self):
        ex = self.EX_4
        d, l, k = GFMM.min_overlap_adjustment(ex.V_f, ex.W_f, ex.Vj, ex.Wj)
        self.assertEqual(d, 0)  # dimension 0 (zero-indexed)
        self.assertEqual(l, 2)  # case 2 (one-indexed)
        self.assertEqual(k, 1)  # hyperbox 2, but second entry in V_f (zero-indexed)
        ex = self.EX_3
        d, l, k = GFMM.min_overlap_adjustment(ex.V, ex.W, ex.Vj, ex.Wj)
        self.assertEqual(d, 0)  # dimension 0 (zero-indexed)
        self.assertEqual(l, 1)  # case 1 (one-indexed)
        self.assertEqual(k, 1)  # second hyperbox (zero-indexed)

    def test__splice_matrix(self):
        # needs splicing
        Xl, Xu = GFMM.splice_matrix(self.X2)
        np.testing.assert_array_equal(Xl, self.X2)
        np.testing.assert_array_equal(Xu, self.X2)
        # doesn't need splicing
        Xl, Xu = GFMM.splice_matrix(self.X3)
        np.testing.assert_array_equal(Xl, self.X3[:,:,0])
        np.testing.assert_array_equal(Xu, self.X3[:,:,1])
    # endregion

    # region Properties
    def test_U(self):
        # m = 2, p = 2
        ex = self.EX_1
        gfmm = ex.gfmm
        np.testing.assert_array_equal(gfmm.U, ex.U)
        # m = 7, p = 3
        ex = self.EX_4
        gfmm = ex.gfmm
        np.testing.assert_array_equal(gfmm.U, ex.U)
    # endregion

    # run before each
    def setUp(self):
        self.gfmm = GFMM()