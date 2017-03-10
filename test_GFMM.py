from unittest import TestCase
from GFMM import GFMM
import numpy as np


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

    # run before each
    def setUp(self):
        self.gfmm = GFMM()

    @property
    def CASE_STUDY_I(self):
        class _CSI:
            def __init__(self):
                self.Va = np.array([[.1, .55, .2],
                                    [.2, .2, .6]])
                self.Wa = np.array([[.4, .75, .25],
                                    [.5, .4, .7]])
                self.gfmm = GFMM()
                self.gfmm.n = 2
                self.gfmm.hboxes = 3
                self.gfmm.V = np.copy(self.Va)
                self.gfmm.W = np.copy(self.Wa)
                self.gfmm.B_cls = [1, 2, 1]
                self.gfmm.ϴ = .301
                self.a1 = np.array([.3, .53])
                self.a2 = np.array([.38, .55])
                self.a3 = np.array([.1, .59])
        return _CSI()

    def test__initialize(self):
        # initially 0 dimensions and no hyperboxes
        self.assertEqual(self.gfmm.n, 0)
        self.assertEqual(self.gfmm.hboxes, 0)
        # 3 dims, 3 examples
        self.gfmm._initialize(self.X1)
        self.assertEqual(self.gfmm.V.shape, (3, 0))
        self.assertEqual(self.gfmm.W.shape, (3, 0))
        self.assertEqual(self.gfmm.n, 3)
        self.assertEqual(self.gfmm.hboxes, 0)
        self.assertEqual(self.gfmm.X_l[0, 0], 1)
        self.assertEqual(self.gfmm.X_u[0, 0], 1)
        # 2 dims, 4 examples
        self.gfmm._initialize(self.X2)
        self.assertEqual(self.gfmm.V.shape, (2, 0))
        self.assertEqual(self.gfmm.W.shape, (2, 0))
        self.assertEqual(self.gfmm.n, 2)
        self.assertEqual(self.gfmm.hboxes, 0)
        self.assertEqual(self.gfmm.X_l[0, 0], .1)
        self.assertEqual(self.gfmm.X_u[0, 0], .1)
        # set with different min/max values
        self.gfmm._initialize(self.X3)
        self.assertEqual(self.gfmm.V.shape, (2, 0))
        self.assertEqual(self.gfmm.W.shape, (2, 0))
        self.assertEqual(self.gfmm.n, 2)
        self.assertEqual(self.gfmm.hboxes, 0)
        self.assertEqual(self.gfmm.X_l[0, 0], .1)
        self.assertEqual(self.gfmm.X_u[0, 0], .15)

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

    def test_fit(self):
        Vf = np.array([[.1, .45],
                       [.1, .3]])
        Wf = np.array([[.45, .7],
                       [.5, .7]])
        self.gfmm.fit(self.X2, self.d2)
        self.assertEqual(self.gfmm.V.shape, (2, 2))
        self.assertEqual(self.gfmm.W.shape, (2, 2))
        self.assertEqual(self.gfmm.hboxes, 2)
        np.testing.assert_array_equal(self.gfmm.V, Vf)
        np.testing.assert_array_equal(self.gfmm.W, Wf)

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
        Vb = np.array([[.1, .55, .2, .3],
                       [.2, .2, .6, .53]])
        Wb = np.array([[.4, .75, .25, .3],
                       [.5, .4, .7, .53]])
        self.assertEqual(s.gfmm.hboxes, 4)
        self.assertEqual(s.gfmm.V.shape, (2, 4))
        np.testing.assert_array_equal(s.gfmm.V, Vb)
        np.testing.assert_array_equal(s.gfmm.W, Wb)
        # ----- Kn == 3 -----
        s = self.CASE_STUDY_I
        s.gfmm.Kn = 3
        s.gfmm._expansion(s.a1, s.a1, 1)
        Vb = np.array([[.1, .55, .2],
                       [.2, .2, .53]])
        Wb = np.array([[.4, .75, .3],
                       [.5, .4, .7]])
        self.assertEqual(s.gfmm.hboxes, 3)
        self.assertEqual(s.gfmm.V.shape, (2, 3))
        np.testing.assert_array_equal(s.gfmm.V, Vb)
        np.testing.assert_array_equal(s.gfmm.W, Wb)
        # ----- Kn > n_hyperboxes -----
        s = self.CASE_STUDY_I
        s.gfmm.Kn = 10
        s.gfmm._expansion(s.a1, s.a1, 1)
        Vb = np.array([[.1, .55, .2],
                       [.2, .2, .53]])
        Wb = np.array([[.4, .75, .3],
                       [.5, .4, .7]])
        self.assertEqual(s.gfmm.hboxes, 3)
        self.assertEqual(s.gfmm.V.shape, (2, 3))
        np.testing.assert_array_equal(s.gfmm.V, Vb)
        np.testing.assert_array_equal(s.gfmm.W, Wb)

    def test_expansion__within_existing_hyperbox(self):
        # same class
        s = self.CASE_STUDY_I
        a0 = np.array([.3, .3])
        s.gfmm._expansion(a0, a0, 1)
        self.assertEqual(s.gfmm.hboxes, 3)
        self.assertEqual(s.gfmm.V.shape, (2, 3))
        np.testing.assert_array_equal(s.gfmm.V, s.Va)
        np.testing.assert_array_equal(s.gfmm.W, s.Wa)
        # unclassified
        s = self.CASE_STUDY_I
        a0 = np.array([.3, .3])
        s.gfmm._expansion(a0, a0, 0)
        self.assertEqual(s.gfmm.hboxes, 3)
        self.assertEqual(s.gfmm.V.shape, (2, 3))
        np.testing.assert_array_equal(s.gfmm.V, s.Va)
        np.testing.assert_array_equal(s.gfmm.W, s.Wa)

    def test_overlap_test(self):
        self.gfmm._initialize(self.X2)
        # no initial overlap
        d, l = self.gfmm._overlap_test()
        self.assertEqual(d, -1)
        # no overlap yet
        Vc = np.array([[.1, .7], [.1, .7]])
        Wc = np.array([[.5, .7], [.5, .7]])
        self.gfmm.V = Vc
        self.gfmm.W = Wc
        d, l = self.gfmm._overlap_test()
        self.assertEqual(d, -1)
        # overlap
        Vd = np.array([[.1, .4], [.1, .3]])
        Wd = np.array([[.5, .7], [.5, .7]])
        self.gfmm.V = Vd
        self.gfmm.W = Wd
        d, l = self.gfmm._overlap_test()
        self.assertEqual(d, 0)
        self.assertEqual(l, 2)

    def test_contraction(self):
        self.gfmm._initialize(self.X2)
        Vd = np.array([[.1, .4], [.1, .3]])
        Wd = np.array([[.5, .7], [.5, .7]])
        self.gfmm.V = Vd
        self.gfmm.W = Wd
        self.gfmm._contraction(0, 2)
        Ve = np.array([[.1, .45], [.1, .3]])
        We = np.array([[.45, .7], [.5, .7]])
        np.testing.assert_array_equal(self.gfmm.V, Ve)
        np.testing.assert_array_equal(self.gfmm.W, We)

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
        self.gfmm._initialize(self.X2)
        Vb = np.array([[.1, .7], [.1, .7]])
        Wb = np.array([[.1, .7], [.1, .7]])
        Vc = np.array([[.1, .7], [.1, .7]])
        Wc = np.array([[.5, .7], [.5, .7]])
        self.gfmm.V = Vb
        self.gfmm.W = Wb
        a3 = np.array([.5, .5])
        self.gfmm._expand(0, a3, a3)
        np.testing.assert_array_equal(self.gfmm.V, Vc)
        np.testing.assert_array_equal(self.gfmm.W, Wc)
