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

    def test__initialize(self):
        # initially 0 dimensions and no hyperboxes
        self.assertEqual(self.gfmm.n, 0)
        self.assertEqual(self.gfmm.num_hboxes, 0)
        # 3 dims, 3 examples
        self.gfmm._initialize(self.X1)
        self.assertEqual(self.gfmm.V.shape, (3, 0))
        self.assertEqual(self.gfmm.W.shape, (3, 0))
        self.assertEqual(self.gfmm.n, 3)
        self.assertEqual(self.gfmm.num_hboxes, 0)
        self.assertEqual(self.gfmm.X_l[0, 0], 1)
        self.assertEqual(self.gfmm.X_u[0, 0], 1)
        # 2 dims, 4 examples
        self.gfmm._initialize(self.X2)
        self.assertEqual(self.gfmm.V.shape, (2, 0))
        self.assertEqual(self.gfmm.W.shape, (2, 0))
        self.assertEqual(self.gfmm.n, 2)
        self.assertEqual(self.gfmm.num_hboxes, 0)
        self.assertEqual(self.gfmm.X_l[0, 0], .1)
        self.assertEqual(self.gfmm.X_u[0, 0], .1)
        # set with different min/max values
        self.gfmm._initialize(self.X3)
        self.assertEqual(self.gfmm.V.shape, (2, 0))
        self.assertEqual(self.gfmm.W.shape, (2, 0))
        self.assertEqual(self.gfmm.n, 2)
        self.assertEqual(self.gfmm.num_hboxes, 0)
        self.assertEqual(self.gfmm.X_l[0, 0], .1)
        self.assertEqual(self.gfmm.X_u[0, 0], .15)

    def test__add_hyperbox(self):
        # initially none
        self.gfmm._initialize(self.X2)
        self.assertEqual(self.gfmm.num_hboxes, 0)
        # 1 hyperbox
        xl = np.array([.1, .1])
        xu = np.array([.15, .15])
        self.gfmm._add_hyperbox(xl, xu)
        self.assertEqual(self.gfmm.num_hboxes, 1)
        self.assertEqual(self.gfmm.V[0, 0], .1)
        self.assertEqual(self.gfmm.W[0, 0], .15)
        self.assertEqual(self.gfmm.V.shape, (2, 1))
        # 2 hyperboxes
        xl = np.array([.7, .7])
        xu = np.array([.75, .75])
        self.gfmm._add_hyperbox(xl, xu)
        self.assertEqual(self.gfmm.num_hboxes, 2)
        self.assertEqual(self.gfmm.V[0, 1], .7)
        self.assertEqual(self.gfmm.V.shape, (2, 2))

    def test_fit(self):
        Vf = np.array([[.1, .45],
                       [.1, .3]])
        Wf = np.array([[.45, .7],
                       [.5, .7]])
        self.gfmm.fit(self.X2, self.d2)
        self.assertEqual(self.gfmm.V.shape, (2, 2))
        self.assertEqual(self.gfmm.W.shape, (2, 2))
        self.assertEqual(self.gfmm.num_hboxes, 2)
        np.testing.assert_array_equal(self.gfmm.V, Vf)
        np.testing.assert_array_equal(self.gfmm.W, Wf)

    def test__expansion(self):
        self.gfmm._initialize(self.X2)
        # first input
        self.gfmm._expansion(self.X2[0, :], self.X2[0, :], self.d2[0])
        self.assertEqual(self.gfmm.num_hboxes, 1)
        self.assertEqual(self.gfmm.V.shape, (2, 1))
        np.testing.assert_array_equal(self.gfmm.V, self.gfmm.W)
        np.testing.assert_array_equal(self.gfmm.V, np.array([[.1], [.1]]))
        # second input
        self.gfmm._expansion(self.X2[1, :], self.X2[1, :], self.d2[1])
        self.assertEqual(self.gfmm.num_hboxes, 2)
        self.assertEqual(self.gfmm.V.shape, (2, 2))
        np.testing.assert_array_equal(self.gfmm.V, np.array([[.1, .7], [.1, .7]]))
        np.testing.assert_array_equal(self.gfmm.W, np.array([[.1, .7], [.1, .7]]))
        # third input
        self.gfmm._expansion(self.X2[2, :], self.X2[2, :], self.d2[2])
        self.assertEqual(self.gfmm.num_hboxes, 2)
        self.assertEqual(self.gfmm.V.shape, (2, 2))
        np.testing.assert_array_equal(self.gfmm.V, np.array([[.1, .7], [.1, .7]]))
        np.testing.assert_array_equal(self.gfmm.W, np.array([[.5, .7], [.5, .7]]))

    def test__overlap_test(self):
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
        self.assertNotEqual(d, -1)
        self.assertNotEqual(l, 0)

    def test__contraction(self):
        self.fail()
