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
        # add 1
        xl = np.array([.1, .1])
        xu = np.array([.15, .15])
        self.gfmm._add_hyperbox(xl, xu)
        self.assertEqual(self.gfmm.num_hboxes, 1)
        self.assertEqual(self.gfmm.V[0, 0], .1)
        self.assertEqual(self.gfmm.W[0, 0], .15)

    def test_fit(self):
        self.fail()

    def test_predict(self):
        self.fail()

    def test__expansion(self):
        self.fail()

    def test__overlap_test(self):
        self.fail()

    def test__contraction(self):
        self.fail()
