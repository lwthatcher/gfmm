from unittest import TestCase
import numpy as np
from gfmm import membership
from gfmm import GFMM


class TestMembershipFunctions(TestCase):

    def setUp(self):
        self.mock_parent = GFMM()
        self.mock_parent.W = np.arange(12).reshape(3, 4)
        self.mock_parent.V = np.arange(12).reshape(3, 4)

    def test_property__n(self):
        f = membership.FuzzyMembershipFunction(self.mock_parent, gamma=1.6)
        self.assertEqual(f.n, 3)
        self.mock_parent.n = 4
        self.assertEqual(f.n, 4)

    def test_classification(self):
        f = membership.FuzzyMembershipFunction(self.mock_parent, gamma=1.6)
        a = np.array([2., 5.3, 10.2])
        bj = f(a, a)
        expected = np.array([0.5, .58666667, 0.78, 0.5])
        np.testing.assert_array_almost_equal(bj, expected)

    def test_classification__gamma_as_vector(self):
        gamma = np.array([1.6, 1.6, 1.6])
        f = membership.FuzzyMembershipFunction(self.mock_parent, gamma=gamma)
        a = np.array([2., 5.3, 10.2])
        bj = f(a, a)
        expected = np.array([0.5, .58666667, 0.78, 0.5])
        np.testing.assert_array_almost_equal(bj, expected)

    def test_clustering(self):
        f = membership.Clustering(self.mock_parent, gamma=1.6)
        a = np.array([2., 5.2, 10.2])
        bj = f(a, a)
        expected = np.array([0.0, .22666667, 0.56, 0.0])
        np.testing.assert_array_almost_equal(bj, expected)
