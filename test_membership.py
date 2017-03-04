from unittest import TestCase
from GFMM import GFMM
import membership
import numpy as np


class TestMembershipFunctions(TestCase):

    def setUp(self):
        self.mock_parent = GFMM()
        self.mock_parent.W = np.arange(12).reshape(3, 4)
        self.mock_parent.V = np.arange(12).reshape(3, 4)

    def test_property__n(self):
        f = membership.Classification(self.mock_parent, gamma=1.6)
        self.assertEqual(f.n, 3)
        self.mock_parent.n = 4
        self.assertEqual(f.n, 4)

    def test_classification(self):
        f = membership.Classification(self.mock_parent, gamma=1.6)
        a = np.array([2., 5.3, 10.2])
        bj = f(a)
        expected = np.array([0.5, .58666667, 0.78, 0.5])
        np.testing.assert_array_almost_equal(bj, expected)
