from unittest import TestCase
from GFMM import GFMM


class TestGFMM(TestCase):

    def test_constructor(self):
        gfmm = GFMM(5)
        self.assertEqual(gfmm.V.shape, (5, 0))
        self.assertEqual(gfmm.W.shape, (5, 0))

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
