from unittest import TestCase
from GFMM import GFMM


class TestGFMM(TestCase):

    def setUp(self):
        self.gfmm = GFMM()

    def test__initialize(self):
        self.gfmm._initialize(5)
        print(self.gfmm.V)
        self.assertEqual(self.gfmm.V.shape, (5, 1))
        self.assertEqual(self.gfmm.W.shape, (5, 1))

        self.gfmm.mfunc([0, 0])

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
