from gfmm import GFMM
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
import numpy as np

from gfmm.evaluation import cross_fold


def run_iris():
    print("loading IRIS data set")
    iris = load_iris()

    print("normalizing data")
    data = iris.data
    data = normalize(data, axis=0)

    print("relabelling target values")
    target = iris.target
    target += 1
    target.astype(int)

    print("shuffling data, and dividing into Train/Test sets")
    kwargs = {"p":3}
    cross_fold(data, target, 10, **kwargs)


if __name__ == "__main__":
    run_iris()
