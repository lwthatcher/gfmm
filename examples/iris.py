from gfmm import GFMM
from sklearn.datasets import load_iris
import numpy as np


def run_iris():
    iris = load_iris()
    print("loaded IRIS data set")
    data = iris.data
    target = iris.target
    print("relabelling target values")
    target += 1
    target.astype(int)
    print("shuffling data")
    idx = np.arange(len(target))
    np.random.shuffle(idx)
    idx_train = idx[:-20]
    idx_test = idx[-20:]
    X_train = data[idx_train,:]
    Y_train = target[idx_train]
    X_test = data[idx_test,:]
    Y_test = target[idx_test]
    # X = data[idx,:]
    # Y = target[idx]
    # print("splitting into folds")
    # Xs = np.split(X, 10)
    # Ys = np.split(Y, 10)
    print("creating classifier")
    gfmm = GFMM(p=3)
    gfmm.fit(X_train, Y_train)
    out = gfmm.predict(X_test)
    print("Actual")
    print(Y_test)
    print("Predicted")
    print(out)


if __name__ == "__main__":
    run_iris()
