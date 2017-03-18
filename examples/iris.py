from gfmm import GFMM
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
print("loaded IRIS data set")
data = iris.data
target = iris.target
target += 1
print("loaded IRIS data set")
gfmm = GFMM(p=3)
out = gfmm.fit(data, target)
out = np.array(out)
print("1 epoch results:", target == out)