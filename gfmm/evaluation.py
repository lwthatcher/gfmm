import numpy as np
from gfmm import GFMM


def cross_fold(X, Y, n, **gfmm_kwargs):
    num_examples = len(Y)
    # shuffle data first
    idx = np.arange(num_examples)
    np.random.shuffle(idx)
    # split into n sets
    splits = np.split(idx, n)
    for i in range(n):
        # get train/test sets
        idx_test = splits[i]
        tr1 = splits[:i]
        tr2 = splits[i+1:]
        tr1.extend(tr2)
        idx_train = np.concatenate(tr1)
        X_train = X[idx_train,:]
        X_test = X[idx_test,:]
        Y_train = Y[idx_train]
        Y_test = Y[idx_test]
        # create new model
        gfmm = GFMM(**gfmm_kwargs)
        # train
        gfmm.fit(X_train, Y_train)
        # compare
        out = gfmm.predict(X_test)
        correct = np.where(out == Y_test)[0]
        # for now, just print results:
        accuracy = len(correct) / len(Y_test)
        print(i, accuracy)
