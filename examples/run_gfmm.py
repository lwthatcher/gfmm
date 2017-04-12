import argparse
import csv
import numpy as np
from gfmm import GFMM
from gfmm.membership import get_membership_function


def run_blobs(m_func, gamma, Kn, theta, percent_labelled):
    train_x, train_y, val_x, val_y, test_x, test_y = _get_sets()
    train_y = _unlabel(train_y, percent_labelled)
    model = GFMM(m_func=m_func, gamma=gamma, n=2, p=3, Kn=Kn, theta=theta)
    # train
    model.fit(train_x, train_y)
    # test
    out = model.predict(test_x)
    correct = np.where(out == test_y)[0]
    accuracy = len(correct) / len(test_y)
    # get details
    kwargs = {'m_func':m_func, 'gamma': gamma, 'Kn': Kn, 'theta': theta, 'percent_labelled': percent_labelled}
    d = _details(model, accuracy, **kwargs)
    print(d)


def _unlabel(lbls, percent_labelled):
    percent_unlabelled = 1 - percent_labelled
    p = int(len(lbls) * percent_unlabelled)
    indices = np.arange(len(lbls))
    idx = np.random.choice(indices, p, replace=False)
    lbls[idx] = 0
    return lbls


def _get_sets():
    X, Y = _get_blob_sets()
    train_x = np.vstack(X[0:2])
    train_y = np.hstack(Y[0:2])
    val_x = X[2]
    val_y = Y[2]
    test_x = np.vstack(X[3:])
    test_y = np.hstack(Y[3:])
    return train_x, train_y, val_x, val_y, test_x, test_y


def _get_blob_sets():
    X = []
    Y = []
    for i in range(5):
        x, y = _load_data_file("blobs_" + str(i) + ".txt")
        X.append(x)
        Y.append(y)
    return X, Y


def _load_data_file(file):
    features = []
    labels = []
    with open(file, newline='') as csv_file:
        data_reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
        for row in data_reader:
            labels.append(row[-1])
            features.append(row[:-1])
    return np.array(features), np.array(labels)


def _details(model, accuracy, **kwargs):
    result = {'accuracy': accuracy, 'V': model.V, 'W': model.W, 'hyperboxes': model.B_cls,
              'num_hyperboxes': len(model.B_cls), 'epochs': model._epoch, 'params': kwargs}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--membership_func', '-m', default='standard', choices=['standard', 'cluster', 'general'],
                        help="membership function")
    parser.add_argument('--gamma', '-g', default=4., type=float, help="the gamma value to use")
    parser.add_argument('-Kn', type=int, default=10, help="the Kn value to use")
    parser.add_argument('--theta', '-t', default=.6, type=float, help="the theta value to use")
    parser.add_argument('--percent_labelled', '-l', type=float, default=1.,
                        help="the percentage of the training set to set as labelled")
    args = parser.parse_args()
    # run the test
    _m = get_membership_function(args.membership_func)
    run_blobs(_m, args.gamma, args.Kn, args.theta, args.percent_labelled)
