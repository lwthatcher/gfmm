import argparse
import csv
import json
import numpy as np
from gfmm import GFMM
from gfmm.membership import get_membership_function


def run_blobs(dataset, m_func, gamma, Kn, theta, percent_labelled):
    train_x, train_y, val_x, val_y, test_x, test_y = _get_sets(dataset)
    train_y = _unlabel(train_y, percent_labelled)
    _m = get_membership_function(m_func)
    f_name = _file_name(dataset, m_func, gamma, Kn, theta, percent_labelled)
    print("file name:", f_name)
    model = GFMM(m_func=_m, gamma=gamma, Kn=Kn, theta=theta, validation_set=(val_x, val_y), patience=4)
    # train
    model.fit(train_x, train_y)
    # test
    out = model.predict(test_x)
    correct = np.where(out == test_y)[0]
    accuracy = len(correct) / len(test_y)
    # get details
    kwargs = {'m_func':m_func, 'gamma': gamma, 'Kn': Kn, 'theta': theta, 'percent_labelled': percent_labelled}
    d = _details(model, accuracy, **kwargs)
    _save_results(d, f_name)
    print("accuracy:", d['accuracy'])
    print("# boxes:", d['num_hyperboxes'])
    print("params:", d['params'])


def _unlabel(lbls, percent_labelled):
    percent_unlabelled = 1 - percent_labelled
    p = int(len(lbls) * percent_unlabelled)
    indices = np.arange(len(lbls))
    idx = np.random.choice(indices, p, replace=False)
    lbls[idx] = 0
    return lbls


def _get_sets(dataset):
    path, n, i = _get_set_info(dataset)
    X, Y = _get_dataset_sets(path, n)
    train_x = np.vstack(X[:i])
    train_y = np.hstack(Y[:i])
    val_x = X[i]
    val_y = Y[i]
    test_x = np.vstack(X[i+1:])
    test_y = np.hstack(Y[i+1:])
    return train_x, train_y, val_x, val_y, test_x, test_y


def _get_dataset_sets(path, n):
    X = []
    Y = []
    for i in range(n):
        x, y = _load_data_file(path + str(i) + ".txt")
        X.append(x)
        Y.append(y)
    return X, Y


def _get_set_info(dataset):
    if dataset == "circles":
        path = './hacks/synthetic_sets/circles/circles_'
        n = 10
        i = 6
    elif dataset == "moons":
        path = './hacks/synthetic_sets/moons/moons_'
        n = 10
        i = 6
    elif dataset == "blobs_4D":
        path = './hacks/synthetic_sets/blobs_4D/blobs_'
        n = 10
        i = 6
    else:  # if dataset == "blobs_2D":
        path = './hacks/synthetic_sets/blobs_2D/blobs_'
        n = 5
        i = 2
    return path, n, i


def _save_results(details, name):
    path = "/local/lthatch1/" + name
    print('saving results to', path)
    with open(path, 'w') as f:
        json.dump(details, f)


def _file_name(dataset, m_func, gamma, Kn, theta, percent_labelled):
    result = dataset + "_"
    result += m_func[0] + "_g" + str(int(gamma)) + "_k" + str(Kn)
    result += "_t" + str(int(theta*100)) + "_p" + str(int(percent_labelled*100))
    result += ".json"
    return result


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
    result = {'accuracy': accuracy, 'V': model.V.tolist(), 'W': model.W.tolist(), 'hyperboxes': model.B_cls.tolist(),
              'num_hyperboxes': model.m, 'epochs': model._epoch, 'params': kwargs}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', '-D', default='blobs_2D', choices=['blobs_2D', 'blobs_4D', 'circles', 'moons'],
                        help="membership function")
    parser.add_argument('--membership_func', '-m', default='standard', choices=['standard', 'cluster', 'general'],
                        help="membership function")
    parser.add_argument('--gamma', '-g', default=4., type=float, help="the gamma value to use")
    parser.add_argument('-Kn', type=int, default=10, help="the Kn value to use")
    parser.add_argument('--theta', '-t', default=.6, type=float, help="the theta value to use")
    parser.add_argument('--percent_labelled', '-l', type=float, default=1.,
                        help="the percentage of the training set to set as labelled")
    args = parser.parse_args()
    # run the test
    run_blobs(args.data_set, args.membership_func, args.gamma, args.Kn, args.theta, args.percent_labelled)
