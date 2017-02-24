import os
import sys

sys.path.insert(0, os.path.abspath('.'))

from mlearn.svm import SVC
from mlearn.utils import plot_decision_boundary
import numpy as np


def test():
    import matplotlib.pyplot as plt
    import sklearn
    import sklearn.datasets
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    for i in range(y.shape[0]):
        y[i] = -1 if y[i] == 0 else y[i]
    clf = SVC(C=5)
    clf.fit(X, y)
    plot_decision_boundary(clf.predict, X, y)
    print('over')


def test1():
    train_x = np.array([[1, 1], [0.1, 0.1], [0.5, 0.7], [10, 10], [10, 11]])
    train_y = np.array([-1, -1, -1, 1, 1])
    test_x = np.array([[11, 12], [12, 13], [11, 13], [0.05, 0.1], [0.11, 0.1], [1, 2]])
    clf = SVC(C=5)
    clf.fit(train_x, train_y)
    print(clf.predict(test_x))


if __name__ == '__main__':
    test1()
    test()
