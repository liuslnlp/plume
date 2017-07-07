from plume.svm import LinearSVC, SVC
from plume.utils import plot_decision_boundary
import sklearn.datasets
import numpy as np

np.random.seed(0)


def test_linearsvc():
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    y = 2 * y - 1
    clf = LinearSVC(C=3)
    clf.fit(X, y)
    plot_decision_boundary(clf.predict, X, y)

def test_svc_():
    train_x = np.array([[1, 1], [0.1, 0.1], [0.5, 0.7], [10, 10], [10, 11]])
    train_y = np.array([-1, -1, -1, 1, 1])
    test_x = np.array([[11, 12], [12, 13], [11, 13], [0.05, 0.1], [0.11, 0.1], [1, 2]])
    clf = SVC()
    clf.fit(train_x, train_y)
    print(clf.predict(test_x))

def test_svc():
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    y = 2 * y - 1
    clf = SVC(C=3, kernel='rbf')
    clf.fit(X, y)
    plot_decision_boundary(clf.predict, X, y, 'Support Vector Machine')

if __name__ == '__main__':
    # test_linearsvc()
    test_svc()