from plume.linear_model import LinearRegression, LogisticRegression
from plume.utils import plot_decision_boundary
import numpy as np


def test_linear_regression():
    clf = LinearRegression()
    X = np.array([[1, 3], [2, 5], [3, 7], [4, 9]])
    clf.fit(X[:, 0], X[:, 1])
    print(clf.predict(np.array([[5], [6], [7], [8]])))

def test_lr():
    clf = LogisticRegression()
    import sklearn.datasets
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    clf.fit(X, y)
    plot_decision_boundary(clf.predict, X, y, 'Logistic Regression')

if __name__ == '__main__':
    test_lr()
    # test_linear_regression()
