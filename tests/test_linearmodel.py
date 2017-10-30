from plume.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from plume.utils import plot_decision_boundary, gen_reg_data
import matplotlib.pyplot as plt
import numpy as np


def test_linear_regression():
    clf = LinearRegression()
    X, y = gen_reg_data()
    clf.fit(X, y)
    plt.plot(X, y, '.')
    X_axis = np.arange(-5, 75, 0.1)
    plt.plot(X_axis, clf.predict(X_axis))
    plt.title("Linear Regression")
    plt.show()

def test_lasso():
    clf = Lasso()
    X, y = gen_reg_data()
    clf.fit(X, y)
    plt.plot(X, y, '.')
    X_axis = np.arange(-5, 75, 0.1).reshape((-1, 1))
    plt.plot(X_axis, clf.predict(X_axis))
    plt.title("Lasso")
    plt.show()

def test_ridge():
    clf = Ridge()
    X, y = gen_reg_data()
    clf.fit(X, y)
    plt.plot(X, y, '.')
    X_axis = np.arange(-5, 75, 0.1)
    plt.plot(X_axis, clf.predict(X_axis.reshape((-1, 1))))
    plt.title("Ridge")
    plt.show()

def test_lr():
    clf = LogisticRegression()
    import sklearn.datasets
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    clf.fit(X, y)
    plot_decision_boundary(clf.predict, X, y, 'Logistic Regression')

if __name__ == '__main__':
    test_lasso()
    # test_ridge()
    # test_linear_regression()
