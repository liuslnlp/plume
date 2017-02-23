import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from mlearn.linear_model import LogisticRegressionClassifier
from mlearn.utils import plot_decision_boundary
import numpy as np

def main():
    import sklearn.datasets
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    clf = LogisticRegressionClassifier()
    clf.fit(X, y)
    plot_decision_boundary(clf.predict, X, y)

if __name__ == '__main__':
    main()