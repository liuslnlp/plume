import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from mlearn.knn import ParallelKNClassifier
from mlearn.utils import plot_decision_boundary
import numpy as np

def main():
    train_x = np.array([[1, 1], [0.1, 0.1], [0.5, 0.7], [10, 10], [10, 11]] )
    train_y = np.array(['A', 'A', 'A', 'B', 'B'])
    test_x = np.array([[11, 12], [12, 13], [11, 13], [0.05, 0.1]])

    k = ParallelKNClassifier(3)
    k.fit(train_x, train_y)
    print(k.predict(test_x))
    import sklearn.datasets
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    clf = ParallelKNClassifier()
    clf.fit(X, y)
    plot_decision_boundary(clf.predict, X, y)


if __name__ == '__main__':
    main()