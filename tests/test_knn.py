from plume.knn import KNeighborClassifier
from plume.utils import plot_decision_boundary
import numpy as np
import unittest


class KNNTestCase(unittest.TestCase):
    def setUp(self):
        self._clf = KNeighborClassifier(3)
        self._train_x = np.array([[1, 1],
                                  [0.1, 0.1],
                                  [0.5, 0.7],
                                  [10, 10],
                                  [10, 11]])
        self._train_y = np.array(['A', 'A', 'A', 'B', 'B'])
        self._test_x = np.array([[11, 12],
                                 [12, 13],
                                 [11, 13],
                                 [0.05, 0.1]])
        self._clf.fit(self._train_x, self._train_y)
    def test_predict(self):
        y_pred = self._clf.predict(self._test_x)
        assert np.all(y_pred == self._train_y[:-1])

def test_knn():
    import sklearn.datasets
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    y = 2 * y - 1
    clf = KNeighborClassifier(3)
    clf.fit(X, y)
    plot_decision_boundary(clf.predict, X, y, 'K Neighbor Classifier')

if __name__ == '__main__':
    # unittest.main()
    test_knn()