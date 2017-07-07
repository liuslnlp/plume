from plume.tree import DecisionTreeClassifier
from plume.knn import KNeighborClassifier
from plume.ensemble import AdaBoostClassifier, BaggingClassifier, \
    RandomForestsClassifier
import numpy as np

def test_adaboost():
    clf = AdaBoostClassifier(DecisionTreeClassifier)
    train_x = np.array([
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
    ])
    train_y = np.array([1, 1, 1, -1, -1])
    clf.fit(train_x, train_y)
    print(clf.predict(train_x))

def test_bagging():

    clfs = [KNeighborClassifier(2) for i in range(7)]
    train_x = np.array([[1, 1], [0.1, 0.1], [0.5, 0.7], [10, 10], [10, 11]])
    train_y = np.array(['A', 'A', 'A', 'B', 'B'])
    test_x = np.array([[11, 12], [12, 13], [11, 13], [0.05, 0.1]])
    b = BaggingClassifier(clfs)
    b.fit(train_x, train_y)
    print(b.predict(test_x))

def test_rf():
    clf = RandomForestsClassifier()
    train_x = np.array([
        [1, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
    ])
    train_y = np.array([1, 1, 1, -1, -1])
    print(clf.fit(train_x, train_y).predict(train_x))



if __name__ == '__main__':
    test_adaboost()
    test_bagging()
    test_rf()

