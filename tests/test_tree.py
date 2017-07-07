from plume.tree import DecisionTreeClassifier
import numpy as np


def test_clf():
    clf = DecisionTreeClassifier()
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
    test_clf()
