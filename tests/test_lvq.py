import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from mlearn.lvq import LVQ
import numpy as np


def main():
    clf = LVQ()
    train_x = [
        [1, 1], [0.9, 1.2], [1.3, 0.8],
        [8, 8], [8.1, 7.9], [8.2, 7.8],
        [9, 0], [9, 0.2], [9.2, 0.3]
    ]
    train_y = [
        "左下", "左下", "左下",
        "右上", "右上", "右上",
        "右下", "右下", "右下",
    ]
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    clf.fit(train_x, train_y)
    test_x = np.array([[1, 1], [9, 0],[9, 9]])
    print(clf.predict(test_x))

if __name__ == '__main__':
    main()