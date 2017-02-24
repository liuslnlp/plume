import os
import sys

sys.path.insert(0, os.path.abspath('.'))

from mlearn.kmeans import KMeans
import numpy as np


def main():
    clf = KMeans(3)
    train_x = [
        [1, 1], [0.9, 1.2], [1.3, 0.8],
        [8, 8], [8.1, 7.9], [8.2, 7.8],
        [9, 0], [9, 0.2], [9.2, 0.3]
    ]
    train_x = np.array(train_x)
    clf.fit(train_x)
    print(clf.clusters)
    print(clf.predict_one(np.array([9, 9])))


if __name__ == '__main__':
    main()
