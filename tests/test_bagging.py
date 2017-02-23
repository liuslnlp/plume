import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from mlearn.bagging import Bagging
import numpy as np

def main():
    from mlearn.knn import KNeighborClassifier
    clfs = [KNeighborClassifier(2) for i in range(7)]
    train_x = np.array([[1, 1], [0.1, 0.1], [0.5, 0.7], [10, 10], [10, 11]])
    train_y = np.array(['A', 'A', 'A', 'B', 'B'])
    test_x = np.array([[11, 12], [12, 13], [11, 13], [0.05, 0.1]])
    b = Bagging(clfs)
    b.fit(train_x, train_y)
    print(b.predict(test_x))


if __name__ == '__main__':
    main()