import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from mlearn.network import BPNetWork
import numpy as np


def main():
    train_x = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    train_y = np.array([[0], [1], [1], [0]])
    n = BPNetWork((2, 2, 1))
    n.fit(train_x, train_y, 10000, 0.1)
    for x in train_x:
        print(x, n.predict(x))

if __name__ == '__main__':
    main()