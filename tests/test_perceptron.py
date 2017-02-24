import os
import sys

sys.path.insert(0, os.path.abspath('.'))

from mlearn.perceptron import Perceptron
import numpy as np


def main():
    train_x = np.array([[3, 3], [4, 3], [1, 1]])
    train_y = np.array([1, 1, -1])
    eta = 1
    w0 = [0, 0]
    b0 = 0
    p = Perceptron(w0, b0, eta)
    p.fit(train_x, train_y)
    print(p.predict(train_x))


if __name__ == '__main__':
    main()
