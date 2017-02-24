import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from mlearn.hmm import *

import numpy as np

A = [[0.5, 0.2, 0.3],
     [0.3, 0.5, 0.2],
     [0.2, 0.3, 0.5]]
A = np.array(A)

B = [[0.5, 0.5],
     [0.4, 0.6],
     [0.7, 0.3]]

B = np.array(B)

pi = np.array([0.2, 0.4, 0.4])


def test_for():
    print(forward_probability(A, B, pi, np.array([0, 1, 0])))


def test_back():
    print(backward_probability(A, B, pi, np.array([0, 1, 0])))


def test_pr_ob():
    print(predict_ob_probability(A, B, pi, np.array([0, 1, 0])))


def test_gamma():
    print(gamma(A, B, pi, np.array([0, 1, 0])))


def test_xi():
    print(xi(A, B, pi, np.array([0, 1, 0])))


def test_get_optimal_path():
    o = np.array([0, 1, 0])
    print(get_optimal_path(A, B, pi, o))


def test_model_param():
    A, B, pi = get_model_param(np.array([0, 1, 0]), 3, 2, error=0.2)
    print('A: ', A)
    print('B: ', B)
    print('pi:', pi)


if __name__ == '__main__':
    test_model_param()
