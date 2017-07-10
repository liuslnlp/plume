from plume.hmm import *
import numpy as np

A = np.array([[0.5, 0.2, 0.3],
              [0.3, 0.5, 0.2],
              [0.2, 0.3, 0.5]])

B = np.array([[0.5, 0.5],
              [0.4, 0.6],
              [0.7, 0.3]])

pi = np.array([0.2, 0.4, 0.4])

obs = np.array([0, 1, 0])

def test_forward():
    print(forward_prob(A, B, pi, obs))

def test_backward():
    print(backward_prob(A, B, pi, obs))



if __name__ == '__main__':
    test_forward()
    test_backward()