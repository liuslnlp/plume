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


hmm = HMMEstimator(A, B, pi)

def test_forward():
    print(hmm.forward_prob(obs))

def test_backward():
    print(hmm.backward_prob(obs))

def test_ecodeing():
    path, prob = hmm.decoding(obs)
    print("path:", path)
    print("prob:", prob)

if __name__ == '__main__':
    # test_forward()
    # test_backward()
    test_ecodeing()