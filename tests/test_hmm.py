import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from mlearn.hmm import HMM
import numpy as np

def main():
    A = [[0.5, 0.2, 0.3],
         [0.3, 0.5, 0.2],
         [0.2, 0.3, 0.5]]
    A = np.array(A)

    B = [[0.5, 0.5],
         [0.4, 0.6],
         [0.7, 0.3]]

    B = np.array(B)

    pi = [0.2, 0.4, 0.4]
    V = ["红", "白"]

    h = HMM(A, B, V, pi)
    p = h.predict(['红', '白', '红'])
    print(p)
    p = h.predict(['红', '白', '红'], "else")
    print(p)


if __name__ == '__main__':
    main()