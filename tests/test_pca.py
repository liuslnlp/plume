import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from mlearn.pca import pca
import numpy as np


def main():
    data = np.array([[1., 1.],
           [0.9, 0.95],
           [1.01, 1.03],
           [2., 2.],
           [2.03, 2.06],
           [1.98, 1.89],
           [3., 3.],
           [3.03, 3.05],
           [2.89, 3.1],
           [4., 4.],
           [4.06, 4.02],
           [3.97, 4.01]])
    from sklearn.decomposition import PCA
    _pca = PCA(n_components=1)
    newData = _pca.fit_transform(data)
    print(newData)
    print(pca(data, 1))

if __name__ == '__main__':
    main()