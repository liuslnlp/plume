from plume.decomposition import PCA, MDS
import numpy as np


def test_pca():
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
    print(PCA(1).fit_transform(data))

def test_mds():
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
    print(MDS(1).fit_transform(data))

if __name__ == '__main__':
    test_mds()