from plume.network import FullyConnNet
from plume.utils import plot_decision_boundary
import sklearn.datasets
import numpy as np

def test_mlp():
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    n = FullyConnNet((2, 3, 1), activation='relu', epochs=1000, learning_rate=0.01)
    n.fit(X, y)
    def tmp(X):
        sign = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
        ans = sign(n.predict(X))
        return ans

    plot_decision_boundary(tmp, X, y, 'Neural Network')



if __name__ == '__main__':
    test_mlp()
