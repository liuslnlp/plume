from plume.network import MLPClassifier
from plume.utils import plot_decision_boundary
import sklearn.datasets
import numpy as np

def test_mlp():
    X, y = sklearn.datasets.make_moons(200, noise=0.20)
    y = y.reshape((-1, 1))
    n = MLPClassifier((2, 4, 1), activation='relu', epochs=300, learning_rate=0.01)
    n.fit(X, y)
    def tmp(X):
        sign = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
        ans = sign(n.predict(X))
        return ans

    plot_decision_boundary(tmp, X, y, 'Neural Network (relu)')



if __name__ == '__main__':
    test_mlp()
