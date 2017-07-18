import numpy as np


class PCA(object):
    """主成分分析法"""
    def __init__(self, n_components=2):
        """
        :param n_components: 簇的个数
        """
        self.n_components = n_components

    def check_X(self, X):
        return X - np.mean(X, axis=0)

    def fit_transform(self, X):
        X = self.check_X(X)
        U, S, V = np.linalg.svd(X)
        U = U[:, :self.n_components]
        return U * S[:self.n_components]


class MDS(object):
    """MDS 降维算法"""
    def __init__(self, n_components=2):
        self.n_components = n_components

    def distance(self, X):
        dis = [np.linalg.norm(X[i] - X, axis=1) for i in range(X.shape[0])]
        return np.array(dis) ** 2

    def dist_i(self, dist):
        return np.mean(dist, axis=0)

    def dist_j(self, dist):
        return np.mean(dist, axis=1)

    def dist_(self, dist):
        return np.mean(dist)

    def fit_transform(self, X):
        dis = self.distance(X)
        B = -0.5 * (dis - self.dist_i(dis).reshape((-1, 1)) - self.dist_j(dis).reshape((1, -1)) + self.dist_(dis))
        w, v = np.linalg.eig(B)
        indices = np.argsort(w)[:-(self.n_components + 1):-1]
        w = w[indices]
        v = v[:, indices]
        return np.sqrt(w) * v

