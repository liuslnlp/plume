import numpy as np
from collections import defaultdict


class KMeans(object):
    """K-Means 聚类算法"""

    def __init__(self, n_clusters, n_iter=200):
        """
        :param n_clusters: 簇的个数
        :param n_iter: 迭代次数
        """
        self.n_iter = n_iter
        self.n_clusters = n_clusters

    def fit(self, X):
        clusters = defaultdict(list)
        indexs = np.random.choice(X.shape[0], self.n_clusters)
        self.mean_vec = X[indexs]
        for _ in range(self.n_iter):

            for x in X:
                distances = np.linalg.norm(x - self.mean_vec, axis=1)
                clusters[np.argmin(distances)].append(x)

            for k, v in clusters.items():
                self.mean_vec[k] = np.mean(np.array(v), axis=0)
            clusters.clear()

    def predict(self, X):
        return np.array([np.argmin(np.linalg.norm(x - self.mean_vec, axis=1))
                         for x in X])


class LVQ(object):
    """LVQ 聚类算法"""

    def __init__(self, epochs=100, learning_rate=0.1):
        """
        :param epochs: int 迭代轮数
        :param learning_rate: float 学习率
        """
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.labels, indices = np.unique(y, return_index=True)
        self.vec = X[indices]
        for _ in range(self.epochs * X.shape[0]):
            patch = np.random.randint(X.shape[0])
            i = np.argmin(np.linalg.norm(X[patch] - self.vec, axis=1))
            if y[patch] == self.labels[i]:
                self.vec[i] += self.learning_rate * (X[patch] - self.vec[i])
            else:
                self.vec[i] -= self.learning_rate * (X[patch] - self.vec[i])

    def predict(self, X):
        return np.array([self.labels[np.argmin(
            np.linalg.norm(x - self.vec, axis=1))] for x in X])


class GaussianMixture(object):
    """
    未完成！！！！！！！！！！！
    高斯混合聚类
    self.alphas: shape = [n_components]
    self.means: shape = [n_components, n_features]
    self.covmat: shape= [n_components, n_features, n_features]
    gamma: shape = [n_samples, n_components]
    """

    def __init__(self, n_components=1, n_iter=500):
        self.n_components = n_components
        self.n_iter = n_iter
        self.alphas = np.ones(n_components) / n_components

    def prob_density(self, mu, sigma):
        mat = []
        for i in range(self.n_components):
            tmp = [(x - mu[i]) @ np.linalg.pinv(sigma[i]) @ (x - mu[i]).T for x in self.X]
            mat.append(tmp)
        mat = np.array(mat).T
        numerator = np.exp(-0.5 * np.array(mat))
        denominator = ((2 * np.pi) ** (sigma.shape[1] / 2)) * np.sqrt(np.linalg.det(sigma))
        return numerator / denominator
        # numerator = np.exp(-0.5 * (self.X - mu).T @ np.linalg.inv(sigma) @ (self.X - mu))
        # denominator = ((2 * np.pi) ** (sigma.shape[1] / 2)) * np.sqrt(np.linalg.det(sigma))
        # return numerator / denominator

    def post_prob(self):
        prob = self.prob_density(self.means, self.covmat)
        tmp = prob * self.alphas
        return tmp / tmp.sum(axis=0)

    def fit(self, X):
        self.X = X
        self.means = X[np.random.choice(X.shape[0], self.n_components)]
        self.covmat = np.array([np.eye(X.shape[1]) * 0.1] * self.n_components)
        for _ in range(self.n_iter):
            gamma = self.post_prob()
            for i in range(self.n_components):
                balance_factor = np.sum(gamma[:, i])
                self.means[i] = np.sum(X.T * gamma[:, i], axis=1) / balance_factor
                self.covmat[i] = (gamma[:, i] * (X - self.means[i]).T @ (X - self.means[i])) / balance_factor
            self.alphas = gamma.mean(axis=0)
        gamma = self.post_prob()
        self.clusters = np.argmax(gamma, axis=1)

    def get_clusters(self):
        return self.clusters
