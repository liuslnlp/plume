import numpy as np
from collections import defaultdict


class KMeans(object):
    """K-Means 聚类算法"""

    def __init__(self, n_clusters, n_iter=200):
        self.n_iter = n_iter
        self.n_clusters = n_clusters

    def fit(self, X):
        clusters = defaultdict(list)
        indexs = np.random.choice(X.shape[0], self.n_clusters)
        self.mean_vec = X[indexs]
        for _ in range(self.n_iter):

            for x in X:
                distances = np.linalg.norm(x - self.mean_vec, axis=1)
                clusters[np.argmax(distances)].append(x)

            for k, v in clusters.items():
                self.mean_vec[k] = np.mean(np.array(v), axis=0)
            clusters.clear()

    def predict(self, X):
        ans = []
        for x in X:
            distances = np.linalg.norm(x - self.mean_vec, axis=1)
            ans.append(np.argmax(distances))
        return np.array(ans)


class LVQ(object):
    """LVQ 聚类算法"""
    pass
