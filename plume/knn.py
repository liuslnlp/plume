import numpy as np


class KNeighborClassifier(object):
    """
    K近邻分类器
    """
    def __init__(self, n_neighbors:int=5, metric:int=2):
        """initialization
        :param n_neighbors: 
        :param metric: 距离函数的度量，metric=1时为曼哈顿距离，metric=2
        时为欧氏距离
        """
        self.X = None
        self.y = None
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y):
        """
        :param X_: shape = [n_samples, n_features] 
        :param y: shape = [n_samples] 
        :return: self
        """
        self.X = X.astype(np.float32)
        self.y = y
        if self.n_neighbors > X.shape[0]:
            self.n_neighbors = X.shape[0]

    def predict_one(self, x):
        """
        :param x: shape = [n_features]
        :return: predict
        """
        distances = np.linalg.norm(self.X - x, self.metric, axis=1)
        neighbors = np.argpartition(-distances, self.n_neighbors)[:self.n_neighbors]
        neighbors_label = self.y[neighbors]
        labels, counts = np.unique(neighbors_label, return_counts=True)
        return labels[np.argmax(counts)]

    def predict(self, X):
        """
        :param X: shape = [n_samples, n_features] 
        :return: shape = [n_samples]
        """
        return np.array([self.predict_one(i) for i in X])



