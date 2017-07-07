import numpy as np
from collections import defaultdict


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
        self.x_train = None
        self.y_train = None
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, x_train, y_train):
        self.x_train = x_train.astype(np.float32)
        self.y_train = y_train
        if self.n_neighbors > x_train.shape[0]:
            self.n_neighbors = x_train.shape[0]

    def predict_one(self, x):
        distances = np.linalg.norm(self.x_train - x, self.metric, axis=1)
        neighbors = np.argpartition(-distances, self.n_neighbors)[:self.n_neighbors]
        neighbors_label = self.y_train[neighbors]
        labels, counts = np.unique(neighbors_label, return_counts=True)
        return labels[np.argmax(counts)]

    def predict(self, x_test):
        return np.array([self.predict_one(i) for i in x_test])



