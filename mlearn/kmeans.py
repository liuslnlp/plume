import numpy as np
from collections import defaultdict


class KMeans(object):
    """K均值(K-means)聚类算法"""

    def __init__(self, n_clusters, threshold=1.0):
        '''
        :param n_clusters: 要划分的类别的个数
        :param threshold: 停止的条件
        '''
        self.n_clusters = n_clusters
        self.clusters = defaultdict(list)
        self.threshold = threshold
        self.k_vec = None

    def choose_init_vec(self, train_x):
        '''
        :param train_x: 训练集X
        :return: 初始化后的K向量
        初始化K向量
        '''
        indexs = np.random.choice(train_x.shape[0],
                                  self.n_clusters, replace=False)
        return train_x.copy()[indexs]

    def fit(self, train_x):
        '''
        :param train_x: 训练集X
        :return: None
        拟合
        '''
        self.k_vec = self.choose_init_vec(train_x)
        while True:
            again = False
            self.clusters.clear()
            for i in train_x:
                distance = [np.linalg.norm(i - v) for v in self.k_vec]
                self.clusters[distance.index(min(distance))].append(i)
            for k, v in self.clusters.items():
                l = np.array(v)
                mean = l.mean(axis=0)
                if np.linalg.norm(self.k_vec[k] - mean) > self.threshold:
                    again = True
                    self.k_vec[k] = mean

            if not again:
                break

    def predict_one(self, x):
        '''
        :param x: 测试集的一个样本
        :return: x的预测值
        预测单个值
        '''
        distance = [np.linalg.norm(x - v) for v in self.k_vec]
        return distance.index(min(distance))

    def predict(self, test_x):
        '''
        :param test_x: 测试集
        :return: 测试集的预测值
        预测一个测试集
        '''
        return [self.predict_one(x) for x in test_x]
