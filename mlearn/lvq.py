import numpy as np


class LVQ(object):
    """
    学习向量量化(Learning Vector Quantization, LVQ)聚类算法。
    """

    def __init__(self, ratio=0.2, rounds=1000):
        '''
        :param ratio: 步长，学习率
        :param rounds: 迭代轮次
        初始化
        '''
        self.ratio = ratio
        self.rounds = rounds

    def fit(self, train_x, train_y):
        '''
        :param train_x: 训练集X
        :param train_y: 训练集Y
        :return: None
        拟合
        '''
        labels = set(train_y)
        self.p_vec = np.ones((len(labels), train_x.shape[1]))
        self.map = []
        for label in labels:
            self.map.append(label)

        for i in range(self.rounds):
            index = np.random.choice(train_x.shape[0])
            x = train_x[index]
            y = train_y[index]
            distance = [np.linalg.norm(x - v) for v in self.p_vec]
            p_min_index = distance.index(min(distance))
            temp = self.ratio * (x - self.p_vec[p_min_index])
            if y == self.map[p_min_index]:
                self.p_vec[p_min_index] += temp
            else:
                self.p_vec[p_min_index] -= temp

    def predict_one(self, x):
        '''
        :param x: 测试集合的一个样本
        :return: test_x的类别
        预测单个样本
        '''
        distance = [np.linalg.norm(x - v) for v in self.p_vec]
        p_min_index = distance.index(min(distance))
        return self.map[p_min_index]

    def predict(self, test_x):
        '''
        :param test_x: 测试集
        :return: 测试集的预测值
        预测一个测试集
        '''
        return [self.predict_one(x) for x in test_x]
