import numpy as np
from collections import defaultdict


class Bagging(object):
    """
    Bagging算法.
    Bagging是一种集成学习算法，学习的本身依赖于基学习算法。
    本算法使用时需先实例化N个基学习器：
        clfs = [KNeighborClassifier(2) for i in range(N)]
    并且基学习算法必须重载了
        fit(train_x, train_y)
        predict_one(x)
        predict(test_x)
    三个方法。

    """

    def __init__(self, basics):
        '''
        :param basics: 实例化后的基学习器
        '''
        self.basics = basics

    def choose_set(self, train):
        '''
        :param train: 总数据集
        :return: 新数据集
        从给定数据集中有放回的抽取一个容量为size(train_x)的新数据集
        '''
        indexs = np.random.choice(train.shape[0], train.shape[0])
        return train.copy()[indexs]

    def vote(self, labels):
        '''
        :param labels: 给定列表
        :return: 列表中所占比重最大的类别
        投票选择，将给定类别簇中所占比重最大的类别选出
        '''
        p = defaultdict(int)
        for label in labels:
            p[label] += 1
        return max(p, key=p.get)

    def fit(self, train_x, train_y):
        '''
        :param train_x: 训练集合X
        :param train_y: 训练集合Y（target）
        :return: None
        '''
        train = np.c_[train_x, train_y]
        for i in self.basics:
            new_train = self.choose_set(train)
            i.fit(new_train[:, :-1], new_train[:, -1])

    def predict_one(self, x):
        '''
        :param x:  待预测的样本X
        :return: X所属的类别
        预测单个值
        '''
        labels = [i.predict_one(x) for i in self.basics]
        return self.vote(labels)

    def predict(self, test_x):
        '''
        :param test_x: 测试集
        :return: 测试集样本的类别集合
        预测多个值
        '''
        return np.array([self.predict_one(x) for x in test_x])
