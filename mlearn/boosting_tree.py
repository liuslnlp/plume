"""
暂时没有写完。
"""
import numpy as np
from .decisiontree import DecisionTreeClassifier

class BaseClassifier(object):
    def __init__(self, train_x, train_y, weights):
        self.train_x = train_x
        self.train_y = train_y

    def train(self):
        pass
    
    def predict(self):
        pass
    
    def predict_one(self):
        pass
    
    def loss_func(self, pre_label, label):
        return (pre_label - label) ** 2

class BoostingTree(object):
    
    def __init__(self, rounds):
        self.rounds = rounds
        self.alphas = np.zeros(rounds)

    def fit(self, train_x, train_y):
        '''
        :param train_x: 训练集合X
        :param train_y: 训练集合Y（target）
        :return: None
        拟合提升树。
        '''
        self.train_x = train_x
        self.train_y = train_y

        self.weights = np.array([1 / train_x.shape[0]] * train_x.shape[0])
        for i in range(self.rounds):
            clf = DecisionTreeClassifier()
            clf.fit(train_x, train_y)
            pre_y = clf.predict(train_x)
            error = self.get_error(pre_y)
            alpha = self.get_coefficient(error)
            self.alphas[i] = alpha
            self.update_weights(alpha, pre_y)

    

    def update_weights(self, alpha, pre_y):
        Zm = 0.0
        w = np.zeros(self.train_y.shape[0])
        new_weights = np.zeros(self.train_y.shape[0])
        for i in range(self.train_y.shape[0]):
            w[i] = self.weights[i] * np.exp(-1 * alpha * self.train_y[i] * pre_y[i])
            Zm += w[i]
        for i in range(self.train_y.shape[0]):
            new_weights[i] = self.weights[i] / Zm * w[i]
        self.weights = new_weights
    
    def get_error(self, pre_y):
        indexs = np.where(self.train_y != pre_y)[0]
        return np.sum(self.weights[indexs])

    def get_coefficient(self, error):
        return 1 / 2 * np.log((1 - error) / error)


    def predict_one(self, x):
        '''
        :param x:  待预测的样本X
        :return: X所属的类别
        预测单个值
        '''
        pass

    def predict(self, test_x):
        '''
        :param test_x: 测试集
        :return: 测试集样本的类别集合
        预测多个值
        '''
        return np.array([self.predict_one(x) for x in test_x])