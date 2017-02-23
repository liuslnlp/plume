import numpy as np

class LinearRegression(object):
    """
    线性回归器。
    """
    def __init__(self):
        self.weight = None
        self.b = None

    def fit(self, train_x, train_y):
        '''
        :param train_x: 训练集X
        :param train_y: 训练集Y
        :return: None
        拟合权重系数
        '''
        ave_x = np.average(train_x, 0)
        a1 = np.array([0.0] * ave_x.shape[0])
        a2 = 0.0
        a3 = np.array([0.0] * ave_x.shape[0])
        for x, y in zip(train_x, train_y):
            a1 += y * (x - ave_x)
            a2 += np.dot(x.T, x)
            a3 += x

        self.weight = 1 / (a2 - np.dot(a3.T, a3) / train_x.shape[0]) * a1
        b = 0.0
        for x, y in zip(train_x, train_y):
            b += y - np.dot(self.weight.T,  x)
        self.b = 1 / train_x.shape[0] * b

    def predict_one(self, x):
        '''
        :param x: 测试集的一个样本
        :return: x的预测值
        预测单个值
        '''
        return np.dot(self.weight.T, x) + self.b

    def predict(self, test_x):
        '''
        :param test_x: 测试集
        :return: 测试集的预测值
        预测一个测试集
        '''
        return np.array([self.predict_one(x) for x in test_x])


class LogisticRegressionClassifier(object):
    """
    逻辑斯蒂回归分类器，采用牛顿法训练。
    """
    def __init__(self, w0=None):
        self.weight = w0

    def first_der(self, X, y):
        '''
        :param X: 训练集X
        :param y: 训练集Y
        :return: 一阶导数
        '''
        ans = 0.0
        for xi, yi in zip(X, y):
            xi = a = np.concatenate((np.ones(1), np.array(xi)))
            ans -= np.dot(xi, yi - self.p1(xi))
        return ans

    def second_der(self, X, y):
        '''
        :param X: 训练集X
        :param y: 训练集Y
        :return: 二阶导数
        '''
        ans = 0.0
        for xi, yi in zip(X, y):
            xi = a = np.concatenate((np.ones(1), np.array(xi)))
            ans += np.dot(xi, xi.T) * self.p1(xi) * (1 - self.p1(xi))
        return ans

    def p1(self, Xi):
        '''
        :param Xi: X的分量
        :return: P(y=1 | x)
        '''
        temp = np.exp(np.dot(self.weight.T, Xi))
        return temp / (1 + temp)

    def fit(self, train_x, train_y, epochs=1000):
        '''
        :param train_x: 训练集X
        :param train_y: 训练集Y
        :param epochs: 迭代次数
        :return: None
        拟合权重系数
        '''
        if self.weight is None:
            self.weight = np.array([1.0] * (train_x.shape[1] + 1))

        for i in range(epochs):
            deta = (1.0 / self.second_der(train_x, train_y)) \
                   * self.first_der(train_x, train_y)
            self.weight -= deta


    def predict_one(self, x):
        '''
        :param x: 测试集合的一个样本
        :return: test_x的类别
        预测单个样本
        '''
        x = np.concatenate((np.ones(1), np.array(x)))
        ans = np.dot(self.weight.T, x)
        return  1 if ans >= 0.5 else 0

    def predict(self, test_x):
        '''
        :param test_x: 测试集
        :return: 测试集的预测值
        预测一个测试集
        '''
        return np.array([self.predict_one(x) for x in test_x])

