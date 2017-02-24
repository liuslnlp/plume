import numpy as np


class Perceptron(object):
    """
    感知机(Perceptron)，二类分类的线性分类模型。
    使用时，正例用1表示，负例用-1表示
    """

    def __init__(self, w0, b0, eta):
        """
        :param w0: 权值向量。
        :param b0: 偏置。
        :param eta: 学习率。
        """
        self.weight = w0
        self.bias = b0
        self.eta = eta

    def fit(self, train_x, train_y):
        """
        :param train_x: 训练集X。 x∈Rn
        :param train_y: 测试集Y。 y={-1, +1}
        :return: None
        梯度下降法拟合，返回超平面
        """
        while True:
            cycle_again = False
            for x, y in zip(train_x, train_y):
                if y * (np.dot(x, self.weight) + self.bias) <= 0:
                    cycle_again = True
                    self.weight += self.eta * y * x
                    self.bias += self.eta * y

            if not cycle_again:
                break

    def predict(self, test_x):
        '''
        :param test_x: 测试集
        :return: 测试集的预测值
        预测一个测试集
        '''
        ans = np.dot(test_x, self.weight) + self.bias
        return np.array([1 if i >= 0 else -1 for i in ans])

    def get_model(self):
        '''
        :return: (感知机的权重，截距)
        返回感知机的参数
        '''
        return (self.weight, self.bias)
