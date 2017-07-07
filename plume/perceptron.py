import numpy as np


class PerceptronClassifier(object):
    """
    Perceptron classifier, training by Stochastic Gradient Descent Algorithm.
    """

    def __init__(self, step: float = 0.01, dual: bool = True, max_epoch: int = 10):
        """initialization
        :param step: 步长
        :param dual: 是否采用对偶形式
        :param max_epoch: 随机梯度下降最大迭代轮次，设置此参数来防止线性不可分情况下的无限迭代。
        迭代次数最多不超过 max_epoch * N(N为训练集样本的数量)
        """
        if dual:
            self.fit = self.fit_dual
        else:
            self.fit = self.fit_undual
        self.step = step
        self.max_epoch = max_epoch
        self.weight = None
        self.bias = None
        self.sign = np.vectorize(lambda x: 1 if x >= 0 else -1)

    def fit_undual(self, x_train: np.array, y_train: np.array):
        """常规训练方式
        :param x_train: 训练集X
        :param y_train: 训练集Y，只能由1或-1组成
        """
        self.weight = np.zeros(x_train.shape[1])
        self.bias = 0

        for _ in range(self.max_epoch * x_train.shape[0]):
            i = np.random.randint(0, x_train.shape[0])
            if y_train[i] * (self.weight @ x_train[i] + self.bias) <= 0:
                temp = self.step * y_train[i]
                self.weight += temp * x_train[i]
                self.bias += temp

    def fit_dual(self, x_train: np.array, y_train: np.array):
        """对偶形式Y
        :param x_train: 训练集X
        :param y_train: 训练集Y，只能由1或-1组成
        """
        # Gram matrix
        gram = x_train @ x_train.T
        alpha = np.zeros(x_train.shape[0])
        self.bias = 0
        epoch = 0
        for _ in range(self.max_epoch * x_train.shape[0]):
            i = np.random.randint(0, x_train.shape[0])
            if y_train[i] * (np.sum(alpha * y_train * gram[i]) + self.bias) <= 0:
                alpha[i] += self.step
                self.bias += self.step * y_train[i]
        self.weight = (alpha * y_train) @ x_train

    def predict(self, x_test):
        """预测
        :param x_test: 测试集合 
        :return: 结果
        """
        return self.sign(self.weight @ x_test.T + self.bias)

    def get_model(self) -> (np.array, float):
        """获取模型的权重和偏置
        :return: weight, bias
        """
        return self.weight, self.bias
