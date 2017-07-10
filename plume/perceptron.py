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

    def fit_undual(self, X: np.array, y: np.array):
        """常规训练方式
        :param X: 训练集X
        :param y: 训练集Y，只能由1或-1组成
        """
        self.weight = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.max_epoch * X.shape[0]):
            i = np.random.randint(0, X.shape[0])
            if y[i] * (self.weight @ X[i] + self.bias) <= 0:
                temp = self.step * y[i]
                self.weight += temp * X[i]
                self.bias += temp

    def fit_dual(self, X: np.array, y: np.array):
        """对偶形式Y
        :param X: 训练集X
        :param y: 训练集Y，只能由1或-1组成
        """
        # Gram matrix
        gram = X @ X.T
        alpha = np.zeros(X.shape[0])
        self.bias = 0
        for _ in range(self.max_epoch * X.shape[0]):
            i = np.random.randint(0, X.shape[0])
            if y[i] * (np.sum(alpha * y * gram[i]) + self.bias) <= 0:
                alpha[i] += self.step
                self.bias += self.step * y[i]
        self.weight = (alpha * y) @ X

    def predict(self, X):
        """预测
        :param X: 测试集合 
        :return: 结果
        """
        return self.sign(self.weight @ X.T + self.bias)

    def get_model(self) -> (np.array, float):
        """获取模型的权重和偏置
        :return: weight, bias
        """
        return self.weight, self.bias
