import numpy as np
from sklearn.preprocessing import LabelBinarizer


class GaussianNB(object):
    """
    朴素贝叶斯分类器，适用于连续型数据。
    """

    @staticmethod
    def gaussfunc(x, mu, singma):
        """高斯函数
        :param x: 数据集
        :param mu: 均值 
        :param singma: 方差 
        :return: 
        """
        sqsingma = singma @ singma
        numerator = -np.exp(np.sum((x - mu) ** 2, axis=1) / (2 * sqsingma))
        return numerator / np.sqrt(2 * np.pi * sqsingma)

    def fit(self, X, y):
        """
        :param X_: shape = [n_samples, n_features] 
        :param y: shape = [n_samples] 
        :return: self
        """
        self.classes, self.classes_count = np.unique(y, return_counts=True)
        self.mean = np.zeros((self.classes_count.shape[0],
                              X.shape[1]), dtype=np.float64)
        self.var = np.zeros((self.classes_count.shape[0],
                             X.shape[1]), dtype=np.float64)
        for i, label in enumerate(self.classes):
            x_i = X[y == label]
            self.mean[i, :] = np.mean(x_i, axis=0)
            self.var[i, :] = np.var(x_i, axis=0)

        return self

    def predict(self, X):
        """
        :param X: shape = [n_samples, n_features] 
        :return: shape = [n_samples]
        """
        likelihood = []
        for i in range(self.classes.shape[0]):
            likelihood.append(self.classes_count[i] *
                              GaussianNB.gaussfunc(X, self.mean[i, :],
                                                   self.var[i, :]))
        likelihood = np.array(likelihood).T
        return np.argmax(likelihood, axis=1)


class MultinomialNB(object):
    """
    朴素贝叶斯分类器，适用于离散型数据。
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        """
        :param X_: shape = [n_samples, n_features] 
        :param y: shape = [n_samples] 
        :return: self
        """
        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes = labelbin.classes_
        self.class_count = np.zeros(Y.shape[1], dtype=np.float64)
        self.feature_count = np.zeros((Y.shape[1], X.shape[1]),
                                      dtype=np.float64)

        self.feature_count += Y.T @ X
        self.class_count += Y.sum(axis=0)
        smoothed_fc = self.feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.feature_log_prob = (np.log(smoothed_fc) -
                                 np.log(smoothed_cc.reshape(-1, 1)))

    def predict(self, X):
        """
        :param X: shape = [n_samples, n_features] 
        :return: shape = [n_samples]
        """
        likelihood = X @ self.feature_log_prob.T
        return np.argmax(likelihood, axis=1)
