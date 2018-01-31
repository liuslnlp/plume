"""
利用数值优化算法实现的 SVM，在 toysvm.py 中有对应的
SMO 算法实现。
"""

import numpy as np
import math
import scipy.optimize as opt
from functools import partial
from numba import jit


class LinearSVC(object):
    def __init__(self, C: float = 1.0):
        """
        :param C: float. Penalty parameters.
        """
        self.C = C
        self.sign = np.vectorize(lambda x: 1 if x >= 0 else -1)

    def cal_weight(self) -> np.array:
        """Get Weight
        :return: shape=[n_features]
        """
        return self.alpha * self.y @ self.X

    def cal_bias(self) -> float:
        for i, alpha in enumerate(self.alpha):
            if 0 < alpha < self.C:
                return self.y[i] - self.y * self.alpha @ (self.X[i] @ self.X.T)

    def minfunc(self, alpha: np.array) -> float:
        X_ = (alpha * self.y * self.X.T).T
        return 0.5 * (np.sum(X_ @ X_.T) - np.sum(alpha))

    def optimize(self):
        """Optimize by SciPy
        :return: Alphas. shape = [n_samples]
        """
        bound = ((0, self.C),) * self.X.shape[0]
        con = ({'type': 'eq', 'fun': lambda alpha: np.sum(alpha * self.y)},)
        result = opt.minimize(self.minfunc, np.zeros(self.X.shape[0]), bounds=bound, constraints=con)
        return result['x']

    def fit(self, X, y):
        """
        :param X_: shape = [n_samples, n_features] 
        :param y: shape = [n_samples] 
        :return: self
        """
        self.X = X
        self.y = y
        self.alpha = self.optimize()
        self.weight = self.cal_weight()
        self.bias = self.cal_bias()

    def predict(self, X) -> np.array:
        """
        :param X: Test vectors. shape=[n_samples, n_features] 
        :return: shape=[n_samples] 
        """
        return self.sign(self.weight @ X.T + self.bias)


class SVC(object):
    def __init__(self, kernel='poly', smo=False, C=3.0, **kwargs):
        if kernel == 'poly':
            self.kernel = partial(SVC.polynomial, p=kwargs.get('p', 3))
        else:
            self.kernel = partial(SVC.gaussian, var=kwargs.get('var', 1))
        self.C = C

    @staticmethod
    def polynomial(X, x, p):
        """多项式核函数
        :param x: 一个样本
        :param X: 样本集
        :param p: 指数
        :return: 
        """
        return (X @ x + 1) ** p

    @staticmethod
    def gaussian(X, x, var):
        """高斯核函数
        :param x: 一个样本
        :param X: 样本集
        :param var: 方差
        :return: 
        """
        return np.exp(-np.linalg.norm(X - x, axis=1) ** 2 / (2 * var * var))

    def get_wx(self, x, alpha):
        return np.sum(alpha * self.y * self.kernel(x, self.X))

    @jit
    def minfunc(self, alpha: np.array) -> float:
        """优化的目标函数
        :param alpha: 拉格朗日乘子
        :return: 
        """
        ans = 0.0
        for i in range(self.X.shape[0]):
            ans += alpha[i] * self.y[i] * self.get_wx(self.X[i], alpha)
        return 0.5 * ans - alpha.sum()

    def cal_bias(self) -> float:
        """求偏置
        :return: bias
        """
        for i, alpha in enumerate(self.alpha):
            if 0 < alpha < self.C:
                ans = self.y[i]
                ans -= self.get_wx(self.X[i], self.alpha)
                return ans

    def optimize(self):
        """Optimize by SciPy
        :return: Alphas. shape = [n_samples]
        """
        bound = ((0, self.C),) * self.X.shape[0]
        con = ({'type': 'eq', 'fun': lambda alpha: np.sum(alpha * self.y)},)
        result = opt.minimize(self.minfunc, np.zeros(self.X.shape[0]), bounds=bound, constraints=con)
        return result['x']

    def fit(self, X, y):
        """
        :param X_: shape = [n_samples, n_features] 
        :param y: shape = [n_samples] 
        :return: self
        """
        self.X = X
        self.y = y
        self.alpha = self.optimize()
        self.bias = self.cal_bias()

    def predict_one(self, x):
        return 1 if self.get_wx(x, self.alpha) + self.bias > 0 else -1

    def predict(self, X) -> np.array:
        return np.array([self.predict_one(i) for i in X])


