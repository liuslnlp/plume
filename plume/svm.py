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

class ExpSVC(object):
    """SVM by SMO"""
    def __init__(self, C=1.0, kernel='rbf', **kwargs):
        if kernel == 'poly':
            self.kernel = partial(ExpSVC.polynomial, p=kwargs.get('p', 3))
        else:
            self.kernel = partial(ExpSVC.gaussian, var=kwargs.get('var', 1))
        self.C = C


    @staticmethod
    def polynomial(X, xj, p):
        return (X @ xj + 1) ** p

    @staticmethod
    def gaussian(X, xj, var):
        if(len(X.shape) == 1):
            return np.exp(-np.linalg.norm(X - xj) ** 2 / (2 * var * var))
        return np.exp(-np.linalg.norm(X - xj, axis=1) ** 2 / (2 * var * var))


    def yita(self, i, j):
        return self.kernel(self.X[i, :], self.X[i, :]) + \
               self.kernel(self.X[j, :], self.X[j, :]) - \
               2 * self.kernel(self.X[i, :], self.X[j, :])


    def g(self, i):
        return np.sum(self.kernel(self.X, self.X[i]) * self.y * self.alphas) + self.bias

    def E(self, i):
        return self.g(i) - self.y[i]

    def aet(self, x, y=1.0, eps=1e-5):
        """about equal to
        """
        return (y - eps) <= x <= (y + eps)

    def opt_i_j(self, i, j):
        # i_old = self.alphas[i]
        # j_old = self.alphas[j]
        j_unc = j + self.y[j] * (self.E(i) - self.E(j)) / (self.yita(i, j) + 0.000001)
        if self.y[i] != self.y[j]:
            L = max(0.0, self.alphas[j] - self.alphas[i])
            H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
        else:
            L = max(0.0, self.alphas[j] + self.alphas[i] - self.C)
            H = min(self.C, self.alphas[j] + self.alphas[i])
        if j_unc > H:
            j_new = H
        elif j_unc < L:
            j_new = L
        else:
            j_new = j_unc

        t = self.bias - self.y[i]
        for n in range(self.X.shape[0]):
            t += self.alphas[n] * self.y[n] * self.kernel(self.X[n], self.X[i])
        self.Es[i] = t

        self.alphas[i] = self.alphas[i] + self.y[i] * self.y[j] * (self.alphas[j] - j_new)
        self.alphas[j] = j_new

        t = self.y[i]
        for n in range(self.X.shape[0]):
            t -= self.alphas[n] * self.y[n] * self.kernel(self.X[n], self.X[i])
        self.bias = t

    def opt(self):
        step = 0
        old_alphas = self.alphas.copy()
        while True:
            step += 1
            if not step % 50:
                if np.linalg.norm(self.alphas - old_alphas) < 0.01:
                    break
                else:
                    old_alphas = self.alphas.copy()
            alpha1 = None
            flag = True
            tmp = []
            for i in range(self.X.shape[0]):
                if 0.0 < self.alphas[i] < self.C:
                    if not self.aet(self.y[i] * self.g(i)):
                        alpha1 = i
                        flag = False
                        break
                else:
                    tmp.append(i)

            if flag:
                for i in tmp:
                    if self.aet(self.alphas[i], 0) and self.y[i] * self.g(i) < 1.0:
                        alpha1 = i
                        break
                    elif self.aet(self.alphas[i], self.C) and self.y[i] * self.g(i) > 1.0:
                        alpha1 = i
                        break

            if alpha1 is None:
                break

            E1 = self.E(alpha1)
            if E1 < 0:
                alpha2 = np.argmax(self.Es)
            else:
                alpha2 = np.argmin(self.Es)
            self.opt_i_j(alpha1, alpha2)


    def fit(self, X, y):
        """
        :param X_: shape = [n_samples, n_features] 
        :param y: shape = [n_samples] 
        :return: self
        """
        self.X = X
        self.y = y
        self.alphas = np.zeros(X.shape[0])
        self.bias = 0
        self.Es = np.array([self.E(i) for i in range(self.X.shape[0])])
        self.opt()


    def _predict(self, x):
        t = np.sum(self.kernel(self.X, x) * self.y * self.alphas) + self.bias
        return 1 if t >= 0 else -1

    def predict(self, X):
        return np.array([self._predict(i) for i in X])
