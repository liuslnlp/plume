from .preprocessing import StandardScaler
import numpy as np
from abc import ABCMeta, abstractmethod


class LinearModel(metaclass=ABCMeta):
    """
    Abstract base class of Linear Model.
    """
    def __init__(self):
        self.scaler = StandardScaler()

    @abstractmethod
    def fit(self, X, y):
        """fit func"""

    def predict(self, X):
        if not hasattr(self, 'coef_'):
            raise Exception('Please run `fit` before predict')
        X = self.scaler.transform(X)
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.coef_

class LinearRegression(LinearModel):
    """
    Linear Regression.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        """
        :param X_: shape = (n_samples + 1, n_features)
        :param y: shape = (n_samples])
        :return: self
        """
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        X = np.c_[np.ones(X.shape[0]), X]
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        return self


class Lasso(LinearModel):
    """
    Lasso Regression, training by Coordinate Descent.
    """
    def __init__(self, alpha=1.0, n_iter=1000, e=0.1):
        self.alpha = alpha
        self.n_iter = n_iter
        self.e = e
        super().__init__()

    def fit(self, X, y):
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        X = np.c_[np.ones(X.shape[0]), X]
        self.coef_ = np.zeros(X.shape[1])
        for _ in range(self.n_iter):
            z = np.sum(X * X, axis=0)
            tmp = np.zeros(X.shape[1])
            for k in range(X.shape[1]):
                wk = self.coef_[k]
                self.coef_[k] = 0
                p_k = X[:, k] @ (y - X @ self.coef_)
                if p_k < -self.alpha / 2:
                    w_k = (p_k + self.alpha / 2) / z[k]
                elif p_k > self.alpha / 2:
                    w_k = (p_k - self.alpha / 2) / z[k]
                else:
                    w_k = 0
                tmp[k] = w_k
                self.coef_[k] = wk
            if np.linalg.norm(self.coef_ - tmp) < self.e:
                break
            self.coef_ = tmp
        return self

class Ridge(LinearModel):
    """
    Ridge Regression.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        super().__init__()

    def fit(self, X, y):
        """
        :param X_: shape = (n_samples + 1, n_features)
        :param y: shape = (n_samples])
        :return: self
        """
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        X_ = np.c_[np.ones(X.shape[0]), X]
        self.coef_ = np.linalg.inv(X_.T @ X_ + self.alpha * np.eye(X_.shape[1])) @ X_.T @ y
        return self


class LogisticRegression(object):
    """
    Logistic Regression Classifier training by Newton Method
    """

    def __init__(self, error: float = 0.7, max_epoch: int = 100):
        """
        :param error: float, if the distance between new weight and 
                      old weight is less than error, the process 
                      of traing will break.
        :param max_epoch: if training epoch >= max_epoch the process 
                          of traing will break.
        """
        self.error = error
        self.max_epoch = max_epoch
        self.coef_ = None
        self.sign = np.vectorize(lambda x: 1 if x >= 0.5 else 0)

    def p_func(self, X_):
        """Get P(y=1 | x)
        :param X_: shape = (n_samples + 1, n_features)
        :return: shape = (n_samples)
        """
        tmp = np.exp(self.coef_ @ X_.T)
        return tmp / (1 + tmp)

    def diff(self, X_, y, p):
        """Get derivative
        :param X_: shape = (n_samples, n_features + 1) 
        :param y: shape = (n_samples)
        :param p: shape = (n_samples) P(y=1 | x)
        :return:  shape = (n_features + 1) first derivative
        """
        return -(y - p) @ X_

    def hess_mat(self, X_, p):
        """Get Hessian Matrix
        :param p: shape = (n_samples) P(y=1 | x)
        :return: shape = (n_features + 1, n_features + 1) second derivative
        """
        hess = np.zeros((X_.shape[1], X_.shape[1]))
        for i in range(X_.shape[0]):
            hess += self.X_XT[i] * p[i] * (1 - p[i])
        return hess

    def newton_method(self, X_, y):
        """Newton Method to calculate weight
        :param X_: shape = (n_samples + 1, n_features)
        :param y: shape = (n_samples)
        :return: None
        """
        self.coef_ = np.ones(X_.shape[1])
        self.X_XT = []
        for i in range(X_.shape[0]):
            t = X_[i, :].reshape((-1, 1))
            self.X_XT.append(t @ t.T)

        for _ in range(self.max_epoch):
            p = self.p_func(X_)
            diff = self.diff(X_, y, p)
            hess = self.hess_mat(X_, p)
            new_weight = self.coef_ - (np.linalg.inv(hess) @ diff.reshape((-1, 1))).flatten()

            if np.linalg.norm(new_weight - self.coef_) <= self.error:
                break
            self.coef_ = new_weight

    def fit(self, X, y):
        """
        :param X_: shape = (n_samples, n_features)
        :param y: shape = (n_samples)
        :return: self
        """
        X_ = np.c_[np.ones(X.shape[0]), X]
        self.newton_method(X_, y)
        return self

    def predict(self, X) -> np.array:
        """
        :param X: shape = (n_samples, n_features] 
        :return: shape = (n_samples]
        """
        X_ = np.c_[np.ones(X.shape[0]), X]
        return self.sign(self.p_func(X_))


# 2018/12/30 补充：
# 基于梯度下降算法的逻辑回归
class LogisticRegressionGD(object):
    def __init__(self, learning_rate=0.05, error=0.05, max_iter=500):
        self.alpha = learning_rate
        self.error = error
        self.sign = np.vectorize(lambda x: 1 if x >= 0.5 else 0)
        self.max_iter = max_iter
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        X = np.c_[np.ones(X.shape[0]), X]
        self.coef_ = np.ones(X.shape[1])
        for i in range(self.max_iter):
            grad = (1 / X.shape[0]) * (y - self.sigmoid(X)) @ X
            if np.linalg.norm(grad) <= self.error:
                break
            else:
                self.coef_ += self.alpha * grad

    def sigmoid(self, X):
        exp = np.exp(X @ self.coef_)
        return exp / (1 + exp)

    def predict(self, X):
        X = self.scaler.transform(X)
        X = np.c_[np.ones(X.shape[0]), X]
        return self.sign(self.sigmoid(X))