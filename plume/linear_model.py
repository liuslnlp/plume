import numpy as np


class LinearRegression(object):
    """
    Linear Regression
    """
    def __init__(self):
        self.weight = None

    def fit(self, X, y):
        """
        :param X_: shape = [n_samples + 1, n_features] 
        :param y: shape = [n_samples] 
        :return: self
        """
        X_ = np.c_[X, np.ones(X.shape[0])]
        self.weight = np.linalg.inv(X_.T @ X_) @ X_.T @ y
        return self

    def predict(self, X):
        """
        :param X: shape = [n_samples, n_features] 
        :return: shape = [n_samples]
        """
        X_ = np.c_[X, np.ones(X.shape[0])]
        return X_ @ self.weight


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
        self.weight = None
        self.sign = np.vectorize(lambda x: 1 if x >= 0.5 else 0)

    def p_yeq1(self, X_):
        """Get P(y=1 | x)
        :param X_: shape = [n_samples + 1, n_features] 
        :return: shape = [n_samples]
        """
        tmp = np.exp(self.weight @ X_.T)
        return tmp / (1 + tmp)

    def diff(self, X_, y, p):
        """Get first derivative
        :param X_: shape = [n_samples, n_features + 1] 
        :param y: shape = [n_samples] 
        :param p: shape = [n_samples] P(y=1 | x)
        :return:  shape = [n_features + 1] first derivative
        """
        return -(y - p) @ X_

    def secdiff(self, X_, p):
        """Get second derivative
        :param p: shape = [n_samples] P(y=1 | x)
        :return: shape = [n_features + 1, n_features + 1] second derivative
        """
        hess = np.zeros((X_.shape[1], X_.shape[1]))
        for i in range(X_.shape[0]):
            hess += self.X_XT[i] * p[i] * (1 - p[i])
        return hess

    def newton_method(self, X_, y):
        """Newton Method to calculate weight
        :param X_: shape = [n_samples + 1, n_features] 
        :param y: shape = [n_samples] 
        :return: None
        """
        self.weight = np.ones(X_.shape[1])
        self.X_XT = []
        for i in range(X_.shape[0]):
            t = X_[i, :].reshape((-1, 1))
            self.X_XT.append(t @ t.T)

        for _ in range(self.max_epoch):
            y_eq_true = self.p_yeq1(X_)
            first_der = self.diff(X_, y, y_eq_true)

            hess = self.secdiff(X_, y_eq_true)
            new_weight = self.weight - (np.linalg.inv(hess) @ first_der.reshape((-1, 1))).flatten()

            if np.linalg.norm(new_weight - self.weight) <= self.error:
                break
            self.weight = new_weight

    def fit(self, X, y):
        """
        :param X_: shape = [n_samples, n_features] 
        :param y: shape = [n_samples] 
        :return: self
        """
        X_ = np.c_[np.ones(X.shape[0]), X]
        self.newton_method(X_, y)
        return self

    def predict(self, X) -> np.array:
        """
        :param X: shape = [n_samples, n_features] 
        :return: shape = [n_samples]
        """
        X_ = np.c_[np.ones(X.shape[0]), X]
        return self.sign(self.p_yeq1(X_))
