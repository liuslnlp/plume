import numpy as np


class LinearRegression(object):
    """
    Linear Regression
    """

    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        """
        :param X_: shape = (n_samples + 1, n_features)
        :param y: shape = (n_samples])
        :return: self
        """
        X_ = np.c_[np.ones(X.shape[0]), X]
        self.coef_ = np.linalg.inv(X_.T @ X_) @ X_.T @ y
        return self

    def predict(self, X):
        """
        :param X: shape = (n_samples, n_features] 
        :return: shape = (n_samples]
        """
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.coef_


class Lasso(object):
    """
    Lasso Regression
    """

    def __init__(self, alpha=1.0, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = None

    def _soft_thresholding_operator(self, x, lamda):
        if x > 0 and lamda < abs(x):
            return x - lamda
        elif x < 0 and lamda < abs(x):
            return x + lamda
        else:
            return 0

    def fit(self, X, y):
        X = np.column_stack((np.ones(len(X)), X))
        standopt = np.sum(X ** 2, axis=0)
        beta = np.zeros(X.shape[1])
        lamda = self.alpha * X.shape[0]

        beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:])) / (X.shape[0])
        for iteration in range(self.max_iter):
            tmp_beta = beta.copy()
            for j in range(1, len(beta)):
                tmp_beta[j] = 0.0
                arg1 = X[:, j] @ (y - X @ tmp_beta)
                beta[j] = self._soft_thresholding_operator(arg1, lamda) / standopt[j]
            beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:])) / (X.shape[0])
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        return self

    def predict(self, X):
        y = np.dot(X, self.coef_) + self.intercept_
        return y


class Ridge(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        """
        :param X_: shape = (n_samples + 1, n_features)
        :param y: shape = (n_samples])
        :return: self
        """
        X_ = np.c_[np.ones(X.shape[0]), X]
        self.weight = np.linalg.inv(X_.T @ X_ + self.alpha * np.eye(X_.shape[1])) @ X_.T @ y
        return self

    def predict(self, X):
        """
        :param X: shape = (n_samples, n_features] 
        :return: shape = (n_samples]
        """
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.weight


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

    def p_func(self, X_):
        """Get P(y=1 | x)
        :param X_: shape = (n_samples + 1, n_features)
        :return: shape = (n_samples)
        """
        tmp = np.exp(self.weight @ X_.T)
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
        self.weight = np.ones(X_.shape[1])
        self.X_XT = []
        for i in range(X_.shape[0]):
            t = X_[i, :].reshape((-1, 1))
            self.X_XT.append(t @ t.T)

        for _ in range(self.max_epoch):
            p = self.p_func(X_)
            diff = self.diff(X_, y, p)
            hess = self.hess_mat(X_, p)
            new_weight = self.weight - (np.linalg.inv(hess) @ diff.reshape((-1, 1))).flatten()

            if np.linalg.norm(new_weight - self.weight) <= self.error:
                break
            self.weight = new_weight

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
