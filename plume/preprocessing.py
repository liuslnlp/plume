import numpy as np


class StandardScaler(object):
    """中心化，标准化"""
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.var = np.var(X, axis=0)
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        return (X - self.mean) / np.sqrt(self.var)


class MinMaxScaler(object):
    """归一化"""
    def __init__(self, minval=0, maxval=1):
        self.minval = minval
        self.maxval = maxval

    def fit_transforn(self, X):
        x_min = np.min(X, axis=0)
        x_max = np.max(X, axis=0)
        X = (X - x_min) / (x_max - x_min)
        return (self.maxval - self.minval) * X + self.minval


