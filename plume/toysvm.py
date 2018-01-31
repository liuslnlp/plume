"""
利用 SMO 算法实现的线性 SVM 和非线性 SVM.
"""

import numpy as np
from functools import partial

class LinearSVC(object):
    def __init__(self, C=0.6, tol=1e-3, max_iter=50):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.err_cache = None
        self.alphas = None
        self.intercept_ = 0.0
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.alphas = np.zeros(X.shape[0])
        self.err_cache = np.zeros((X.shape[0], 2))
        self.smo_solver()
        self.coef_ = np.sum(self.alphas * y * X.T, axis=1)
        return self
    
    def select_j_rand(self, i, m):
        j = i
        while j == i:
            j = random.randint(0, m - 1)
        return j

    def select_j(self, i, ei):
        self.err_cache[i, :] = [1, ei]
        check_idxs = np.nonzero(self.err_cache[:, 0])[0]

        if check_idxs.shape[0] <= 1:
            j = self.select_j_rand(i, self.X.shape[0])
            return j, self.error(j)

        errors = np.vectorize(lambda idx: self.error(idx))(check_idxs)
        arg = np.argmax(np.abs(errors - ei))
        return check_idxs[arg], errors[arg]

    def update_err_cache(self, idx):
        e = self.error(idx)
        self.err_cache[idx] = [1, e]

    def inner_loop(self, i):
        ei = self.error(i)

        if not (self.y[i] * ei < -self.tol and self.alphas[i] < self.C or \
        self.y[i] * ei > self.tol and self.alphas[i] > 0):
            return 0

        j, ej = self.select_j(i, ei)

        alphai_old = self.alphas[i]
        alphaj_old = self.alphas[j]
        if self.y[i] != self.y[j]:
            L = max(0, self.alphas[j] - self.alphas[i])
            H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
        else:
            L = max(0, self.alphas[j] + self.alphas[i] - self.C)
            H = min(self.C, self.alphas[j] + self.alphas[i])
        if L == H:
            return 0
        eta = self.X[i] - self.X[j]
        eta = -eta @ eta
        if eta >= 0:
            return 0
     
        self.alphas[j] -= self.y[j] * (ei - ej) / eta
        self.alphas[j] = np.clip(self.alphas[j], L, H)
        self.update_err_cache(j)

        if abs(self.alphas[j] - alphaj_old) < 1e-5:
            return 0
        
        self.alphas[i] += self.y[j] * self.y[i] * (alphaj_old - self.alphas[j])
        self.update_err_cache(i)

        b1 = self.intercept_ - ei - self.y[i] * (self.alphas[i] - alphai_old) * self.X[i] @ self.X[i] - \
        self.y[j] * (self.alphas[j] - alphaj_old) * self.X[i] @ self.X[j]
        b2 = self.intercept_ - ej - self.y[i] * (self.alphas[i] - alphai_old) * self.X[i] @ self.X[j] - \
        self.y[j] * (self.alphas[j] - alphaj_old) * self.X[j] @ self.X[j]

        if 0 < self.alphas[i] < self.C:
            self.intercept_ = b1
        elif 0 < self.alphas[j] < self.C:
            self.intercept_ = b2
        else:
            self.intercept_ = (b1 + b2) / 2
        return 1


    def smo_solver(self):
        alpha_changed_count = 0
        n_iter = 0
        entire_flag = True
        while n_iter < self.max_iter and (alpha_changed_count > 0 or entire_flag):
            alpha_changed_count = 0
            if entire_flag:
                alpha_changed_count += sum(self.inner_loop(i) for i in range(self.X.shape[0]))
            else:
                non_bound_idxs = np.nonzero((self.alphas > 0) * (self.alphas < self.C))[0]
                alpha_changed_count += sum(self.inner_loop(i) for i in non_bound_idxs)
            if entire_flag:
                entire_flag = False  
            elif alpha_changed_count == 0:
                entire_flag = True
            n_iter += 1

        print(f"Total iter: {n_iter}")

    def randj(self, i, m):
        j = i
        while j == i:
            j = random.randint(0, m - 1)
        return j
    
    def error(self, idx):
        self.coef_ = np.sum(self.alphas * self.y * self.X.T, axis=1)
        return self.g(self.X[idx]) - self.y[idx]

    def g(self, X):
        return X @ self.coef_ + self.intercept_

    def predict(self, X):
        return np.sign(np.array([self._predict(i) for i in X]))



def polynomial(X, xj, p):
    return (X @ xj + 1) ** p


def gaussian(X, xj, var):
    if len(X.shape) == 1:
        return np.exp(-np.linalg.norm(X - xj) ** 2 / (2 * var * var))
    return np.exp(-np.linalg.norm(X - xj, axis=1) ** 2 / (2 * var * var))

class SVC(object):
    def __init__(self, C=0.6, kernel='rbf', tol=1e-6, max_iter=50, **kwargs):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.err_cache = None
        self.alphas = None
        self.intercept_ = 0.0
        if kernel == 'rbf':
            self.kernel = partial(gaussian, var=kwargs.get('var', 0.2))
        elif kernel == 'poly':
            self.kernel = partial(poly, var=kwargs.get('p', 3))
        else:
            raise NameError(f'Kernel function {kernel} does not exit!')
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.alphas = np.zeros(X.shape[0])
        self.err_cache = np.zeros((X.shape[0], 2))
        self.k = np.zeros((self.X.shape[0], self.X.shape[0]))
        for i in range(self.X.shape[0]):
            self.k[i, :] = self.kernel(X, X[i, :])
        self.smo_solver()
        return self
    
    def select_j_rand(self, i, m):
        j = i
        while j == i:
            j = random.randint(0, m - 1)
        return j

    def select_j(self, i, ei):
        self.err_cache[i, :] = [1, ei]
        check_idxs = np.nonzero(self.err_cache[:, 0])[0]

        if check_idxs.shape[0] <= 1:
            j = self.select_j_rand(i, self.X.shape[0])
            return j, self.error(j)

        errors = np.vectorize(lambda idx: self.error(idx))(check_idxs)
        arg = np.argmax(np.abs(errors - ei))
        return check_idxs[arg], errors[arg]

    def update_err_cache(self, idx):
        e = self.error(idx)
        self.err_cache[idx] = [1, e]

    def yita(self, i, j):
        return self.k[i, i] + self.k[j, j] - 2 * self.k[i, j]

    def inner_loop(self, i):
        ei = self.error(i)

        if not (self.y[i] * ei < -self.tol and self.alphas[i] < self.C or \
        self.y[i] * ei > self.tol and self.alphas[i] > 0):
            return 0

        j, ej = self.select_j(i, ei)

        alphai_old = self.alphas[i]
        alphaj_old = self.alphas[j]
        if self.y[i] != self.y[j]:
            L = max(0, self.alphas[j] - self.alphas[i])
            H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
        else:
            L = max(0, self.alphas[j] + self.alphas[i] - self.C)
            H = min(self.C, self.alphas[j] + self.alphas[i])
        if L == H:
            return 0
        eta = -self.yita(i, j)
        if eta >= 0:
            return 0
     
        self.alphas[j] -= self.y[j] * (ei - ej) / eta
        self.alphas[j] = np.clip(self.alphas[j], L, H)
        self.update_err_cache(j)

        if abs(self.alphas[j] - alphaj_old) < 1e-6:
            return 0
        
        self.alphas[i] += self.y[j] * self.y[i] * (alphaj_old - self.alphas[j])
        self.update_err_cache(i)
        X = self.X
        b1 = self.intercept_ - ei - self.y[i] * (self.alphas[i] - alphai_old) * self.k[i, i] - \
        self.y[j] * (self.alphas[j] - alphaj_old) * self.k[i, j]
        b2 = self.intercept_ - ej - self.y[i] * (self.alphas[i] - alphai_old) * self.k[i, j] - \
        self.y[j] * (self.alphas[j] - alphaj_old) * self.k[j, j]

        if 0 < self.alphas[i] < self.C:
            self.intercept_ = b1
        elif 0 < self.alphas[j] < self.C:
            self.intercept_ = b2
        else:
            self.intercept_ = (b1 + b2) / 2
        return 1


    def smo_solver(self):
        alpha_changed_count = 0
        n_iter = 0
        entire_flag = True
        while n_iter < self.max_iter and (alpha_changed_count > 0 or entire_flag):
        # while n_iter < self.max_iter:
            
            alpha_changed_count = 0
            if entire_flag:
                alpha_changed_count += sum(self.inner_loop(i) for i in range(self.X.shape[0]))
            else:
                non_bound_idxs = np.nonzero((self.alphas > 0) * (self.alphas < self.C))[0]
                alpha_changed_count += sum(self.inner_loop(i) for i in non_bound_idxs)
   
            if entire_flag:
                entire_flag = False  
            elif alpha_changed_count == 0:
                entire_flag = True
            n_iter += 1


    def randj(self, i, m):
        j = i
        while j == i:
            j = random.randint(0, m - 1)
        return j
    
    def error(self, idx):
        return self.g(self.X[idx]) - self.y[idx]

    def g(self, x):
        return np.sum(self.kernel(self.X, x) * self.alphas * self.y) + self.intercept_

    def predict(self, X):
        return np.sign(np.array([self.g(i) + self.intercept_ for i in X]))
