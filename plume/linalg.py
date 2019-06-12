# 一些基本的线性代数算法

import numpy as np


def lu(A):
    # LU 分解，要求必须非奇异
    A = A.copy()
    U = np.zeros(A.shape)
    L = np.eye(A.shape[1])
    for i in range(A.shape[1] - 1):
        U[:i + 1, i] = A[:i + 1, i]
        L[i + 1:, i] = A[i + 1:, i] / U[i, i]
        A[i + 1:] -= L[i + 1:, i].reshape(-1, 1) * A[i, :]
    U[:, -1] = A[:, -1]
    return L, U


def ldu(A):
    # LDU 分解
    L, U = lu(A)
    D = U.diagonal()
    U = U / D.reshape(-1, 1)
    return L, D, U


def forward_substitution(A, b):
    # 前向代换
    x = np.zeros(b.shape)
    for i in range(b.shape[0]):
        x[i] = (b[i] - np.sum(x[0:i] * A[i, :i].reshape(-1, 1), axis=0))/A[i, i]
    return x


def backward_substitution(A, b):
    # 后向代换
    x = np.zeros(b.shape)
    for i in range(b.shape[0] - 1, -1, -1):
        x[i] = (b[i] - np.sum(x[i + 1:] * A[i, i + 1:].reshape(-1, 1), axis=0))/A[i, i]
    return x


def solve(A, b):
    # 解方程组（A非奇异）,b为多组向量
    L, U = lu(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x


def inv(A):
    # 矩阵求逆
    return solve(A, np.eye(A.shape[0]))


if __name__ == '__main__':
    c = np.array([[2., 2., 3.], [4., 7., 7.], [-2., 4., 5.]])
    print(np.linalg.inv(c))
    print(inv(c))
