"""
隐马尔科夫模型相关算法，包含观测序列概率算法，学习算法
和预测算法。
``````````````````````````````````````````````
求一个观测序列出现的概率：
    predict_ob_probability(state_trans, ob, pi, obseq)
给定观测序列，预测状态转移矩阵，观测矩阵，初始状态概率向量：
    get_model_param(obseq, status_count, ob_count, error=0.2)
求最优观测路径：
    get_optimal_path(state_trans, ob, pi, obseq)
"""

import numpy as np


def get_optimal_path(state_trans, ob, pi, obseq):
    '''
    :param state_trans: 状态转移矩阵
    :param ob: 观测概率矩阵
    :param pi: 初始状态概率向量
    :param obseq: 观测序列
    :return: 最优观测路径
    求最优观测路径的维特比算法。
    '''
    N = pi.shape[0]
    T = obseq.shape[0]
    delte = np.array([pi[i] * ob[i][obseq[0]] for i in range(N)])
    psi = np.zeros((T, N), dtype=np.int8)
    for t in range(1, T):
        for i in range(N):
            temp = delte * state_trans[:, i]
            max_val = temp.max()
            delte[i] = max_val * ob[i][obseq[t]]
            psi[t][i] = np.argwhere(temp == max_val)[0][0]

    path = np.zeros(T, dtype=np.int8)
    path[T - 1] = np.argwhere(delte == delte.max())
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1][path[t + 1]]
    return path


def forward_probability(state_trans, ob, pi, obseq):
    '''
    :param state_trans: 状态转移矩阵
    :param ob: 观测概率矩阵
    :param pi: 初始状态概率向量
    :param obseq: 观测序列
    :return: 前向概率矩阵
    观测序列的前向算法。
    '''
    alpha = np.zeros((obseq.shape[0], pi.shape[0]))
    alpha[0] = pi * ob[:, obseq[0]]
    for t in range(0, obseq.shape[0] - 1):
        alpha[t + 1] = np.dot(alpha[t], state_trans) * ob[:, obseq[t + 1]]
    return alpha


def backward_probability(state_trans, ob, pi, obseq):
    '''
    :param state_trans: 状态转移矩阵
    :param ob: 观测概率矩阵
    :param pi: 初始状态概率向量
    :param obseq: 观测序列
    :return: 后向概率矩阵
    观测序列的后向算法。
    '''
    beta = np.zeros((obseq.shape[0], pi.shape[0]))
    beta[obseq.shape[0] - 1] = np.ones(pi.shape[0])
    for t in range(obseq.shape[0] - 2, -1, -1):
        beta[t] = np.sum(state_trans * ob[:, obseq[t + 1]] * beta[t + 1], axis=1)
    return beta


def predict_ob_probability(state_trans, ob, pi, obseq):
    '''
    :param state_trans: 状态转移矩阵
    :param ob: 观测概率矩阵
    :param pi: 初始状态概率向量
    :param obseq: 观测序列
    :return: 观测序列概率
    观测序列概率算法，求一个观测序列出现的概率，默认采用前向算法。
    '''

    pro_mat = forward_probability(state_trans, ob, pi, obseq)
    return np.sum(pro_mat[obseq.shape[0] - 1, :])


def gamma(state_trans, ob, pi, obseq):
    '''
    :param state_trans: 状态转移矩阵
    :param ob: 观测概率矩阵
    :param pi: 初始状态概率向量
    :param obseq: 观测序列
    :return: 在t时刻处于状态qi的概率矩阵。
    求t时刻处于状态q的概率 即 gamma[t][i] = P(it = qi | O, lambda)
    '''
    alpha = forward_probability(state_trans, ob, pi, obseq)
    beta = backward_probability(state_trans, ob, pi, obseq)
    g = np.zeros((obseq.shape[0], pi.shape[0]))
    for t in range(0, obseq.shape[0]):
        g[t] = alpha[t, :] * beta[t, :] / (np.dot(alpha[t, :], beta[t, :]))
    return g


def xi(state_trans, ob, pi, obseq):
    '''
    :param state_trans: 状态转移矩阵
    :param ob: 观测概率矩阵
    :param pi: 初始状态概率向量
    :param obseq: 观测序列
    :return: 在t时刻处于状态qi切在t+1处于状态qj的概率矩阵和。
    在t时刻处于状态qi切在t+1处于状态qj的概率 即 xi[t][i] = P(it = qi, it+1 = qj | O, lambda)
    '''
    alpha = forward_probability(state_trans, ob, pi, obseq)
    beta = backward_probability(state_trans, ob, pi, obseq)
    x = np.zeros((obseq.shape[0] - 1, pi.shape[0], pi.shape[0]))
    for t in range(0, obseq.shape[0] - 1):
        temp = np.sum(alpha[t] * state_trans * ob[:, obseq[t + 1]] * beta[t + 1])
        temp_mat = alpha[t] * state_trans * ob[:, obseq[t + 1]] * beta[t + 1]
        x[t] = temp_mat / temp
    return x


def get_model_param(obseq, status_count, ob_count, error=0.2):
    '''
    :param obseq: 观测序列
    :param status_count: 状态的种数
    :param ob_count: 观测的种数
    :return: state_trans, ob, pi
    Baum-Welch学习算法。
    给定观测序列，预测状态转移矩阵，观测矩阵，初始状态概率向量。
    '''
    state_trans = np.array([[1 / status_count] * status_count] * status_count)
    ob = np.array([[1 / ob_count] * ob_count] * status_count)
    pi = np.array([1 / status_count] * status_count)
    while True:
        a, b, p = state_trans.copy(), ob.copy(), pi.copy()
        g = gamma(state_trans, ob, pi, obseq)
        state_trans = np.sum(xi(state_trans, ob, pi, obseq), axis=0) / np.sum(g[:-1])
        for j in range(status_count):
            for k in range(ob_count):
                temp = 0.0
                for t in range(obseq.shape[0]):
                    if obseq[t] == k:
                        temp += g[t][j]
                ob[j, k] = temp / np.sum(g[:, j])

        if np.linalg.norm(a - state_trans) < error \
                and np.linalg.norm(b - ob) < error \
                and np.linalg.norm(p - pi) < error:
            break
    return state_trans, ob, pi
