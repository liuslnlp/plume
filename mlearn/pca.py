'''
主成份分析
'''
import numpy as np


def pca(train_x, d):
    '''
    :param train_x: 训练集合
    :param d: 新的维度
    :return: 降维后的训练集
    主成份分析法(PCA)，一种常用的降维算法。
    '''
    # 均值化
    mean_val = np.mean(train_x, axis=0)
    zero_mean_val = train_x - mean_val
    # 求协方差矩阵
    cov_mat = np.cov(zero_mean_val, rowvar=0)
    # 求特征值和特征向量
    vals, vecs = np.linalg.eig(np.mat(cov_mat))

    args = np.argsort(vals)[:-(d + 1):-1]
    eig_vecs = vecs[:, args]
    return zero_mean_val * eig_vecs
