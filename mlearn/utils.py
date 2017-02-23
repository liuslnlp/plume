"""
工具包，包含了一些实用的函数。
"""


import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(pred_func, X, y):
    '''
    :param pred_func: predicet函数
    :param X: 训练集X
    :param y: 训练集Y
    :return: None
    分类器画图函数，可画出样本点和决策边界
    '''

    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    plt.show()