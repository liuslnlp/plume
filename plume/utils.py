import numpy as np
import matplotlib.pyplot as plt


def check_X_y(X, y):
    if X.shape[0] != y.shape[0]:
        raise ValueError('Shape does not match')
    return X, y


def gen_reg_data():
    X = np.arange(0, 45, 0.1)
    X = X + np.random.random(size=X.shape[0]) * 20
    y = 2 * X + np.random.random(size=X.shape[0]) * 20 + 10
    return X.reshape((-1, 1)), y



def plot_decision_boundary(pred_func, X, y, title=None):
    """分类器画图函数，可画出样本点和决策边界
    :param pred_func: predict函数
    :param X: 训练集X
    :param y: 训练集Y
    :return: None
    """

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

    if title:
        plt.title(title)
    plt.show()
