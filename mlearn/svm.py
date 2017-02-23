import numpy as np

class SVC(object):
    """
    支持向量机(Support Vector Machine)二分类器, 默认的核函数是
    多项式函数。
    使用时，正类用1表示，负类用-1表示。
    如果有必要，可自行编写新的核函数。
    """
    def __init__(self, C, e=0.3, kernel='poly'):
        '''
        :param C: 常数C，用来控制非线性部分的权重
        :param e: 误差，做为训练停止的条件
        :param kernel: 核函数，默认为多项式函数
        '''
        self.kernel = self.poly_kern
        self.e = e
        self.C = C
        self.bias = 0

    def poly_kern(self, x, z, p=1):
        '''
        :param x: 第一个向量
        :param z: 第二个向量
        :param p: 指数，默认为1，代表线性。
        :return: (np.dot(x, z) + 1) ** p
        '''
        return (np.dot(x, z) + 1) ** p

    def fit(self, train_x, train_y):
        '''
        :param train_x: 训练集X
        :param train_y: 训练集Y
        :return: None
        拟合SVM
        '''
        self.train_x = train_x
        self.train_y = train_y
        self.alpha = np.zeros(train_x.shape[0])
        while True:
            index1, index2 = self.choose_alpha()
            if index1 == -1:
                break
            if self.train(index1, index2):
                break

    def get_index_two(self, index1):
        '''
        :param index1: alpha1的下标
        :return: alpha2的下标
        SMO算法的步骤，给定alpha1的坐标，通过MAX|E1 - E2|来获取alpha2的下标
        '''
        errors = np.array([abs(self.error(i) - self.error(index1)) for i in range(self.train_x.shape[0])])
        return np.where(errors == errors.max())[0][0]

    def choose_alpha(self):
        '''
        :return: alpha1和alpha2的下标
        SMO算法的步骤，用来选取alpha1
        '''
        for i in range(self.alpha.shape[0]):
            if 0 < self.alpha[i] < self.C:
                if 1 - self.e <= self.train_y[i] * self.g(self.train_x[i]) <= 1 + self.e:
                    index2 = self.get_index_two(i)
                    return i, index2

        for i in range(self.alpha.shape[0]):
            if 0 < self.alpha[i] < self.C:
                continue
            if self.alpha[i] == 0:
                if self.train_y[i] * self.g(self.train_x[i]) < 1:
                    index2 = self.get_index_two(i)
                    return i, index2
            else:
                if self.train_y[i] * self.g(self.train_x[i]) > 1:
                    index2 = self.get_index_two(i)
                    return i, index2
        return -1, -1

    def train(self, i1, i2):
        '''
        :param i1: alpha1
        :param i2: alpha2
        :return: 停止训练返回True，继续训练返回False
        采用SMO训练alpha，同时更新bias的值
        '''
        x1 = self.train_x[i1]
        x2 = self.train_x[i2]
        y1 = self.train_y[i1]
        y2 = self.train_y[i2]
        old_alpha = self.alpha.copy()

        eta = self.kernel(x1, x1) \
              + self.kernel(x2, x2) \
              - 2.0 * self.kernel(x1, x2)

        alpha2 = self.alpha[i2] \
                 + self.train_y[i2] * (self.error(i1) - self.error(i2)) / eta
        if y1 == y2:
            l = max(0, self.alpha[i2] + self.alpha[i1] - self.C)
            h = min(self.C, self.alpha[i2] + self.alpha[i1])

        else:
            l = max(0, self.alpha[i2] - self.alpha[i1])
            h = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])
        if alpha2 > h:
            alpha2 = h
        elif alpha2 < l:
            alpha2 = l
        alpha_old_1 = self.alpha[i1]
        self.alpha[i1] += y1 * y2 * (self.alpha[i2] - alpha2)
        self.alpha[i2] = alpha2

        b1 = -1 * self.error(i1) \
             - y1 * self.kernel(x1, x1) * (self.alpha[i1] - alpha_old_1) \
             - y2 * self.kernel(x2, x1) * (self.alpha[2] - alpha2) \
             + self.bias


        b2 = -1 * self.error(i2) \
             - y1 * self.kernel(x1, x2) * (self.alpha[i1] - alpha_old_1) \
             - y2 * self.kernel(x2, x2) * (self.alpha[2] - alpha2) \
             + self.bias

        if 0 < self.alpha[i1] < self.C:
            self.bias = b1
        elif 0 < self.alpha[i2] < self.C:
            self.bias = b2
        else:
            self.bias = 0.5 * (b1 + b2)

        if np.linalg.norm(old_alpha - self.alpha) < self.e:
            return True
        else:
            return False


    def error(self, index):
        '''
        :param index: 要计算的训练集下标
        :return: E(X)
        求E(X) = g(x) - y，即预测值与真实值的偏差
        '''
        return self.g(self.train_x[index]) - self.train_y[index]

    def g(self, x):
        '''
        :param x: 样本
        :return: g(x)
        预测函数 g(x) = Σ(alpha[i] * y[i] * k(x, xi)) + bias
        '''
        ans = 0.0
        for i in range(self.train_x.shape[0]):
            ans += self.alpha[i] * self.train_y[i] * self.kernel(x, self.train_x[i])
        return ans + self.bias

    def predict_one(self, x):
        '''
        :param x: 测试样本.
        :return: 样本所属类别
        预测单个样本
        '''
        return 1 if self.g(x) >= 0 else -1

    def predict(self, test_x):
        '''
        :param test_x: 测试集
        :return: 测试集样本的类别
        '''
        return np.array([self.predict_one(x) for x in test_x])



