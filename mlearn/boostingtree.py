"""
暂时没有写完。
"""
import numpy as np
from .decisiontree import BoostingTreeClassifier


class BoostingTree(object):
    
    def __init__(self, rounds):
        self.rounds = rounds
        self.alphas = np.zeros(rounds)
        self.clfs = []

    def fit(self, train_x, train_y):
        '''
        :param train_x: 训练集合X
        :param train_y: 训练集合Y（target）
        :return: None
        拟合提升树。
        '''
        self.init_train_x = train_x.copy()
        self.init_train_y = train_y.copy()
        self.train_x = train_x.copy()
        self.train_y = train_y.copy()

        self.weights = np.array([1 / train_x.shape[0]] * train_x.shape[0])

        for i in range(self.rounds):
            clf = BoostingTreeClassifier(max_depth=2)
            clf.fit(train_x, train_y)
            self.clfs.append(clf)
            pre_y = clf.predict(train_x).astype(np.int32)
            error = self.get_error(pre_y)
            if error > 0.5:
                break
            alpha = self.get_coefficient(error)
            self.alphas[i] = alpha
            self.update_weights(alpha, pre_y)
            self.update_train_set()


    def update_weights(self, alpha, pre_y):
        Zm = 0.0
        w = np.zeros(self.train_y.shape[0])
        new_weights = np.zeros(self.train_y.shape[0])
        for i in range(self.train_y.shape[0]):
            
            w[i] = self.weights[i] * np.exp(-1 * alpha * self.train_y[i] * pre_y[i])
            Zm += w[i]
        for i in range(self.train_y.shape[0]):
            new_weights[i] = self.weights[i] / Zm * w[i]
        self.weights = new_weights
    
    def update_train_set(self):
        new_weights = (self.weights * 10).astype(np.int8)
        self.train_x = np.zeros(np.sum(new_weights))
        self.train_y = np.zeros(np.sum(new_weights))

        for i in range(self.train_x.shape[0]):
            for _ in new_weights:
                self.train_x.append(self.init_train_x[i])
                self.train_y.append(self.init_train_y[i])

    def get_error(self, pre_y):
        indexs = np.where(self.train_y != pre_y)[0]
        if not indexs:
            return 0.01
        return np.sum(self.weights[indexs])

    def get_coefficient(self, error):
        return 1 / 2 * np.log((1 - error) / error)

    def sign(self, x):
        return 1 if x >=0 else -1

    def predict_one(self, x):
        '''
        :param x:  待预测的样本X
        :return: X所属的类别
        预测单个值
        '''
        ans = 0.0
        for clf, alpha in zip(self.clfs, self.alphas):
            ans += float(clf.predict_one(x)) * alpha
        
        return self.sign(ans)

    def predict(self, test_x):
        '''
        :param test_x: 测试集
        :return: 测试集样本的类别集合
        预测多个值
        '''
        return np.array([self.predict_one(x) for x in test_x])