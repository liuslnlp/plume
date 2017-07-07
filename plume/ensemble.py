from .tree import DecisionTreeClassifier
import numpy as np
from functools import partial


class AdaBoostClassifier(object):
    def __init__(self, clf, epoch: int = 5, **kwargs):
        """
        :param clf: 所采用的基分类器
        :param epoch: 最大迭代轮
        :param kwargs: 基分类器参数
        """
        self.baseclf = partial(clf, **kwargs)
        self.epoch = epoch
        # 注意停止条件
        self.alphas = np.zeros(epoch)
        self.sign = np.vectorize(lambda x: 1 if x >= 0 else -1)

        self.clfs = []

    def get_err(self, y_pred):
        """计算错误率
        :param y_pred: shape=[n_samples]
        :return: 
        """
        tmp = np.sum(y_pred != self.y)
        return tmp if tmp != 0 else 0.01

    def get_coef(self, error):
        """计算系数
        :param error: 错误率. float 
        :return: 
        """
        return 0.5 * np.log((1 - error) / error)

    def update_weight(self, alpha, y_pred):
        new_weight = self.weight * np.exp(-alpha * self.y * y_pred)
        self.weight = new_weight / new_weight.sum()

    def update_train_set(self):
        coef = (self.weight * 50).astype(np.int64)
        self.X = np.zeros(np.sum(coef))
        self.y = np.zeros(np.sum(coef))
        cursor = 0
        for i in range(self.init_X.shape[0]):
            step = coef[i]
            self.X[cursor: cursor + step] = np.array([self.init_X[i]] * step)
            self.y[cursor: cursor + step] = np.array([self.init_y[i]] * step)
            cursor += step

    def fit(self, X, y):
        """
        :param X: shape = [n_samples, n_features] 
        :param y: shape = [n_samples] 
        :return: self
        """
        self.weight = np.ones(X.shape[0]) / X.shape[0]
        self.init_X = X
        self.init_y = y
        self.X, self.y = X.copy(), y.copy()
        for i in range(self.epoch):

            clf = self.baseclf()
            clf.fit(self.X, self.y)
            y_pred = clf.predict(self.X)
            error = self.get_err(y_pred)
            if error > 0.5:
                break
            self.clfs.append(clf)
            alpha = self.get_coef(error)
            self.alphas[i] = alpha
            self.update_weight(alpha, y_pred)
        return self

    def predict(self, X) -> np.array:
        """
        :param X: shape = [n_samples, n_features] 
        :return: shape = [n_samples]
        """
        return self.sign(self.alphas[:len(self.clfs)] @
                         np.array([clf.predict(X) for clf in self.clfs]))


class BaggingClassifier(object):
    def __init__(self, basics):
        self.basics = basics

    def choose_set(self, X_):
        indexs = np.random.choice(X_.shape[0], X_.shape[0])
        return X_[indexs]

    @staticmethod
    def vote(labels):
        """
        :param labels: 给定列表
        :return: 列表中所占比重最大的类别
        投票选择，将给定类别簇中所占比重最大的类别选出
        """
        labels, labels_count = np.unique(labels, return_counts=True)
        return labels[np.argmax(labels_count)]

    def fit(self, X, y):
        X_ = np.c_[X, y]
        for i in self.basics:
            new_X_ = self.choose_set(X_)
            i.fit(new_X_[:, :-1], new_X_[:, -1])
        return self

    def predict_one(self, x):
        labels = np.array([i.predict_one(x) for i in self.basics])
        return self.vote(labels)

    def predict(self, X):
        tmp = np.array([clf.predict(X) for clf in self.basics]).T
        return np.array([self.vote(i) for i in tmp])


class _Tree(DecisionTreeClassifier):
    """随机森林的基分类器，继承自决策树
    """

    def __init__(self, max_depth=None):
        super().__init__(max_depth)

    @staticmethod
    def get_best_index(X_, features):
        """寻找最优切分特征
        :param X_: 候选集 shape=[ni_samples, n_features + 1]
        :param features: 特征的候选集合
        :return: 最优切分特征的编号和特征值
        """
        if features >= 3:
            indexs = np.random.choice(features.shape[0], int(np.log2(features.shape[0])))
            features = features[indexs]
        ginis = [
            DecisionTreeClassifier.get_fea_best_val(
                np.c_[X_[:, i], X_[:, -1]]) for i in features]
        ginis = np.array(ginis)
        i = np.argmax(ginis[:, 1])
        return features[i], ginis[i, 0]


class RandomForestsClassifier(object):
    def __init__(self, tree_num=5, max_depth=6):
        trees = [_Tree(max_depth) for _ in range(tree_num)]
        self.clf = BaggingClassifier(trees)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)
