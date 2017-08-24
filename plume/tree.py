import numpy as np
from collections import namedtuple
from abc import ABCMeta, abstractmethod, abstractstaticmethod

# 决策树结点
# Parameters
# ----------
# feature : 特征，需要进行比对的特征名
# val : 特征值，当特征为离散值时，如果对应的特征值等于val，将其放入左子树，否则放入右子树
# left : 左子树
# right : 右子树
# label : 所属的类
TreeNode = namedtuple("TreeNode", 'feature val left right label')


class DecisionTree(object, metaclass=ABCMeta):
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def build(self, X_, features, depth=None):

        if self.unique_cond(X_):
            return TreeNode(None, None, None, None, self.get_target(X_))
        if features.shape[0] == 0 or depth and depth >= self.max_depth:
            return TreeNode(None, None, None, None, self.stop_early_target(X_))
        feature, val = self.get_best_index(X_, features)
        new_features = features[features != feature]
        del features
        left, right = self.devide(X_, feature, val)
        if left.any():
            left_branch = self.build(left, new_features, depth + 1 if depth else None)
        else:
            left_branch = TreeNode(None, None, None, None, val)
        if right.any():
            right_branch = self.build(right, new_features, depth + 1 if depth else None)
        else:
            right_branch = TreeNode(None, None, None, None, val)
        return TreeNode(feature, val, left_branch, right_branch, None)

    def fit(self, X, y):
        features = np.arange(X.shape[1])
        X_ = np.c_[X, y]
        self.root = self.build(X_, features)
        return self

    def predict_one(self, x):
        p = self.root
        while p.label is None:
            # print('feature', x[p.feature])
            # print(p.val)
            p = p.left if self.judge(x[p.feature], p.val) else p.right
        return p.label

    def predict(self, X):
        """
        :param X: shape = [n_samples, n_features]
        :return: shape = [n_samples]
        """
        return np.array([self.predict_one(x) for x in X])


    @abstractstaticmethod
    def devide(X_, feature, val):
        pass

    @abstractmethod
    def unique_cond(self, X_):
        pass

    @abstractmethod
    def get_target(self, X_):
        pass

    @abstractmethod
    def stop_early_target(self, X_):
        pass

    @abstractmethod
    def judge(self, node_val, val):
        pass

    @abstractmethod
    def get_best_index(self, X_, features):
        pass


class DecisionTreeClassifier(DecisionTree):
    def get_best_index(self, X_, features):
        ginis = [DecisionTreeClassifier.get_fea_best_val(
            np.c_[X_[:, i], X_[:, -1]]) for i in features]
        ginis = np.array(ginis)
        i = np.argmax(ginis[:, 1])
        return features[i], ginis[i, 0]

    def unique_cond(self, X_):
        return True if np.unique(X_[:, -1]).shape[0] == 1 else False

    def judge(self, node_val, val):
        return True if node_val == val else False

    def stop_early_target(self, X_):
        classes, classes_count = np.unique(X_[:, -1], return_counts=True)
        return classes[np.argmax(classes_count)]

    def get_target(self, X_):
        return X_[0, -1]

    @staticmethod
    def devide(X_, feature, val):
        return X_[X_[:, feature] == val], X_[X_[:, feature] != val]

    @staticmethod
    def gini(D):
        """求基尼指数 Gini(D)
        :param D: shape = [ni_samples]
        :return: Gini(D)
        """
        # 目前版本的 numpy.unique 不支持 axis 参数
        _, cls_counts = np.unique(D, return_counts=True)
        probability = cls_counts / cls_counts.sum()
        return 1 - (probability ** 2).sum()

    @staticmethod
    def congini(D_, val):
        """求基尼指数 Gini(D, A)
        :param D_: 被计算的列. shape=[ni_samples, 2]
        :param val: 被计算的列对应的切分变量
        :return: Gini(D, A)
        """
        left, right = D_[D_[:, 0] == val], D_[D_[:, 0] != val]
        return DecisionTreeClassifier.gini(left[:, -1]) * left.shape[0] / D_.shape[0] + \
               DecisionTreeClassifier.gini(right[:, -1]) * right.shape[0] / D_.shape[0]

    @staticmethod
    def get_fea_best_val(D_):
        """寻找当前特征对应的最优切分变量
        :param D_: 被计算的列. shape=[ni_samples, 2]
        :return: 最优切分变量的值和基尼指数的最大值
        """
        vals = np.unique(D_[:, :-1])
        tmp = np.array([DecisionTreeClassifier.congini(D_, val) for val in vals])
        return vals[np.argmax(tmp)], tmp.max()


class DecisionTreeRegressor(DecisionTree):
    def get_best_index(self, X_, features):
        losses = np.array([DecisionTreeRegressor.feature_min_loss(X_, feature) for feature in features])
        i = np.argmin(losses[:, 1])
        return features[i], losses[i, 0]

    def unique_cond(self, X_):
        return True if X_[:, -1].std() <= 0.1 else False

    def judge(self, node_val, val):
        return True if node_val <= val else False

    def stop_early_target(self, X_):
        self.get_target(X_)

    def get_target(self, X_):
        return X_[:, -1].mean()

    @staticmethod
    def devide(X_, feature, val):
        return X_[X_[:, feature] <= val], X_[X_[:, feature] >= val]

    @staticmethod
    def feature_loss(X_, feature, val):
        left, right = DecisionTreeRegressor.devide(X_, feature, val)
        left_loss = np.sum((left[:, -1] - left[:, -1].mean()) ** 2)
        right_loss = np.sum((right[:, -1] - right[:, -1].mean()) ** 2)
        return left_loss + right_loss

    @staticmethod
    def feature_min_loss(X_, feature):
        losses = np.array(list(map(lambda val: DecisionTreeRegressor.feature_loss(X_, feature, val), X_[:, feature])))
        i = np.argmin(losses)
        return X_[i, feature], losses[i]


# class DecisionTreeClassifier(object):
#     """
#     决策树分类器，采用 CART 算法训练
#     """
#
#     def __init__(self, max_depth=None):
#         self.max_depth = max_depth
#
#     @staticmethod
#     def devide_X(X_, feature, val):
#         """切分集合
#         :param X_: 被切分的集合, shape=[ni_samples, ni_features + 1]
#         :param feature: 切分变量
#         :param val: 切分变量的值
#         :return: 左集合，有集合
#         """
#         return X_[X_[:, feature] == val], X_[X_[:, feature] != val]
#
#     @staticmethod
#     def gini(D):
#         """求基尼指数 Gini(D)
#         :param D: shape = [ni_samples]
#         :return: Gini(D)
#         """
#         # 目前版本的 numpy.unique 不支持 axis 参数
#         _, cls_counts = np.unique(D, return_counts=True)
#         probability = cls_counts / cls_counts.sum()
#         return 1 - (probability ** 2).sum()
#
#     @staticmethod
#     def congini(D_, val):
#         """求基尼指数 Gini(D, A)
#         :param D_: 被计算的列. shape=[ni_samples, 2]
#         :param val: 被计算的列对应的切分变量
#         :return: Gini(D, A)
#         """
#         left, right = D_[D_[:, 0] == val], D_[D_[:, 0] != val]
#         return DecisionTreeClassifier.gini(left[:, -1]) * left.shape[0] / D_.shape[0] + \
#                DecisionTreeClassifier.gini(right[:, -1]) * right.shape[0] / D_.shape[0]
#
#     @staticmethod
#     def get_fea_best_val(D_):
#         """寻找当前特征对应的最优切分变量
#         :param D_: 被计算的列. shape=[ni_samples, 2]
#         :return: 最优切分变量的值和基尼指数的最大值
#         """
#         vals = np.unique(D_[:, :-1])
#         tmp = np.array([DecisionTreeClassifier.congini(D_, val) for val in vals])
#         return vals[np.argmax(tmp)], tmp.max()
#
#     @staticmethod
#     def get_best_index(X_, features):
#         """寻找最优切分特征
#         :param X_: 候选集 shape=[ni_samples, n_features + 1]
#         :param features: 特征的候选集合
#         :return: 最优切分特征的编号和特征值
#         """
#         ginis = [
#             DecisionTreeClassifier.get_fea_best_val(
#                 np.c_[X_[:, i], X_[:, -1]]) for i in features]
#         ginis = np.array(ginis)
#         i = np.argmax(ginis[:, 1])
#         return features[i], ginis[i, 0]
#
#     def build(self, X_, features, depth=None):
#         """建树
#         :param X_: 候选集 shape=[ni_samples, n_features + 1]
#         :param features: 候选特征集
#         :param depth: 当前深度
#         :return: 结点
#         """
#         if np.unique(X_[:, -1]).shape[0] == 1:
#             return TreeNode(None, None, None, None, X_[0, -1])
#         if features.shape[0] == 0 or depth and depth >= self.max_depth:
#             classes, classes_count = np.unique(X_[:, -1], return_counts=True)
#             return TreeNode(None, None, None, None, classes[np.argmax(classes_count)])
#         feature_index, val = DecisionTreeClassifier.get_best_index(X_, features)
#         new_features = features[features != feature_index]
#         del features
#         left, right = DecisionTreeClassifier.devide_X(X_, feature_index, val)
#         left_branch = self.build(left, new_features, depth + 1 if depth else None)
#         right_branch = self.build(right, new_features, depth + 1 if depth else None)
#         return TreeNode(feature_index, val, left_branch, right_branch, None)
#
#     def fit(self, X, y):
#         """
#         :param X_: shape = [n_samples, n_features]
#         :param y: shape = [n_samples]
#         :return: self
#         """
#         features = np.arange(X.shape[1])
#         X_ = np.c_[X, y]
#         self.root = self.build(X_, features)
#         return self
#
#     def predict_one(self, x):
#         p = self.root
#         while p.label is None:
#             p = p.left if x[p.feature] == p.val else p.right
#         return p.label
#
#     def predict(self, X):
#         """
#         :param X: shape = [n_samples, n_features]
#         :return: shape = [n_samples]
#         """
#         return np.array([self.predict_one(x) for x in X])
#
#
# class DecisionTreeRegressor(object):
#     def __init__(self, max_depth=None):
#         self.max_depth = max_depth
#         self.root = None
#
#     @staticmethod
#     def feature_loss(X_, feature, val):
#         left, right = DecisionTreeClassifier.devide_X(X_, feature, val)
#         left_loss = np.sum((left[:, -1] - left[:, -1].mean()) ** 2)
#         right_loss = np.sum((right[:, -1] - right[:, -1].mean()) ** 2)
#         return left_loss + right_loss
#
#     @staticmethod
#     def feature_min_loss(X_, feature):
#         losses = np.array(list(map(lambda val: DecisionTreeRegressor.feature_loss(X_, feature, val), X_[:, feature])))
#         i = np.argmin(losses)
#         return X_[i, feature], losses[i]
#
#     @staticmethod
#     def get_best_index(X_, features):
#         losses = np.array([DecisionTreeRegressor.feature_min_loss(X_, feature) for feature in features])
#         i = np.argmin(losses[:, 1])
#         return features[i], losses[i, 0]
#
#     @staticmethod
#     def devide_X(X_, feature, val):
#         """切分集合
#         :param X_: 被切分的集合, shape=[ni_samples, ni_features + 1]
#         :param feature: 切分变量
#         :param val: 切分变量的值
#         :return: 左集合，有集合
#         """
#         return X_[X_[:, feature] <= val], X_[X_[:, feature] >= val]
#
#     def build(self, X_, features, depth=None):
#         pass
#
#     def fit(self, X, y):
#         pass
#
#     def predict_one(self, x):
#         p = self.root
#         while p.label is None:
#             p = p.left if x[p.feature] <= p.val else p.right
#         return p.label
