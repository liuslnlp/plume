from collections import defaultdict


class FeatureElement(object):
    """
    单个特征元素。
    """

    def __init__(self, feature_label, label, class_label):
        """
        :param feature_label: int 特征编号.
        :param label: str 特征值.
        :param class_label: str 类别.
        """
        self.feature_label = feature_label
        self.label = label
        self.class_label = class_label

    def __eq__(self, other):
        if self.feature_label == other.feature_label \
                and self.label == other.label \
                and self.class_label == other.class_label:
            return True
        else:
            return False

    def __hash__(self):
        return (hash(self.class_label) + hash(self.class_label)) << self.feature_label


class NaiveBayesClassifier(object):
    """
    朴素贝叶斯分类器(Naive Bayes Model)。
    """

    def __init__(self, lamda=1):
        """
        :param lamda: 系数，默认为1.
        初始化
        """
        self.lamda = lamda
        # 先验概率
        self.prior_probabilities = {}
        # 条件概率
        self.conditional_probabilities = {}

    def train(self):
        """
        学习训练集，建立模型。
        """
        class_labels = defaultdict(int)
        for class_label in self.train_y:
            class_labels[class_label] += 1

            # 先验概率计算
        # P(Y = Ck) = (ΣI(yi = Ck) + lamda) / (N + K*lamda)
        for k, v in class_labels.items():
            self.prior_probabilities[k] = (v + self.lamda) \
                                          / (len(self.train_y) + self.lamda * len(class_labels))

        conditional_count = defaultdict(int)
        for i in range(len(self.train_x)):
            for j in range(len(self.train_x[i])):
                temp = FeatureElement(j, self.train_x[i][j], self.train_y[i])
                conditional_count[temp] += 1

        label_set = []
        for i in range(len(self.train_x[0])):
            label_set.append(set())

        for k, v in conditional_count.items():
            label_set[k.feature_label].add(k.label)

        # 条件概率计算
        # P(X(i) = aji | Y = Ck) = (I(x(j)i = aji, yi = Ck) + lamda) / (ΣI(yi = Ck) + Sj*lamda)
        for class_label, count in class_labels.items():
            for k, v in conditional_count.items():
                if k.class_label == class_label:
                    temp = (v + self.lamda) \
                           / (count + self.lamda * len(label_set[k.feature_label]))
                    self.conditional_probabilities[k] = temp

    def fit(self, train_x, train_y):
        """
        :param train_x: 训练集合X.
        :param train_y: 训练集合Y.
        :return: None
        拟合
        """
        self.train_x = train_x
        self.train_y = train_y
        self.train()

    def predict_one(self, x):
        '''
        :param x: 测试集合
        :return: test_x的类别
        '''
        result_map = {}
        for feature, probability in self.prior_probabilities.items():
            class_label = feature
            result_map[class_label] = probability

            for k, v in self.conditional_probabilities.items():
                for i in range(len(x)):
                    if k.feature_label == i and k.label == x[i] \
                            and k.class_label == class_label:
                        result_map[class_label] *= v

        ans = max(result_map, key=result_map.get)
        return ans

    def predict(self, test_x):
        '''
        :param test_x: 测试集
        :return: 测试集的预测值
        预测一个测试集
        '''
        return [self.predict_one(x) for x in test_x]
