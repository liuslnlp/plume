from collections import defaultdict
import numpy as np
from math import log


class TreeNode(object):
    """决策树节点"""

    def __init__(self, **kwargs):
        '''
        attr_index: 属性编号
        attr: 属性值
        label: 类别（y）
        left_chuld: 左子结点
        right_child: 右子节点
        '''
        self.attr_index = kwargs.get('attr_index')
        self.attr = kwargs.get('attr')
        self.label = kwargs.get('label')
        self.left_child = kwargs.get('left_child')
        self.right_child = kwargs.get('right_child')


class RandomForestsClassifier(object):
    """
    随机森林（Random Forests）分类器，使用的决策树是CART树。
    """
    def __init__(self, tree_num=5):
        '''
        :param tree_num: 森林使用决策树的个数
        '''
        self.tree_num = tree_num
        self.roots = []
        self.k = None
    
    def gini(self, cluster):
        '''
        :param cluster: 训练集的一个子集
        :return: 数据集的基尼系数
        求给定数据集的基尼系数
        '''
        p = defaultdict(int)
        for line in cluster:
            p[line[-1]] += 1
        temp = 1.0
        for k, v in p.items():
            temp -= (v / len(cluster)) ** 2
        
        return temp
    
    def gini_index(self, cluster, attr_index):
        '''
        :param cluster: 训练集的一个子集
        :param attr_index:  特征编号（第N个特征）
        :return: 第N个特征的特征值， 该值的基尼指数
        返回给定列标号下的最优切分属性和该属性的基尼指数
        '''
        p = defaultdict(list)
        for line in cluster:
            p[line[attr_index]].append(line)
        attr_gini = {}
        for k, v in p.items():
            els = []
            for k1, v1 in p.items():
                if k1 == k:
                    continue
                els.extend(v)
            count = (self.gini(v) * len(v) + self.gini(els) * len(els)) / len(cluster)  
            attr_gini[k] = count    
        attr = min(attr_gini, key=attr_gini.get)
        return attr, attr_gini[attr]
    
    def devide_set(self, cluster, index, attr):
        '''
        :param cluster: 给定集合（为训练集的一个子集）
        :param index: 特征编号
        :param attr: 特征值
        :return: 左半部分，右半部分
        将给定集合切分为两部分返回，第index个特征的特征值等于attr的为一组
        不等于attr的为一组。
        '''
        left = []
        right = []
        for line in cluster:
            if line[index] == attr:
                left.append(line)
            else:
                right.append(line)
        return np.array(left), np.array(right)

    def get_best_index(self, cluster, attr_indexs):
        '''
        :param cluster: 给定数据集
        :param attr_indexs: 给定的可供切分的特征编号的集合
        :return: 最佳切分点，最佳切分变量
        求给定切分点集合中的最佳切分点和其对应的最佳切分变量
        '''
        temp = np.array(list(attr_indexs))
        indexs = np.random.choice(temp.shape[0], self.k)
        new_attr_indexs = temp[indexs]
        p = {}
        for attr_index in new_attr_indexs:
            p[attr_index] = (self.gini_index(cluster, attr_index))
        attr_index = min(p, key = lambda x: p.get(x)[1])
        attr = p[attr_index][0]
        return attr_index, attr
    
    def vote(self, cluster):
        '''
        :param cluster: 给定数据集
        :return: 给定数据集中所占比重最大的类别
        投票选择，将给定数据集中所占比重最大的类别选出
        '''
        p = defaultdict(int)
        for line in cluster:
            p[line[-1]] += 1
        return max(p, key=p.get)

    def build_tree(self, cluster, attr_indexs):
        '''
        :param cluster: 给定数据集
        :param attr_indexs: 给定的可供切分的特征编号的集合
        :return: 一个决策树结点
        递归构建决策树
        '''
        flag = cluster[0, -1]
        for i in cluster[:, -1]:
            if i != flag:
                break
        else:
            return TreeNode(label=flag)
        if not attr_indexs:
            return TreeNode(label=self.vote(cluster))
        for i in attr_indexs:
            flag = cluster[i][0]
            f = False
            for j in cluster[:, i]:
                if j != flag:
                    f = True
                    break
            if f:
                break
        else:
            return TreeNode(label=self.vote(cluster))

        attr_index, attr = self.get_best_index(cluster, attr_indexs)
        left, right = self.devide_set(cluster, attr_index, attr)
        if left.shape[0] == 0:
            return TreeNode(label=self.vote(right))
        elif right.shape[0] == 0:
            return TreeNode(label=self.vote(left))

        new_attr_indexs = attr_indexs - set([attr_index])

        left_branch = self.build_tree(left, new_attr_indexs)
        right_branch = self.build_tree(right, new_attr_indexs)

        return TreeNode(left_child=left_branch,
                        right_child=right_branch,
                        attr_index=attr_index,
                        attr=attr)
  
    def choose_set(self, train_x):
        '''
        :param train_x: 总数据集
        :return: 新数据集
        从给定数据集中放回的抽取一个容量为size(train_x)的新数据集
        '''
        indexs = np.random.choice(train_x.shape[0], train_x.shape[0])
        return train_x.copy()[indexs]

    def fit(self, train_x, train_y):
        '''
        :param train_x: 训练集合X
        :param train_y: 训练集合Y（target）
        :return: None
        拟合随机森林
        '''
        attr_indexs = set(range(train_x.shape[1]))
        self.k = int(log(len(attr_indexs), 2))
        self.train_x = np.c_[train_x, train_y]
        for i in range(self.tree_num):
            new_train_x = self.choose_set(self.train_x)
            self.roots.append(self.build_tree(new_train_x, attr_indexs))
        self.roots = np.array(self.roots)
     
    def predict_in_one_tree(self, root, x):
        '''
        :param root: 选定的树的根结点
        :param x: 待预测的样本X
        :return: X在这颗树上的预测值
        在单颗决策树上预测结果
        '''
        node_p = root
        while node_p.label == None:
            if x[node_p.attr_index] == node_p.attr:
                node_p = node_p.left_child
            else:
                node_p = node_p.right_child
        return node_p.label
    
    def predict_one(self, x):
        '''
        :param x:  待预测的样本X
        :return: X所属的类别
        预测单个值
        '''
        count = defaultdict(int)
        for root in self.roots:
            label = self.predict_in_one_tree(root, x)
            count[label] += 1
        return max(count, key=count.get)
        
    def predict(self, test_x):
        '''
        :param test_x: 测试集
        :return: 测试集样本的类别集合
        预测多个值
        '''
        return np.array([self.predict_one(x) for x in test_x])
    
