from math import pow
from collections import defaultdict
from multiprocessing import Process, cpu_count, Queue
import numpy as np


class Neighbor(object):
    """
    一个结构体，用来描述一个邻居所属的类别和与该邻居的距离
    """

    def __init__(self, class_label, distance):
        """
        :param class_label: 类别(y).
        :param distance: 距离.
        初始化。
        """
        self.class_label = class_label
        self.distance = distance


class KNeighborClassifier(object):
    """
    K-近邻算法分类器(k-Nearest Neighbor, KNN)，无KD树优化。
    """

    def __init__(self, n_neighbors=5, metric='euclidean'):
        """
        :param n_neighbors: 近邻数，默认为5.
        :param metric: 测算距离采用的度量,默认为欧氏距离.
        初始化。
        """
        self.n_neighbors = n_neighbors

        # p=2为欧氏距离，p=1为曼哈顿距离，其余的方式可自行添加。
        if metric == 'euclidean':
            self.p = 2
        elif metric == 'manhattan':
            self.p = 1

    def fit(self, train_x, train_y):
        """
        :param train_x: 训练集X.
        :param trian_y: 训练集Y.
        :return: None
        接收训练参数
        """
        self.train_x = train_x.astype(np.float32)
        self.train_y = train_y

    def predict_one(self, one_test):
        '''
        :param one_test: 测试集合的一个样本
        :return: test_x的类别
        预测单个样本
        '''
        # 用于储存所有样本点与测试点之间的距离
        neighbors = []
        for x, y in zip(self.train_x, self.train_y):
            distance = self.get_distance(x, one_test)
            neighbors.append(Neighbor(y, distance))

        # 将邻居根据距离由小到大排序
        neighbors.sort(key=lambda x: x.distance)

        # 如果近邻值大于训练集的样本数，则用后者取代前者
        if self.n_neighbors > len(self.train_x):
            self.n_neighbors = len(self.train_x)

        # 用于储存不同标签的近邻数
        cls_count = defaultdict(int)

        for i in range(self.n_neighbors):
            cls_count[neighbors[i].class_label] += 1

        # 返回结果
        ans = max(cls_count, key=cls_count.get)
        return ans

    def predict(self, test_x):
        '''
        :param test_x: 测试集
        :return: 测试集的预测值
        预测一个测试集
        '''
        return np.array([self.predict_one(x) for x in test_x])

    def get_distance(self, input, x):
        """
        :param input: 训练集的一个样本.
        :param x: 测试集合.
        :return: 两点距离
        工具方法，求两点之间的距离.
        """
        if self.p == 2:
            return np.linalg.norm(input - x)
        ans = 0
        for i, t in zip(input, x):
            ans += pow(abs(i - t), self.p)
        return pow(ans, 1 / self.p)


class ParallelKNClassifier(KNeighborClassifier):
    """
    并行K近邻算法分类器
    """

    def __init__(self, n_neighbors=5, metric='euclidean'):
        super(ParallelKNClassifier, self).__init__(n_neighbors, metric)
        self.task_queue = Queue()
        self.ans_queue = Queue()

    def do_parallel_task(self):
        '''
        :return: None
        单个进程的，并行任务。
        进程不断从任务队列里取出测试样本，
        计算完成后将参数放入答案队列
        '''
        while not self.task_queue.empty():
            id, one = self.task_queue.get()
            ans = self.predict_one(one)
            self.ans_queue.put((id, ans))

    def predict(self, test_x):
        '''
        :param test_x: 测试集
        :return: 测试集的预测值
        预测一个测试集
        '''
        for i, v in enumerate(test_x):
            self.task_queue.put((i, v))

        pool = []
        for i in range(cpu_count()):
            process = Process(target=self.do_parallel_task)
            pool.append(process)
            process.start()
        for i in pool:
            i.join()

        ans = []
        while not self.ans_queue.empty():
            ans.append(self.ans_queue.get())

        ans.sort(key=lambda x: x[0])
        ans = np.array([i[1] for i in ans])
        return ans
