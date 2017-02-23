"""
注意，本文件只完成了部分，预测算法后期补上
"""

import numpy as np

class HMM(object):
    """
    隐马尔可夫模型(Hidden Markov Model，HMM)，包含前向和后项两种算法。
    预测时，需给定模型 lamda = (A, B, pi),
    观测集合V,及观测序列O
    """
    
    def __init__(self, state_trans, ob_probability, vis_set, pi):
        """
        :param state_trans: 状态转移矩阵.
        :param ob_probability: 观测概率矩阵.
        :param vis_set: 所有可能观测的集合.
        :param pi: 初始状态的概率向量.
        初始化
        """
        self.state_trans = state_trans
        self.ob_probability = ob_probability
        self.vis_set = vis_set
        self.pi = pi
    
    def find_index(self, observations):
        """
        :param observations: 观测值.
        辅助函数用于求给定观测值对应的下标
        """
        for i, v in enumerate(self.vis_set):
            if v == observations:
                return i

    def forward_alg(self, test):
        """
        :param test: 观测序列.
        前向算法
        """
        alphas = []
        index = self.find_index(test[0])

        for i, j in zip(self.pi, self.ob_probability):
            alphas.append(i*j[index])

        for i in range(1, len(test)):
            new_alphas = []
            for status in range(len(test)):
                  temp = 0
                  for j, alpha in enumerate(alphas):
                      temp += self.state_trans[j][status] * alpha
                  index = self.find_index(test[i])
                  temp *= self.ob_probability[status][index]
                  new_alphas.append(temp)
            alphas = new_alphas

        return sum(alphas)

    def backward_alg(self, test):
        """
        :param test: 观测序列.
        后向算法
        """
        betas = []
        for i in range(len(self.ob_probability)):
            betas.append(1)

        for t in range(len(test)-2, -1, -1):
            new_betas = []
            for i in range(len(self.ob_probability)):
                temp = 0
                for j in range(len(self.ob_probability)):
                    index = self.find_index(test[t+1])
                    temp += self.state_trans[i][j] \
                        * self.ob_probability[j][index] * betas[j]
                new_betas.append(temp)
            betas = new_betas

        ans = 0
        index = self.find_index(test[0])
        
        for i in range(len(self.ob_probability)):
            ans += self.pi[i] * self.ob_probability[i][index] * betas[i]

        return ans


    def predict(self, test, algorithm='forward'):
        """
        :param test: 观测序列.
        :param algorithm: 使用的算法.
        预测
        """
        if algorithm == 'forward':
            return self.forward_alg(test)
        else:
            return self.backward_alg(test)


