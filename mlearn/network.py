import numpy as np

class BPNetWork(object):
    """
    全连接神经网络，采用BP算法训练。
    """
    
    def __init__(self, layers, act_func='tanh'):
        """
        :param layers: 神经网络的结构
        :param act_func: 激励函数
        输入样例：
        ann = BPNN((2, 3, 1))
        表示一个输入层，一个隐含层，一个输出层，输入层有2个结点，
        隐含层有3个结点，输出层有1个结点
        ann = BPNN((2, 3, 3, 1))
        表示一个输入层，二个隐含层，一个输出层，输入层有2个结点，
        第一层隐含层有3个结点，第二层隐含层有3个结点，输出层有1个结点
        """

        # 初始化神经元的值
        self.networks = []

        # 初始化神经元权重
        self.weights = []
        for i in range(len(layers) - 1):
            weight = 2 * np.random.random((layers[i], layers[i + 1])) - 1
            network = [1.0] * layers[i]
            self.networks.append(network)
            self.weights.append(weight)
        self.networks.append([1.0] * layers[-1])
        self.networks = np.array(self.networks)

        # 初始化神经元阈值
        self.thresholds = []
        for i in range(1, len(layers)):
            threshold = 2 * np.random.random(layers[i]) - 1
            self.thresholds.append(threshold)

        # 选择激励函数和它的导函数
        if act_func == 'tanh':

            self.act_func = self.tanh
            self.dact_func = self.dthanh
        else:
            self.act_func = self.sigmoid
            self.dact_func = self.dsigmoid

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def dthanh(self, x):
        return 1.0 - np.tanh(x) ** 2

    def fit(self, train_x, train_y, epochs, learn_rate):
        '''
        :param train_x: 训练集X
        :param train_y: 训练集Y
        :param epochs: 迭代次数
        :param learn_rate: 学习率，步长
        :return: None
        拟合神经网络
        '''

        for i in range(epochs):
            i = np.random.randint(train_x.shape[0], high=None)
            self.update(train_x[i])
            self.back_propagate(train_y[i], learn_rate)

    def predict(self, test_x):
        '''
        :param test_x: 测试集合
        :return: 预测值
        '''
        self.update(test_x)
        return self.networks[-1].copy()

    def update(self, inputs):
        '''
        :param inputs: X的输入值
        :return: None
        更新一次神经元的值
        '''
        self.networks[0] = inputs.copy()
        for i in range(len(self.weights)):
            count = np.dot(self.networks[i], self.weights[i]) - self.thresholds[i]
            self.networks[i + 1] = self.act_func(count)

    def back_propagate(self, y, rate):
        '''
        :param y: target
        :param rate: 学习率
        :return: None
        BP算法，对神经网络的权值和阈值进行更新
        '''
        errors = y - self.networks[-1]
        gradients = [self.dact_func(self.networks[-1]) * errors]

        self.thresholds[-1] += (-1) * rate * gradients[-1]
        for i in range(len(self.weights) - 1, 0, -1):
            gradients.append(gradients[-1].dot(self.weights[i].T) * self.dact_func(self.networks[i]))
            self.thresholds[i - 1] += (-1) * rate * gradients[-1]

        gradients.reverse()
        for i in range(len(self.weights)):
            self.weights[i] += rate * self.networks[i].reshape((-1, 1)) * gradients[i]

