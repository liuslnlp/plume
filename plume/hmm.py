import numpy as np


class HMMEstimator(object):
    """隐马尔科夫模型(hidden Markov model, HMM)"""

    def __init__(self, transmat=None,
                 obprob=None,
                 startprob=None,
                 n_states=None,
                 n_iter=100,
                 error=0.1):

        self.transmat = transmat
        self.obprob = obprob
        self.startprob = startprob
        self.n_iter = n_iter
        self.error = error
        if startprob is not None:
            self.n_states = startprob.shape[0]
        else:
            self.n_states = n_states

    def fit(self, seq):
        """根据已有的观测序列，推测模型的参数"""
        obcount = np.unique(seq).shape[0]
        self.transmat = np.ones((self.n_states, self.n_states)) / self.n_states
        self.obprob = np.ones((self.n_states, obcount)) / obcount
        self.startprob = np.ones(self.n_states) / self.n_states
        for _ in range(self.n_iter):
            p = self.transmat.copy(), self.obprob.copy(), self.startprob.copy()
            g = self._gamma(seq)
            self.transmat = np.sum(self._xi(seq)[:-1], axis=0) / np.sum(g[:-1], axis=0)
            for k in range(obcount):
                b[:, k] = np.sum(g[seq == k], axis=0) / np.sum(g, axis=0)

            # for j in range(self.n_states):
            #     for k in range(obcount):
            #         temp = 0.0
            #         for t in range(obseq.shape[0]):
            #             if seq[t] == k:
            #                 temp += g[t][j]
            #         self.obprob[j, k] = temp / np.sum(g[:, j])

            self.startprob = g[0]

            if np.linalg.norm(p[0] - self.transmat) < self.error and \
                            np.linalg.norm(p[1] - self.obprob) < self.error and \
                            np.linalg.norm(p[2] - self.startprob) < self.error:
                break

    def predict(self, seq):
        """给定一个观测序列，返回最有可能的状态序列"""
        pass

    def score(self, seq, method='forward'):
        """返回一个观测序列出现的概率"""
        if method == 'forward':
            return self.forward_prob(seq)
        else:
            return self.backward_prob(seq)

    def decoding(self, seq):
        """给定一个观测序列，返回最有可能的状态序列"""
        seqlen = seq.shape[0]
        delte = self.startprob * self.obprob[:, seq[0]]
        psi = np.zeros((seqlen, self.n_states), dtype=np.int8)
        for i in range(1, seqlen):
            tmp = delte * self.transmat.T
            psi[i] = np.argmax(tmp, axis=1)
            delte = tmp.max(axis=1) * self.obprob[:, seq[i]]
        path = np.zeros(seqlen, dtype=np.int8)
        path[seqlen - 1] = np.argmax(delte)
        optimal_path_prob = delte[path[seqlen - 1]]
        for t in range(seqlen - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path, optimal_path_prob

    def _forward(self, seq):
        alpha = np.zeros((seq.shape[0], self.startprob.shape[0]))
        alpha[0] = self.startprob * self.obprob[:, seq[0]]
        for t in range(seq.shape[0] - 1):
            alpha[t + 1] = alpha[t] @ self.transmat * \
                           self.obprob[:, seq[t + 1]]
        return alpha

    def forward_prob(self, seq):
        alpha = self._forward(seq)
        return alpha[alpha.shape[0] - 1].sum()

    def backward_prob(self, seq):
        beta = self._backward(seq)
        return np.sum(self.startprob * self.obprob[:, seq[0]] * beta[0])

    def _backward(self, seq):
        beta = np.zeros((seq.shape[0], self.startprob.shape[0]))
        beta[seq.shape[0] - 1] = np.ones(self.startprob.shape[0])
        for t in range(seq.shape[0] - 2, -1, -1):
            beta[t] = self.obprob[:, seq[t + 1]] * \
                      beta[t + 1] @ self.transmat.T
        return beta

    def _gamma(self, seq):
        """
        :param seq: shape = [n_times]
        :return: gamma[t][i] = P(it = qi | O, lambda) shape = [n_times, n_states]
        """
        alpha = self._forward(seq)
        beta = self._backward(seq)
        return alpha * beta / (alpha @ beta.T)

    def _xi(self, seq):
        """
        :param seq: shape = [n_times]
        :return: shape = [n_times, n_states, n_states]
        """
        alpha = self._forward(seq)
        beta = self._backward(seq)
        x = np.zeros(
            (seq.shape[0] - 1, self.startprob.shape[0], self.startprob.shape[0].shape[0]))
        for t in range(0, obseq.shape[0] - 1):
            tmp_mat = alpha[t] * self.transmat * \
                      self.obprob[:, obseq[t + 1]] * beta[t + 1]
            x[t] = tmp_mat / tmp_mat.sum()
        return x
