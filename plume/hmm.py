import numpy as np


class HMMEstimator(object):
    """隐马尔科夫模型(hidden Markov model, HMM)"""

    def __init__(self, n_states=None,
                 transmat=None,
                 obprob=None,
                 startprob=None,
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

    def _forward(self, obs):
        alpha = np.zeros((obs.shape[0], self.startprob.shape[0]))
        alpha[0] = self.startprob * self.obprob[:, obs[0]]
        for t in range(obs.shape[0] - 1):
            alpha[t + 1] = alpha[t] @ self.transmat * self.obprob[:, obs[t + 1]]
        return alpha

    def _backward(self, obs):
        beta = np.zeros((obs.shape[0], self.startprob.shape[0]))
        beta[obs.shape[0] - 1] = np.ones(self.startprob.shape[0])
        for t in range(obs.shape[0] - 2, -1, -1):
            beta[t] = self.obprob[:, obs[t + 1]] * beta[t + 1] @ self.transmat.T
        return beta

    def forward_prob(self, obs):
        alpha = self._forward(obs)
        return alpha[alpha.shape[0] - 1].sum()

    def backward_prob(self, obs):
        beta = self._backward(obs)
        return np.sum(self.startprob * self.obprob[:, obseq[0]] * beta[0])

    def fit(self, obs):
        """根据已有的观测序列，推测模型的参数"""
        obcount = np.unique(obs).shape[0]
        self.transmat = np.ones((self.n_states, self.n_states)) / self.n_states
        self.obprob = np.ones((self.n_states, obcount)) / obcount
        self.startprob = np.ones(self.n_states) / self.n_states
        for _ in range(self.n_iter):
            p = self.transmat.copy(), self.obprob.copy(), self.startprob.copy()
            g = self._gamma(obs)
            self.transmat = np.sum(self._xi(obs), axis=0) / np.sum(g[:-1])
            for j in range(self.n_states):
                for k in range(obcount):
                    temp = 0.0
                    for t in range(obseq.shape[0]):
                        if obs[t] == k:
                            temp += g[t][j]
                    self.obprob[j, k] = temp / np.sum(g[:, j])

            if np.linalg.norm(p[0] - self.transmat) < self.error and \
                            np.linalg.norm(p[1] - self.obprob) < self.error and \
                            np.linalg.norm(p[2] - self.startprob) < self.error:
                break

    def predict(self, obs):
        """给定一个观测序列，返回最有可能的状态序列"""
        pass

    def score(self, obs, method='forward'):
        """返回一个观测序列出现的概率"""
        if method == 'forward':
            return self.forward_prob(obs)
        else:
            return self.backward_prob(obs)

    def decoding(self, obs):
        """给定一个观测序列，返回最有可能的状态序列"""
        pass

    def decode(transmat, obprob, startprob, obseq):
        pass

    def _gamma(self, obs):
        alpha = self._forward(obs)
        beta = self._backward(obs)
        return alpha * beta / (alpha @ beta.T)

    def _xi(self, obs):
        alpha = self._forward(obs)
        beta = self._backward(obs)
        x = np.zeros((obs.shape[0] - 1, self.startprob.shape[0], self.startprob.shape[0].shape[0]))
        for t in range(0, obseq.shape[0] - 1):
            tmp_mat = alpha[t] * self.transmat * self.obprob[:, obseq[t + 1]] * beta[t + 1]
            x[t] = tmp_mat / tmp_mat.sum()
        return x


def _forward(transmat, obprob, startprob, obs):
    alpha = np.zeros((obs.shape[0], startprob.shape[0]))
    alpha[0] = startprob * obprob[:, obs[0]]
    for t in range(obs.shape[0] - 1):
        alpha[t + 1] = alpha[t] @ transmat * obprob[:, obs[t + 1]]
    return alpha


def _backward(transmat, obprob, startprob, obs):
    beta = np.zeros((obs.shape[0], startprob.shape[0]))
    beta[obs.shape[0] - 1] = np.ones(startprob.shape[0])
    for t in range(obs.shape[0] - 2, -1, -1):
        beta[t] = obprob[:, obs[t + 1]] * beta[t + 1] @ transmat.T
    return beta


def forward_prob(transmat, obprob, startprob, obseq):
    alpha = _forward(transmat, obprob, startprob, obseq)
    return alpha[alpha.shape[0] - 1].sum()


def backward_prob(transmat, obprob, startprob, obseq):
    beta = _backward(transmat, obprob, startprob, obseq)
    return np.sum(startprob * obprob[:, obseq[0]] * beta[0])


def predict_ob_prob(transmat, obprob, startprob, obseq):
    pass


def gamma(transmat, obprob, startprob, obseq):
    pass


def xi(transmat, obprob, startprob, obseq):
    pass


def get_model_param(obseq, n_components, n_ob, error=0.2):
    pass
