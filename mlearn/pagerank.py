import numpy as np

def page_rank(pages, rounds=20, alpha=0.001):
    rank = np.array([1.0 / pages.shape[0]] * pages.shape[0])
    smooth = alpha / rank.shape[0] * np.eye(rank.shape[0]) + (1 - alpha) * pages
    for i in range(rounds):
        rank = np.dot(smooth.T, rank)
    return rank