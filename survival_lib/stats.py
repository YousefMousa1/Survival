import numpy as np


def benjamini_hochberg(p_values):
    pvals = np.asarray(p_values, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    qvals = pvals * n / ranks
    qvals = np.minimum.accumulate(qvals[np.argsort(order)[::-1]])[::-1]
    qvals = np.clip(qvals, 0, 1)
    return qvals
