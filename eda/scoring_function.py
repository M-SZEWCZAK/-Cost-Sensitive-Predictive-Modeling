import pandas as pd
import numpy as np
from pandas.conftest import ascending


def score_model_optimal_k(y_true, y_proba, n_vars, max_k=1000, keep_fp_tp=False):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba).squeeze()

    order = np.argsort(y_proba)[::-1]
    y_true_sorted = y_true[order]

    best_score = -np.inf
    best_k = 0
    tp = 0
    fp = 0
    best_fp=0
    best_tp=0
    for k in range(1, min(max_k, len(y_true)) + 1):
        if y_true_sorted[k-1] == 1:
            tp += 1
        else:
            fp += 1

        score = (tp * 10) - (fp * 5) - (n_vars * 200)

        if score > best_score:
            best_score = score
            best_k = k
            best_fp = fp
            best_tp = tp
    if keep_fp_tp:
        return best_score, best_k, best_fp, best_tp
    return best_score, best_k