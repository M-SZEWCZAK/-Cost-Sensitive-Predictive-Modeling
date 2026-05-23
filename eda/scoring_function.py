import numpy as np


def score_model_optimal_k(y_true, y_proba, n_vars, max_k=1000, keep_fp_tp=False,tp_reward=10,fp_penalty=5,feature_penalty=200):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba).squeeze()

    order = np.argsort(y_proba)[::-1]
    y_true_sorted = y_true[order]
    y_proba_sorted = y_proba[order]
    best_score = -np.inf
    best_k = 0
    tp = 0
    fp = 0
    best_fp=0
    best_tp=0
    best_threshold=0
    for k in range(1, min(max_k, len(y_true)) + 1):
        if y_true_sorted[k-1] == 1:
            tp += 1
        else:
            fp += 1

        score = (tp * tp_reward) - (fp * fp_penalty) - (n_vars * feature_penalty)

        if score > best_score:
            best_score = score
            best_k = k
            best_fp = fp
            best_tp = tp
            best_threshold = y_proba_sorted[k - 1]
    if keep_fp_tp:
        return best_score,best_threshold, best_k, best_fp, best_tp
    return best_score,best_threshold, best_k