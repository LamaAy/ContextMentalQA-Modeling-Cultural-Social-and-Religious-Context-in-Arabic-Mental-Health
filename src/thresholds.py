import numpy as np

def compute_adaptive_thresholds(probs_val, y_val):
    N, C = probs_val.shape
    thresholds = {}
    for c in range(C):
        pi_c = y_val[:, c].mean()
        q = 1.0 - pi_c
        q = float(np.clip(q, 0.0, 1.0))
        tau_c = float(np.quantile(probs_val[:, c], q))
        thresholds[str(c)] = tau_c
    return thresholds
