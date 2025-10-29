import json, random, numpy as np, torch
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, accuracy_score, hamming_loss

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_pos_weight(Y):
    N, C = Y.shape
    pc = Y.sum(axis=0).clip(min=1.0)
    weight = (N - pc) / pc
    return torch.tensor(weight, dtype=torch.float32)

def compute_metrics(y_true, y_pred):
    return {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "subset_acc": accuracy_score(y_true, y_pred),
        "jaccard_micro": jaccard_score(y_true, y_pred, average="micro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
    }
