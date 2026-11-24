import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score)


def precision_at_k(y_true, proba, frac):
    """Churn rate among the top `frac` of customers ranked by risk."""
    k = int(len(proba) * frac)
    top = np.argsort(proba)[::-1][:k]
    return np.asarray(y_true)[top].mean()


def evaluate(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall":    recall_score(y_true, pred),
        "roc_auc":   roc_auc_score(y_true, proba),
        "pr_auc":    average_precision_score(y_true, proba),
        "prec@10%":  precision_at_k(y_true, proba, 0.10),
        "prec@20%":  precision_at_k(y_true, proba, 0.20),
    }
