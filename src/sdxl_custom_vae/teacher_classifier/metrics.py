from __future__ import annotations
import numpy as np


def compute_multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score

    C = y_true.shape[1]
    aurocs, auprcs = [], []

    for c in range(C):
        yt, yp = y_true[:, c], y_prob[:, c]
        if yt.max() == 0 or yt.min() == 1:
            aurocs.append(None)
            auprcs.append(None)
            continue
        aurocs.append(float(roc_auc_score(yt, yp)))
        auprcs.append(float(average_precision_score(yt, yp)))

    def safe_mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    return {
        "macro_auroc": safe_mean(aurocs),
        "macro_auprc": safe_mean(auprcs),
        "per_class_auroc": aurocs,
        "per_class_auprc": auprcs,
    }
