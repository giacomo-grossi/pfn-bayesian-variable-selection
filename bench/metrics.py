"""Evaluation metrics of the thesis (Section 6.1.3): AUPRC of the ranking, FDP and TPR of the
median probability model."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ranking_metrics(score, gamma):
    score = np.asarray(score, dtype=float)
    gamma = np.asarray(gamma, dtype=int)
    if gamma.min() == gamma.max():
        return {"auprc": np.nan, "auroc": np.nan}
    return {"auprc": float(average_precision_score(gamma, score)),
            "auroc": float(roc_auc_score(gamma, score))}


def selection_metrics(selected, gamma):
    sel = np.asarray(selected, dtype=bool)
    g = np.asarray(gamma, dtype=bool)
    tp = int((sel & g).sum())
    k = int(sel.sum())
    return {"tpr": tp / max(1, int(g.sum())),
            "fdp": (k - tp) / max(1, k),
            "fpr": int((sel & ~g).sum()) / max(1, int((~g).sum())),
            "fnr": 1 - tp / max(1, int(g.sum())),
            "exact": float(np.array_equal(sel, g)),
            # F1 as in Ye and Li (2025): 2TP / (2TP + FP + FN); 0 when nothing is selected
            "f1": 2 * tp / max(1, 2 * tp + (k - tp) + (int(g.sum()) - tp)),
            "n_selected": k}
