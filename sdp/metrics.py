"""Evaluation metrics: F1, AUC, PofB20 and the trivial reference rule."""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def pofb20(score, bugs, loc):
    """Share of all bugs found when classes are inspected in order of decreasing score
    until 20% of the total lines of code have been read.  Returns nan if there are no bugs."""
    score, bugs, loc = (np.asarray(a, dtype=float) for a in (score, bugs, loc))
    if bugs.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")          # stable: ties keep file order
    k = np.searchsorted(np.cumsum(loc[order]), 0.2 * loc.sum(), side="right")
    return float(bugs[order][:k].sum() / bugs.sum())


def pofb20_per_loc(score, bugs, loc):
    """Variant of PofB20: rank by score per line of code (the usual effort-aware order)."""
    return pofb20(np.asarray(score, dtype=float) / np.maximum(np.asarray(loc, dtype=float), 1.0), bugs, loc)


def all_buggy_f1(y):
    """F1 of the rule 'every class is buggy': precision = p (share of buggy), recall = 1."""
    p = float(np.mean(y))
    return 2 * p / (1 + p)


def score_all(y_true, proba, bugs, loc, threshold=0.5):
    """F1 at a fixed threshold, AUC, PofB20 and PofB20 per LOC for one test release."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {"f1": f1_score(y_true, pred, zero_division=0),
            "auc": roc_auc_score(y_true, proba),
            "pofb20": pofb20(proba, bugs, loc),
            "pofb20_per_loc": pofb20_per_loc(proba, bugs, loc)}
