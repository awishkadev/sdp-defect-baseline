"""Model factory and class balancing.  Scaling is part of the pipeline, so it is fitted on
whatever data the pipeline is fitted on (the training release) and never on test data."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def log1p_transform(X):
    """log(1 + x) for the heavy-tailed, non-negative count metrics."""
    return np.log1p(np.clip(X, 0, None))


def make_model(kind, seed, params=None):
    """kind: 'LR', 'LR-weighted', 'RF', 'RF-weighted'.  params overrides the defaults
    (the best values found by sdp.tuning)."""
    params = dict(params or {})
    if kind in ("LR", "LR-weighted"):
        cw = "balanced" if kind == "LR-weighted" else None
        C = params.get("logisticregression__C", 1.0)
        return make_pipeline(FunctionTransformer(log1p_transform), StandardScaler(),
                             LogisticRegression(C=C, max_iter=2000, class_weight=cw, random_state=seed))
    if kind in ("RF", "RF-weighted"):
        cw = "balanced_subsample" if kind == "RF-weighted" else None
        return RandomForestClassifier(n_estimators=200, class_weight=cw, random_state=seed, n_jobs=-1, **params)
    raise ValueError(f"unknown model kind: {kind}")


def undersample(X, y, rng):
    """Balanced random undersampling: keep all of the minority class and an equal number of
    randomly chosen majority classes.  Applied to TRAINING data only; inputs are not modified."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    idx = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    return X[idx], y[idx]
