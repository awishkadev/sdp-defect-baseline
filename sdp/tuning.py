"""Hyperparameter tuning.

Only the training (older) release is used.  Each candidate is scored with repeated stratified
5-fold cross-validation on ROC AUC (threshold-free, so it does not depend on class balance).
The test release is never touched.  The best setting is then fixed for all 30 repeats."""
import numpy as np
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from .config import BASE_SEED, GRID_LR, GRID_RF
from .models import make_model


def tune(kind, X, y, seed=BASE_SEED):
    """Return (best_params, best_cv_auc, table_of_all_candidates) for 'LR' or 'RF'."""
    grid = GRID_LR if kind.startswith("LR") else GRID_RF
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=seed)
    search = GridSearchCV(make_model(kind, seed), grid, scoring="roc_auc", cv=cv,
                          n_jobs=1, refit=False)
    search.fit(X, y)
    res = search.cv_results_
    table = [{"params": p, "mean_auc": m, "sd_auc": s, "rank": int(r)}
             for p, m, s, r in zip(res["params"], res["mean_test_score"], res["std_test_score"], res["rank_test_score"])]
    return search.best_params_, float(search.best_score_), table
