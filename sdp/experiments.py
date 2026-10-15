"""Experiment loops: tuning, cross-version training/testing, and the within-release check."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .config import BASE_SEED, DATA_DIR, N_REPEATS, PAIRS, WV_REPEATS
from .data import list_releases, load_release
from .metrics import all_buggy_f1, score_all
from .models import make_model, undersample
from .tuning import tune

# (label, model kind, tuned family, undersample?)   tuned family picks which tuned params to use
CONFIGS = [("LR", "LR", "LR", False), ("LR + undersampling", "LR", "LR", True),
           ("LR + class weights", "LR-weighted", "LR", False),
           ("RF", "RF", "RF", False), ("RF + undersampling", "RF", "RF", True),
           ("RF + class weights", "RF-weighted", "RF", False)]


def fit_and_score(train, test, kind, params, us, seed):
    """Fit one model on the training release and score it on the test release."""
    Xtr, ytr = train.X, train.y
    if us:
        Xtr, ytr = undersample(Xtr, ytr, np.random.default_rng(seed))
    model = make_model(kind, seed, params).fit(Xtr, ytr)
    return score_all(test.y, model.predict_proba(test.X)[:, 1], test.bugs, test.loc)


def tune_all(pairs=PAIRS, data_dir=DATA_DIR):
    """Tune LR and RF once per training release. Returns {old_release: {'LR': params, 'RF': params}}
    and a long table of every candidate (for the report)."""
    best, rows = {}, []
    for old, _ in pairs:
        tr = load_release(old, data_dir)
        best[old] = {}
        for fam in ("LR", "RF"):
            p, auc, table = tune(fam, tr.X, tr.y)
            best[old][fam] = p
            rows += [dict(train_release=old, model=fam, params=str(t["params"]), mean_auc=t["mean_auc"],
                          sd_auc=t["sd_auc"], rank=t["rank"]) for t in table]
    return best, pd.DataFrame(rows)


def cross_version(best_params=None, pairs=PAIRS, n_repeats=N_REPEATS, data_dir=DATA_DIR, configs=CONFIGS):
    """Train on the older release, test on the newer one, n_repeats times with seeds BASE_SEED+r.
    best_params=None uses library defaults (the untuned baseline)."""
    rows = []
    for old, new in pairs:
        tr, te = load_release(old, data_dir), load_release(new, data_dir)
        seen = {tuple(r) for r in tr.X}
        identical = float(np.mean([tuple(r) in seen for r in te.X]))    # unchanged files across releases
        rows.append(dict(pair=f"{old} -> {new}", test=new, model="All-buggy rule", tuned=False,
                         f1=all_buggy_f1(te.y), auc=0.5, pofb20=np.nan, pofb20_per_loc=np.nan, f1_sd=0.0,
                         train_rate=tr.y.mean(), test_rate=te.y.mean(), identical_rows=identical))
        for label, kind, fam, us in configs:
            params = best_params[old][fam] if best_params else None
            res = pd.DataFrame([fit_and_score(tr, te, kind, params, us, BASE_SEED + r) for r in range(n_repeats)])
            rows.append(dict(pair=f"{old} -> {new}", test=new, model=label, tuned=best_params is not None,
                             f1=res.f1.mean(), auc=res.auc.mean(), pofb20=res.pofb20.mean(),
                             pofb20_per_loc=res.pofb20_per_loc.mean(), f1_sd=res.f1.std(ddof=0),
                             train_rate=tr.y.mean(), test_rate=te.y.mean(), identical_rows=identical))
    return pd.DataFrame(rows)


def within_version(n_repeats=WV_REPEATS, data_dir=DATA_DIR):
    """Repeated stratified 5-fold CV inside ONE release, for all 12 releases (default parameters).
    Not the paper's protocol (it ignores release order); only gives a number for Ant-1.7 and log4j-1.1."""
    rows = []
    for name in list_releases(data_dir):
        rel = load_release(name, data_dir)
        for label in ("LR", "RF"):
            per_rep = []
            for rep in range(n_repeats):
                folds = StratifiedKFold(5, shuffle=True, random_state=BASE_SEED + rep).split(rel.X, rel.y)
                sc = []
                for tr_idx, te_idx in folds:
                    model = make_model(label, BASE_SEED + rep).fit(rel.X[tr_idx], rel.y[tr_idx])
                    sc.append(score_all(rel.y[te_idx], model.predict_proba(rel.X[te_idx])[:, 1],
                                        rel.bugs[te_idx], rel.loc[te_idx]))
                per_rep.append(pd.DataFrame(sc).mean())
            m = pd.DataFrame(per_rep).mean()
            rows.append(dict(dataset=name, model=label, f1=m.f1, auc=m.auc, pofb20=m.pofb20,
                             f1_trivial=all_buggy_f1(rel.y)))
    return pd.DataFrame(rows)
