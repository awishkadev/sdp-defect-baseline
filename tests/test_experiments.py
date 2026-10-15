import numpy as np
from sdp import experiments as ex
from sdp.config import PAIRS
from sdp.data import load_release


def test_pair_test_release_is_never_in_training_release():
    for old, new in PAIRS:
        assert old != new


def test_tuning_uses_only_training_release(monkeypatch):
    """tune_all must load only the older release of every pair."""
    loaded = []
    real = ex.load_release
    monkeypatch.setattr(ex, "load_release", lambda n, d=None, **k: (loaded.append(n), real(n))[1])
    monkeypatch.setattr(ex, "tune", lambda fam, X, y: ({}, 0.5, []))
    ex.tune_all()
    assert set(loaded) == {old for old, _ in PAIRS}


def test_cross_version_smoke_and_schema():
    cv = ex.cross_version(None, pairs=PAIRS[:1], n_repeats=2, configs=ex.CONFIGS[:1])
    assert set(cv.model) == {"All-buggy rule", "LR"}
    row = cv[cv.model == "LR"].iloc[0]
    assert 0 <= row.f1 <= 1 and 0.5 <= row.auc <= 1


def test_whole_run_is_reproducible():
    a = ex.cross_version(None, pairs=PAIRS[:1], n_repeats=2, configs=ex.CONFIGS[3:4])
    b = ex.cross_version(None, pairs=PAIRS[:1], n_repeats=2, configs=ex.CONFIGS[3:4])
    assert np.allclose(a[["f1", "auc", "pofb20"]].fillna(-1), b[["f1", "auc", "pofb20"]].fillna(-1))


def test_trivial_rule_row_matches_formula():
    cv = ex.cross_version(None, pairs=PAIRS[:1], n_repeats=1, configs=[])
    te = load_release(PAIRS[0][1])
    p = te.y.mean()
    assert cv.iloc[0].f1 == np.float64(2 * p / (1 + p))
