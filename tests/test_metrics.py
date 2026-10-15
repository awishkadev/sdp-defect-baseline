import numpy as np
import pytest
from sdp.metrics import all_buggy_f1, pofb20, pofb20_per_loc, score_all


def test_all_buggy_f1_formula():
    y = np.array([1, 1, 0, 0])           # p = 0.5 -> F1 = 2*0.5/1.5
    assert all_buggy_f1(y) == pytest.approx(2 / 3)


def test_all_buggy_f1_matches_sklearn():
    from sklearn.metrics import f1_score
    y = np.array([1, 0, 0, 0, 1, 0])
    assert all_buggy_f1(y) == pytest.approx(f1_score(y, np.ones_like(y)))


def test_pofb20_perfect_and_worst_ranking():
    loc = np.array([100, 100, 100, 100, 100])
    bugs = np.array([5, 0, 0, 0, 0])
    assert pofb20(np.array([9, 1, 1, 1, 1.0]), bugs, loc) == pytest.approx(1.0)   # buggy file read first
    assert pofb20(np.array([1, 9, 8, 7, 6.0]), bugs, loc) == pytest.approx(0.0)   # buggy file read last


def test_pofb20_respects_effort_budget():
    # 20% of 1000 lines = 200: only the first file (200 lines) fits
    loc = np.array([200, 400, 400])
    bugs = np.array([1, 3, 0])
    assert pofb20(np.array([3, 2, 1.0]), bugs, loc) == pytest.approx(1 / 4)


def test_pofb20_no_bugs_is_nan():
    assert np.isnan(pofb20(np.array([1.0, 2.0]), np.array([0, 0]), np.array([10, 10])))


def test_pofb20_per_loc_prefers_small_files():
    loc = np.array([1000, 10, 10])
    bugs = np.array([1, 1, 0])
    score = np.array([0.9, 0.8, 0.1])
    assert pofb20_per_loc(score, bugs, loc) >= pofb20(score, bugs, loc)


def test_score_all_keys_and_ranges():
    y = np.array([0, 1, 0, 1]); proba = np.array([0.1, 0.8, 0.3, 0.6])
    s = score_all(y, proba, np.array([0, 2, 0, 1]), np.array([10, 10, 10, 10]))
    assert set(s) == {"f1", "auc", "pofb20", "pofb20_per_loc"}
    assert s["auc"] == pytest.approx(1.0) and s["f1"] == pytest.approx(1.0)
