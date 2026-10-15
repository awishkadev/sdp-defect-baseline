import numpy as np
import pytest
from sdp.models import log1p_transform, make_model, undersample


def test_undersample_balances_and_keeps_all_minority():
    rng = np.random.default_rng(0)
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array([1] * 5 + [0] * 15)
    Xs, ys = undersample(X, y, rng)
    assert ys.sum() == 5 and (ys == 0).sum() == 5
    minority = {tuple(r) for r in X[y == 1]}
    assert minority <= {tuple(r) for r in Xs}


def test_undersample_does_not_modify_input():
    X = np.random.default_rng(1).random((10, 3)); y = np.array([1] * 3 + [0] * 7)
    X0, y0 = X.copy(), y.copy()
    undersample(X, y, np.random.default_rng(2))
    assert np.array_equal(X, X0) and np.array_equal(y, y0)


def test_undersample_is_reproducible_for_same_seed():
    X = np.arange(60, dtype=float).reshape(30, 2); y = np.array([1] * 6 + [0] * 24)
    a = undersample(X, y, np.random.default_rng(7))[0]
    b = undersample(X, y, np.random.default_rng(7))[0]
    assert np.array_equal(a, b)


def test_log1p_handles_zero_and_is_monotone():
    out = log1p_transform(np.array([[0.0, 1.0, 100.0]]))
    assert out[0, 0] == 0 and out[0, 0] < out[0, 1] < out[0, 2]


def test_scaler_is_fitted_on_training_data_only():
    """Leakage test: changing the test data must not change what the pipeline learned."""
    rng = np.random.default_rng(3)
    Xtr = rng.integers(0, 50, (80, 4)).astype(float); ytr = rng.integers(0, 2, 80)
    m = make_model("LR", seed=1).fit(Xtr, ytr)
    scaler = m.named_steps["standardscaler"]
    expected_mean = log1p_transform(Xtr).mean(axis=0)
    assert np.allclose(scaler.mean_, expected_mean)
    before = scaler.mean_.copy()
    m.predict_proba(rng.integers(0, 5000, (30, 4)).astype(float))       # predicting must not refit
    assert np.array_equal(scaler.mean_, before)


@pytest.mark.parametrize("kind", ["LR", "LR-weighted", "RF", "RF-weighted"])
def test_models_output_probabilities(kind):
    rng = np.random.default_rng(4)
    X = rng.random((60, 5)); y = rng.integers(0, 2, 60)
    p = make_model(kind, seed=0).fit(X, y).predict_proba(X)
    assert p.shape == (60, 2) and np.allclose(p.sum(axis=1), 1)


def test_unknown_model_kind():
    with pytest.raises(ValueError):
        make_model("SVM", 0)


def test_same_seed_gives_same_random_forest():
    rng = np.random.default_rng(5)
    X = rng.random((80, 6)); y = rng.integers(0, 2, 80)
    a = make_model("RF", 11).fit(X, y).predict_proba(X)
    b = make_model("RF", 11).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b)
