import numpy as np
import pandas as pd
import pytest
from sdp.config import FEATURES, PAIRS
from sdp.data import list_releases, load_release, to_binary


def test_label_is_bug_greater_than_zero():
    assert to_binary([0, 1, 2, 47]).tolist() == [0, 1, 1, 1]


def test_all_twelve_releases_load_without_missing_values():
    names = list_releases()
    assert len(names) == 12
    for n in names:
        r = load_release(n)
        assert r.X.shape[1] == len(FEATURES) == 20
        assert not np.isnan(r.X).any()
        assert set(np.unique(r.y)) <= {0, 1}


def test_class_counts_match_phase1_analysis():
    # 4,679 classes and 1,681 buggy over the twelve releases (Phase 1 table)
    rel = [load_release(n) for n in list_releases()]
    assert sum(len(r.y) for r in rel) == 4679
    assert sum(r.y.sum() for r in rel) == 1681


def test_every_pair_has_both_releases():
    names = set(list_releases())
    for old, new in PAIRS:
        assert old in names and new in names


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_release("nope", tmp_path)


def test_missing_column_is_detected(tmp_path):
    pd.DataFrame({"wmc": [1], "bug": [0]}).to_csv(tmp_path / "x.csv", index=False)
    with pytest.raises(ValueError):
        load_release("x", tmp_path)
