"""Loading the PROMISE tables and building the binary target."""
from pathlib import Path
from typing import NamedTuple
import numpy as np
import pandas as pd
from .config import BUG_COLUMN, DATA_DIR, FEATURES, LOC_COLUMN


class Release(NamedTuple):
    """One release of one project."""
    name: str
    X: np.ndarray      # (n_classes, 20) metrics
    y: np.ndarray      # (n_classes,) 1 = buggy, 0 = clean
    bugs: np.ndarray   # (n_classes,) raw bug counts (needed by PofB20)
    loc: np.ndarray    # (n_classes,) lines of code (the "effort")


def to_binary(bugs):
    """Target variable: a class is buggy if at least one bug was recorded."""
    return (np.asarray(bugs, dtype=float) > 0).astype(int)


def load_release(name, data_dir: Path = DATA_DIR) -> Release:
    """Read <data_dir>/<name>.csv and return the metrics, label, bug counts and LOC."""
    path = Path(data_dir) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing data file: {path}")
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [BUG_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    X = df[FEATURES].astype(float).to_numpy()
    if np.isnan(X).any():
        raise ValueError(f"{path.name}: unexpected missing values")   # none exist in the 12 tables
    bugs = df[BUG_COLUMN].astype(float).to_numpy()
    return Release(name, X, to_binary(bugs), bugs, df[LOC_COLUMN].astype(float).to_numpy())


def list_releases(data_dir: Path = DATA_DIR):
    return sorted(p.stem for p in Path(data_dir).glob("*.csv"))
