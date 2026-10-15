"""Constants shared by all modules. Nothing here runs code."""
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "promise"

# The 20 object-oriented metrics of the PROMISE tables (column names in lower case).
FEATURES = ["wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", "lcom3", "loc",
            "dam", "moa", "mfa", "cam", "ic", "cbm", "amc", "max_cc", "avg_cc"]
BUG_COLUMN = "bug"      # number of bugs found in the class; label is bug > 0
LOC_COLUMN = "loc"      # used as the "effort" in PofB20

# Older -> newer release of the same project. The paper does not state its pairs.
# Ant-1.7 and log4j-1.1 have only one release in the data, so they have no pair.
PAIRS = [("xalan-2.4", "xalan-2.6"), ("poi-1.5", "poi-3.0"), ("jedit-4.0", "jedit-4.1"),
         ("lucene-2.0", "lucene-2.2"), ("synapse-1.1", "synapse-1.2")]

N_REPEATS = 30          # the paper also repeats every experiment 30 times
WV_REPEATS = 10         # within-release check: repeats of 5-fold CV (kept small for speed)
BASE_SEED = 2026        # seed of repeat r is BASE_SEED + r

# CNN-MLP results printed in the paper: (F1, AUC, PofB20) from its Tables 5, 6 and 10.
# Assumption: a row is the result on the version it is named after.
PAPER = {
    "xalan-2.6": (0.706, 0.626, 0.626), "poi-3.0": (0.745, 0.616, 0.492),
    "jedit-4.1": (0.652, 0.582, 0.389), "lucene-2.2": (0.578, 0.596, 0.612),
    "synapse-1.2": (0.690, 0.683, 0.521),
    "ant-1.7": (0.719, 0.673, 0.406), "log4j-1.1": (0.845, 0.592, 0.415),
}

# Hyperparameter grids searched on the training release (see sdp/tuning.py).
GRID_LR = {"logisticregression__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
GRID_RF = {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 3, 5], "max_features": ["sqrt", 0.5]}
