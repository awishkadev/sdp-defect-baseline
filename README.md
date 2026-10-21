# Baseline models for software defect prediction (PROMISE data)

Phase 2 (Option A, baseline implementation) of the module *Machine Learning and Related Applications*.

The paper under review is Abdu et al. (2024), *Semantic and traditional feature fusion for software
defect prediction using hybrid deep learning model*, Scientific Reports 14:14771
(doi:10.1038/s41598-024-65639-4). This repository trains simple baselines, Logistic Regression and
Random Forest, on the 20 object-oriented metrics of the paper's twelve PROMISE tables, and compares
them with the CNN-MLP results printed in the paper.

**Task.** Supervised binary classification of a Java class: `y = 1` if the recorded bug count is above 0.
**Protocol.** Train on the older release of a project, test on the newer one
(Xalan 2.4→2.6, Poi 1.5→3.0, jEdit 4.0→4.1, Lucene 2.0→2.2, Synapse 1.1→1.2), 30 repeats, fixed seeds.
**Metrics.** F1 (threshold 0.5), AUC, PofB20.

## Layout

```
run_baseline.py        command line entry point (tuning, experiments, tables)
check_paper_tables.py  re-checks of the paper's Tables 5, 6 and 10 (Wilcoxon tests with Holm correction, F1 against the all-buggy rule)
run_paper_recheck.py   Phase 3: checks on the numbers printed in the paper (no model is trained)
sdp/config.py          constants: features, version pairs, seeds, paper numbers, tuning grids
sdp/data.py            loading the csv tables, binary label
sdp/metrics.py         F1, AUC, PofB20, all-buggy reference rule
sdp/models.py          model pipelines (log1p + scaler + LR, RF) and undersampling
sdp/tuning.py          hyperparameter search on the training release only
sdp/experiments.py     experiment loops (tuning, cross-version, within-release)
sdp/paper_tables.py    the paper's Tables 5, 6 and 10 and the checks used in Phase 3
tests/                 37 pytest tests
data/promise/          the 12 PROMISE tables used in the paper
results/               csv files written by run_baseline.py
notebooks/             notebook that calls the package and shows the results
```

## Setup and run

```
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest -v             # 37 tests
python run_baseline.py --quick  # short smoke run (default parameters, 2 repeats)
python run_baseline.py          # full run: tuning + 30 repeats (about 16 minutes on my laptop)
python check_paper_tables.py    # Phase 3: checks of the numbers printed in the paper (a few seconds)
python run_paper_recheck.py     # Phase 3: checks on the paper's tables (a few seconds)
```

Random seeds: repeat `r` uses seed `2026 + r` for undersampling, Random Forest and Logistic Regression,
so repeated runs give the same numbers (Random Forest can differ in the 6th decimal between machines).

## Data

The twelve csv files are the PROMISE defect tables (Jureczko and Madeyski, 2010) for the project versions
listed in the paper's Table 1. They were taken from a public backup of the PROMISE repository
(`github.com/hjx-zhans/promise-backup`). Columns: `name`, the 20 metrics, `bug`.

## Limits

Only the 20 traditional metrics are used (the paper's semantic AST features need the source code).
The paper does not state its train/test pairs, so the pairs above are my choice. Ant-1.7 and log4j-1.1
have only one release and are not part of the cross-version comparison.
