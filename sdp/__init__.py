"""Baseline models for cross-version software defect prediction (PROMISE data).

Components (one module each):
    config      constants: file names, feature list, version pairs, paper numbers
    data        loading the csv tables and building the binary label
    metrics     F1, AUC, PofB20 and the all-buggy reference rule
    models      model factory (pipelines) and balanced undersampling
    tuning      hyperparameter search on the training release only
    experiments cross-version and within-version experiment loops
"""
__version__ = "1.0.0"
