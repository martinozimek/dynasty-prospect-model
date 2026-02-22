"""
dynaff — shared utilities for dynasty-prospect-model.

The primary entry point for building and fitting models is the scripts/:
    scripts/build_training_set.py  — joins cfb + nfl data into training CSVs
    scripts/fit_model.py           — trains position models, runs cross-validation
    scripts/score_class.py         — applies fitted model to a draft class

Research workflow (EDA before fitting):
    notebooks/01_feature_correlations.ipynb  — B2S correlation by position
    notebooks/02_baseline_regression.ipynb   — OLS baseline, feature selection
    notebooks/03_gradient_boosting.ipynb     — LightGBM with monotonic constraints
"""
