"""
Train position-specific dynasty prospect models and persist them for scoring.

Workflow:
  1. Load training CSVs (from build_training_set.py)
  2. Engineer features (same as analyze.py for consistency)
  3. Lasso CV to select informative features per position
  4. Fit two models per position:
       - Ridge regression  (interpretable, stable with small N)
       - LightGBM          (captures non-linear interactions)
  5. Evaluate both with Leave-One-Year-Out CV (temporal holdout)
     OR rolling-window CV / k-fold CV (selectable via --cv-strategy)
  6. Save sklearn Pipelines + metadata to models/

Overfitting remediation (2026-03-13, see plan):
  - Training restricted to 2014+ draft classes (default; PFF coverage complete from 2014;
    pre-2014 rows use median-imputed PFF values creating era bias)
  - VIF-pruned candidate feature sets (Step 2 + 7a):
      WR: overall_pick removed (r=−0.873 with draft_tier)
      RB: overall_pick + capital_x_dominator removed (r=0.946/0.928 with capital_x_age)
      TE: combined_ath_x_capital removed (r=0.981 with capital_x_age);
          draft_premium removed (r=−0.953 with position_rank)
  - Nested Lasso CV: feature selection re-run per fold (honest LOYO, Step 3)
  - Per-position RidgeCV alpha tuning (Step 4 + 7b):
      WR/RB: 0.1–100; TE: 1–1000 (wider for small N=97)
  - Per-position LightGBM hyperparameter overrides (Step 5 + 7d)
  - TE LightGBM: retired if LOYO gap > 0.10 vs Ridge (Step 0 + 7d)
  - Rolling-window + k-fold CV computed for TE (Step 7c)

Output files:
  models/{POS}_ridge.pkl      — sklearn Pipeline: SimpleImputer → StandardScaler → RidgeCV
  models/{POS}_lgbm.pkl       — sklearn Pipeline: SimpleImputer → LGBMRegressor
  models/{POS}_features.json  — ordered feature list used by the pipeline
  models/metadata.json        — CV scores, feature counts, training details

Usage:
    python scripts/fit_model.py --all
    python scripts/fit_model.py --position WR
    python scripts/fit_model.py --position WR --no-lgbm      # Ridge only
    python scripts/fit_model.py --all --start-year 2011      # override 2014 default
    python scripts/fit_model.py --all --no-capital           # Phase I: no-capital model
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # for analyze.py imports

from analyze import engineer_features, load_training, QuantileClipper

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# QuantileClipper is imported from analyze.py so that both fit_model.py and
# score_class.py resolve the same class when loading pickled Phase I models.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-position candidate feature sets (VIF-pruned 2026-03-13, Step 2 + 7a)
# ---------------------------------------------------------------------------

_CANDIDATE_FEATURES = {
    "WR": [
        # Draft capital (capital model only — excluded from nocap candidate pool)
        # overall_pick REMOVED — r=−0.873 with draft_tier (VIF Step 2)
        "log_draft_capital", "draft_round",
        "draft_premium",
        # Draft tier (ep 1086 — capital works in tiers, not precise rankings)
        "draft_tier",
        # Pre-draft market consensus
        "consensus_rank", "position_rank",
        # Capital interaction terms
        "capital_x_age", "capital_x_dominator", "rec_rate_x_capital",
        "breakout_score_x_capital", "breakout_score_x_dominator",
        "college_fpg_x_capital",
        # College production — rates
        "best_rec_rate", "log_rec_rate", "best_dominator", "log_dominator",
        "best_reception_share", "best_ppa_pass",
        "best_usage_pass",
        # Breakout score
        "best_breakout_score",
        # College production — counting / other
        "college_fantasy_ppg", "best_rec_yards", "best_receptions",
        "career_rec_yards", "career_receptions", "career_yardage",
        # Age / breakout
        "best_age",
        "age_at_draft",             # JJ's "under-22 on draft day" threshold
        "early_declare",            # JJ includes this; captures draft commitment beyond just age; let Lasso decide
        # Athleticism — size only; pure speed excluded per JJ & Barrett
        "weight_lbs",
        # Team context
        "teammate_score",
        # Conference tier (JJ conference factor)
        "power4_conf",
        # TD share (JJ's sub-20% ceiling flag)
        "best_rec_td_pct",
        # Recruiting
        "recruit_rating",
        # PFF metrics
        "best_yprr", "best_receiving_grade", "best_routes_per_game",
        "best_drop_rate", "best_target_sep",
        "yprr_x_capital",
        # New Phase II capital interactions (Part 3)
        "pff_grade_x_capital",   # best_receiving_grade × log_draft_capital
        "usage_x_capital",       # best_usage_pass × log_draft_capital
        "ppa_x_capital",         # best_ppa_pass × log_draft_capital
        # PFF split metrics — depth / scheme / concept zones
        "best_deep_yprr", "best_deep_target_rate", "deep_target_x_capital", "deep_yprr_x_capital",
        "best_slot_yprr", "best_slot_target_rate", "slot_rate_x_capital",
        "best_man_yprr", "best_zone_yprr", "best_man_zone_delta", "man_delta_x_capital",
        "best_screen_rate", "best_behind_los_rate",
    ],
    "RB": [
        # Draft capital
        # overall_pick REMOVED — redundant with log_draft_capital + draft_tier (VIF Step 2)
        # capital_x_dominator REMOVED — r=0.928 with capital_x_age (VIF Step 2)
        # draft_capital_score REMOVED — r=0.875 with log_draft_capital (2nd-round VIF)
        "log_draft_capital", "draft_round",
        # Draft tier (ep 1086 — top-half R1 RBs R²=0.526)
        "draft_tier", "is_top16_rb",
        # Capital interactions
        "capital_x_age",
        "total_yards_rate_x_capital",
        # Phase C new features (all cleared ≥+0.005 LOYO threshold)
        "log_pick_capital",           # JJ-style curve +0.068
        "pick_capital_x_age",
        "total_yards_rate_x_pick_capital",
        "teammate_score_x_capital",   # interaction +0.057
        "best_breakout_score",        # +0.055
        "blended_capital_rb",         # 83/17 actual+consensus blend
        "log_blended_capital",
        # Phase D: total plays denominator variant
        "best_total_yards_rate_v2",
        # Pre-draft market
        "consensus_rank", "position_rank",
        # College production — total contribution (JJ's adjusted yards per team play)
        "best_total_yards_rate",
        # College production — receiving focused
        "best_rec_rate", "best_reception_share",
        "best_usage_pass", "college_fantasy_ppg",
        "best_rush_ypc", "best_yards_per_touch",
        "career_rec_yards", "career_yardage",
        # Part 3 additions
        "career_rush_attempts",     # career workload signal
        "best_man_yprr",            # man coverage receiving
        "best_behind_los_rate",     # LOS pass involvement
        "best_deep_yprr",           # WR-route usage
        "height_inches",
        "forty_time",
        # Age
        "best_age",
        "age_at_draft",             # JJ's "under-22 on draft day" threshold
        "early_declare",            # early declarers tend to be top RB prospects; let Lasso decide
        # Athleticism
        "weight_lbs", "speed_score", "agility_score", "combined_ath",
        "vertical_jump", "broad_jump",
        # Team context
        "teammate_score",
        # Conference tier
        "power4_conf",
        # Recruiting
        "recruit_rating",
        # PFF split metrics
        "best_screen_rate", "best_zone_yprr", "best_man_zone_delta",
    ],
    "TE": [
        # Draft capital (dominant signal for TEs)
        # draft_round REMOVED — r=0.983 with overall_pick (2nd-round VIF)
        # position_rank REMOVED — r=0.942 with consensus_rank (2nd-round VIF)
        "log_draft_capital", "draft_capital_score", "overall_pick",
        # draft_premium REMOVED — r=−0.953 with position_rank (VIF Step 7a)
        # Draft tier
        "draft_tier",
        # Capital interactions
        # combined_ath_x_capital REMOVED — r=0.981 with capital_x_age (VIF Step 7a)
        "capital_x_age", "capital_x_dominator",
        "breakout_score_x_capital", "breakout_score_x_dominator",
        # Pre-draft market (position_rank removed above — r=0.942 with consensus_rank)
        "consensus_rank",
        # College production
        "best_rec_rate", "best_dominator", "best_reception_share",
        "career_rec_per_target", "college_fantasy_ppg",
        "best_ppa_pass",
        # Part 3 additions
        "career_rec_yards",
        "best_catch_rate",
        "best_contested_catch_rate",
        "best_behind_los_rate",
        "best_screen_rate",
        # Breakout score
        "best_breakout_score",
        # Age / breakout — age encoded in breakout_score; early_declare not included for TE
        "best_age",
        "age_at_draft",             # JJ's "under-22 on draft day" threshold
        # Athleticism — important for TEs (necessary but not sufficient)
        "weight_lbs", "speed_score", "combined_ath", "agility_score",
        "forty_time", "vertical_jump", "broad_jump",
        # Team context
        "teammate_score",
        # Conference tier
        "power4_conf",
        # TD share
        "best_rec_td_pct",
        # Recruiting
        "recruit_rating",
        # PFF metrics
        "best_yprr", "best_receiving_grade", "best_routes_per_game",
        "best_target_sep",
        "yprr_x_capital",
        # New Phase II capital interactions (Part 3)
        "ppa_x_capital",        # best_ppa_pass × log_draft_capital
        "rec_td_x_capital",     # best_rec_td_pct × log_draft_capital
        # PFF split metrics
        "best_deep_yprr", "best_deep_target_rate", "deep_target_x_capital", "deep_yprr_x_capital",
        "best_slot_yprr", "best_slot_target_rate", "slot_rate_x_capital",
        "best_man_yprr", "best_zone_yprr", "best_man_zone_delta", "man_delta_x_capital",
    ],
}

# ---------------------------------------------------------------------------
# Phase I (no-capital) candidate feature sets
# All capital, market, and interaction features excluded.
# Phase I score answers: "where does this player rank on pure college/combine evidence?"
# ---------------------------------------------------------------------------
_CANDIDATE_FEATURES_NOCAP = {
    "WR": [
        # --- Primary production ---
        "best_breakout_score",      # yprr(PFF) × SOS × age(T) — age encoded here; not separately
        "breakout_tier",            # ordinal 0/1/2 — captures nonlinear step in breakout_score
        "best_rec_rate",
        "log_rec_rate",             # log transform (right-skewed distribution)
        "best_dominator",
        # --- Target opportunity ---
        "best_usage_pass",          # target rate relative to pass attempts
        "best_reception_share",     # % of team receptions
        "best_rec_td_pct",          # JJ's sub-20% TD ceiling flag
        "best_catch_rate",          # target completion efficiency
        # --- Quality signals ---
        "best_ppa_pass",            # EPA per pass play (orthogonal to rec_rate)
        # --- Career volume ---
        "college_fantasy_ppg",
        "career_rec_yards",
        "career_receptions",
        "career_yardage",           # rec + rush total
        # NOTE: best_age / age_at_draft removed — age is encoded in best_breakout_score
        # --- PFF base metrics (season-locked after Part 0 fix) ---
        "best_yprr",
        "best_receiving_grade",
        "best_routes_per_game",
        "best_drop_rate",
        "best_target_sep",
        # --- PFF split metrics (independent peaks — correct as-is) ---
        "best_man_yprr",
        "best_zone_yprr",
        "best_man_zone_delta",
        "best_slot_yprr",
        "best_slot_target_rate",
        "best_deep_yprr",
        "best_deep_target_rate",
        # --- Context ---
        "power4_conf",
        "teammate_score",
        "recruit_rating",
        # --- Athleticism (speed_score excluded per JJ for WRs) ---
        "forty_time",
        "broad_jump",
        "vertical_jump",
        "agility_score",
        "weight_lbs",
        "height_inches",
        # --- Phase I interaction terms ---
        "breakout_score_x_yprr",    # production AND efficiency agree → highest-conviction Phase I signal
        "rec_rate_x_routes",        # volume-adjusted efficiency (rec_rate × routes/game)
        "best_yprr_x_routes",       # YPRR quality anchored by route quantity
        "grade_x_routes",           # PFF grade confirmed by usage
        "usage_x_yprr",             # target opportunity × route efficiency
        "recruit_x_breakout",       # recruiting ceiling × production floor
    ],
    "RB": [
        # --- Primary production ---
        "best_total_yards_rate",    # (rec+rush)/team_pass_att × SOS × age
        "best_breakout_score",      # yprr(PFF) × SOS × age(T) — pass-catching signal
        "breakout_tier",            # ordinal 0/1/2 — nonlinear tier
        "best_rec_rate",
        "best_dominator",
        # --- Receiving role ---
        "best_usage_pass",
        "best_reception_share",
        "best_rush_ypc",
        "best_yards_per_touch",     # combined rec+rush efficiency
        "college_fantasy_ppg",
        # --- Career volume ---
        "career_rec_yards",
        "career_yardage",
        "career_rush_attempts",     # career workload signal
        # NOTE: best_age / age_at_draft removed — age is encoded in best_breakout_score
        # --- PFF base metrics ---
        "best_yprr",
        "best_receiving_grade",
        "best_routes_per_game",
        # --- PFF split metrics ---
        "best_zone_yprr",
        "best_man_yprr",
        "best_behind_los_rate",     # LOS pass involvement
        "best_deep_yprr",           # WR-route usage
        # --- Context ---
        "power4_conf",
        "recruit_rating",
        "teammate_score",
        # --- Athleticism ---
        "speed_score",
        "forty_time",
        "broad_jump",
        "vertical_jump",
        "agility_score",
        "weight_lbs",
        "height_inches",
        # --- Phase I interaction terms ---
        "total_yards_x_youth",      # mirrors breakout_score structure for RBs
        "total_yards_rate_x_yprr",  # total yards production × pass-catching efficiency
        "recruit_x_breakout",       # recruiting ceiling × production floor
    ],
    "TE": [
        # --- Primary production ---
        "best_breakout_score",
        "breakout_tier",            # ordinal 0/1/2 — nonlinear tier (median split)
        "breakout_elite",           # binary top-25% flag — Phase A: bs_above_2_5 R²=0.235 for TEs
        "best_rec_rate",
        "best_dominator",
        "best_reception_share",
        # --- Career efficiency ---
        "career_rec_per_target",    # yards per target over career
        "career_rec_yards",
        "college_fantasy_ppg",
        # --- Quality signals ---
        "best_ppa_pass",            # EPA per pass play — strong TE signal
        "best_rec_td_pct",
        "best_catch_rate",
        # NOTE: best_age / age_at_draft removed — age is encoded in best_breakout_score
        # --- Context ---
        "power4_conf",
        "recruit_rating",
        "teammate_score",
        # --- PFF base metrics ---
        "best_yprr",
        "best_receiving_grade",
        "best_routes_per_game",
        "best_drop_rate",
        "best_target_sep",
        "best_contested_catch_rate",
        # --- PFF split metrics ---
        "best_man_yprr",
        "best_zone_yprr",
        "best_man_zone_delta",
        "best_slot_yprr",
        "best_slot_target_rate",
        "best_deep_yprr",
        "best_deep_target_rate",
        "best_behind_los_rate",
        "best_screen_rate",
        # --- Athleticism (important for TEs per JJ & Barrett) ---
        "speed_score",
        "forty_time",
        "broad_jump",
        "vertical_jump",
        "agility_score",
        "weight_lbs",
        "height_inches",
        # --- Phase I interaction terms ---
        "breakout_score_x_grade",   # production quality × PFF grade (grade captures separation)
        "dominator_x_yprr",         # receiving dominance × route efficiency
        "breakout_x_ppa",           # production × scheme quality
        "recruit_x_breakout",       # recruiting ceiling × production floor
    ],
}

# ---------------------------------------------------------------------------
# Capital-only feature sets (Part 1 — baseline: how much does raw draft capital predict?)
# Used with --capital-only flag OR as a secondary diagnostic pass during standard runs.
# If full ORBIT LOYO ≤ capital-only LOYO + 0.020, the model fails to beat its baseline.
# ---------------------------------------------------------------------------
_CAPITAL_FEATURES_ONLY = {
    "WR": [
        "log_draft_capital", "draft_round", "draft_tier", "draft_premium",
        "consensus_rank", "position_rank",
    ],
    "RB": [
        "log_draft_capital", "draft_round", "draft_tier",
        "log_pick_capital", "blended_capital_rb", "log_blended_capital",
        "consensus_rank", "position_rank", "is_top16_rb",
    ],
    "TE": [
        "log_draft_capital", "draft_capital_score", "overall_pick",
        "draft_tier", "consensus_rank",
    ],
}

# ---------------------------------------------------------------------------
# Max features cap per position (Part 2 — overfitting guard)
# Hard cap applied after Lasso selection, keeping top-N by |coef|.
# TE=7 is critical: 97 rows / 10 features = 0.103 features-per-obs (red flag).
# ---------------------------------------------------------------------------
_MAX_FEATURES = {
    "WR": 9,    # 224 rows — cap prevents selection of marginal split metrics
    "RB": 10,   # 147 rows
    "TE": 7,    # 97 rows — critical; current 10 features is too many
}

# FAIL threshold: full ORBIT LOYO must beat capital-only LOYO by at least this margin
_CAPITAL_DELTA_THRESHOLD = 0.020

# ---------------------------------------------------------------------------
# LGBM monotone constraints
# ---------------------------------------------------------------------------
_MONOTONE_DIRECTIONS = {
    "log_draft_capital": 1,   "draft_capital_score": 1,
    "draft_premium": 1,       "capital_x_age": 1,
    "capital_x_dominator": 1, "rec_rate_x_capital": 1,
    "college_fpg_x_capital": 1,
    "best_breakout_score": 1,
    "breakout_score_x_capital": 1, "breakout_score_x_dominator": 1,
    "best_total_yards_rate": 1,    "total_yards_rate_x_capital": 1,
    "best_total_yards_rate_v2": 1,
    "log_pick_capital": 1,         "pick_capital_x_age": 1,
    "total_yards_rate_x_pick_capital": 1,
    "teammate_score_x_capital": 1,
    "blended_capital_rb": 1,       "log_blended_capital": 1,
    "combined_ath_x_capital": 1,
    "overall_pick": -1,       "draft_round": -1,
    "consensus_rank": -1,     "position_rank": -1,
    "best_rec_rate": 1,       "log_rec_rate": 1,
    "best_dominator": 1,      "log_dominator": 1,
    "best_reception_share": 1, "college_fantasy_ppg": 1,
    "best_ppa_pass": 1,        "best_usage_pass": 1,
    "career_rec_yards": 1,     "career_receptions": 1,
    "career_yardage": 1,       "career_rec_per_target": 1,
    "best_rec_yards": 1,       "best_receptions": 1,
    "best_rush_ypc": 1,        "best_yards_per_touch": 1,
    "speed_score": 1,          "combined_ath": 1,
    "vertical_jump": 1,        "broad_jump": 1,
    "agility_score": -1,       "forty_time": -1,
    "recruit_rating": 1,
    "teammate_score": 1,
    "power4_conf": 1,
    "best_rec_td_pct": 1,
    "draft_tier": 1,
    "is_top16_rb": 1,
    "best_yprr": 1,          "yprr_x_capital": 1,
    "best_receiving_grade": 1, "best_routes_per_game": 1,
    "best_contested_catch_rate": 1,
    "best_drop_rate": -1,
    "best_target_sep": 1,
    "best_deep_yprr": 1,        "deep_yprr_x_capital": 1,
    "best_deep_target_rate": 1, "deep_target_x_capital": 1,
    "best_slot_yprr": 1,        "slot_rate_x_capital": 1,
    "best_slot_target_rate": 1,
    "best_man_yprr": 1,         "best_zone_yprr": 1,
    "best_screen_rate": -1,     "best_behind_los_rate": -1,
    # Age at draft day (younger = better → direction -1)
    "age_at_draft": -1,
    # Early declare (1 = ≤3 years of college; more talent → direction +1)
    "early_declare": 1,
    # Phase I no-capital interaction terms (higher = better)
    "breakout_score_x_yprr": 1, "rec_rate_x_routes": 1,
    "total_yards_x_youth": 1,   "breakout_score_x_grade": 1,
    # Phase I expanded interactions (Part 2 additions)
    "best_yprr_x_routes":       1,
    "grade_x_routes":           1,
    "usage_x_yprr":             1,
    "dominator_x_yprr":         1,
    "total_yards_rate_x_yprr":  1,
    "recruit_x_breakout":       1,
    "breakout_x_ppa":           1,
    # Phase II capital interaction terms (Part 4 additions)
    "pff_grade_x_capital":      1,
    "usage_x_capital":          1,
    "ppa_x_capital":            1,
    "rec_td_x_capital":         1,
}

# ---------------------------------------------------------------------------
# Per-position Ridge alpha search ranges (Steps 4 + 7b)
# WR/RB: 0.1–100 (standard N); TE: 1–1000 (wider for small N=97)
# ---------------------------------------------------------------------------
_RIDGE_ALPHA_RANGE = {
    "WR": np.logspace(-1, 2, 50),    # 0.1 – 100
    "RB": np.logspace(-1, 2, 50),    # 0.1 – 100
    "TE": np.logspace(0, 3, 50),     # 1 – 1000 (small N → prefer more shrinkage)
}

# ---------------------------------------------------------------------------
# Per-position LightGBM hyperparameter overrides (Steps 5 + 7d)
# WR: aggressive regularization to close LOYO gap=0.094
# RB: minor adjustment (gap=0.029 acceptable)
# TE: aggressive regularization for retune attempt (N=97); retired if gap > 0.10
# ---------------------------------------------------------------------------
_LGBM_PARAMS_OVERRIDES = {
    "WR": {
        "n_estimators": 150, "max_depth": 3, "min_child_samples": 10,
        "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
        "min_split_gain": 0.0,
    },
    "RB": {
        "n_estimators": 200, "max_depth": 4, "min_child_samples": 8,
        "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
        "min_split_gain": 0.0,
    },
    "TE": {
        # Step 7d: aggressive regularization retune attempt
        # Formally retired if LGBM LOYO R² < Ridge LOYO R² − 0.10
        "n_estimators": 100, "max_depth": 2, "min_child_samples": 15,
        "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7,
        "min_split_gain": 0.1,
    },
}


# ---------------------------------------------------------------------------
# CV helpers
# ---------------------------------------------------------------------------

def _loyo_cv(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    model_type: str,  # "ridge" or "lgbm"
    lgbm_params: dict | None = None,
    alpha_range: np.ndarray | None = None,       # Step 4/7b: per-position Ridge alpha range
    feature_candidates: list[str] | None = None, # Step 3: nested Lasso per fold
    clip_outliers: bool = False,                 # B3: add QuantileClipper to ridge pipeline
    max_features: int | None = None,             # Part 2: overfitting cap applied inside nested Lasso
) -> dict:
    """
    Leave-One-Year-Out cross-validation.

    If feature_candidates is provided (Ridge only): Lasso selection is re-run on
    each fold's training data (nested CV — removes selection leakage from LOYO R²).
    If alpha_range is provided: RidgeCV selects alpha on each fold's training data.
    """
    years = sorted(df["draft_year"].dropna().unique())
    year_results = []
    all_actual, all_predicted = [], []

    for test_year in years:
        train_df = df[df["draft_year"] != test_year]
        test_df  = df[df["draft_year"] == test_year]

        train_sub = train_df[train_df[target].notna()].copy()
        test_sub  = test_df[test_df[target].notna()].copy()

        if len(train_sub) < 10 or len(test_sub) < 2:
            continue

        # Nested Lasso: re-select features on this fold's training data (Step 3)
        if feature_candidates is not None and model_type == "ridge":
            fold_avail = [
                f for f in feature_candidates
                if f in train_sub.columns
                and train_sub[f].notna().sum() >= 5
                and train_sub[f].nunique(dropna=True) > 1
            ]
            fold_features = _lasso_select(
                train_sub[fold_avail], train_sub[target], cv_folds=3,
                max_features=max_features,  # Part 2: overfitting cap
            )
            if not fold_features:
                fold_features = fold_avail
        else:
            fold_features = features

        used = [f for f in fold_features if f in train_sub.columns]
        X_train = train_sub[used].values
        y_train = train_sub[target].values
        X_test  = test_sub[used].values
        y_test  = test_sub[target].values

        if model_type == "ridge":
            _steps = [("imp", SimpleImputer(strategy="median"))]
            if clip_outliers:
                _steps.append(("clip99", QuantileClipper(upper_quantile=0.99)))
            _steps.append(("scl", StandardScaler()))
            if alpha_range is not None:
                _steps.append(("mdl", RidgeCV(alphas=alpha_range, cv=5)))
            else:
                _steps.append(("mdl", Ridge(alpha=10.0)))
            pipe = Pipeline(_steps)
        else:
            mc = [_MONOTONE_DIRECTIONS.get(f, 0) for f in used]
            params = lgbm_params or {}
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("mdl", LGBMRegressor(
                    n_estimators=params.get("n_estimators", 300),
                    learning_rate=params.get("learning_rate", 0.05),
                    max_depth=params.get("max_depth", 4),
                    min_child_samples=params.get("min_child_samples", 5),
                    subsample=params.get("subsample", 0.8),
                    colsample_bytree=params.get("colsample_bytree", 0.8),
                    min_split_gain=params.get("min_split_gain", 0.0),
                    monotone_constraints=mc,
                    random_state=42,
                    verbose=-1,
                )),
            ])

        pipe.fit(X_train, y_train)
        preds = np.clip(pipe.predict(X_test), 0, None)

        mae  = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
        ss_res = np.sum((y_test - preds) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        year_results.append({
            "year": int(test_year), "n_test": len(y_test),
            "r2": r2, "mae": mae, "rmse": rmse,
        })
        all_actual.extend(y_test.tolist())
        all_predicted.extend(preds.tolist())

    if not all_actual:
        return {"year_results": year_results, "r2": float("nan"),
                "mae": float("nan"), "rmse": float("nan"),
                "abs_residuals": [], "spearman_rho": float("nan"),
                "top25_hit_rate": float("nan")}

    all_actual    = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    agg_r2   = float(r2_score(all_actual, all_predicted))
    agg_mae  = float(mean_absolute_error(all_actual, all_predicted))
    agg_rmse = float(np.sqrt(np.mean((all_actual - all_predicted) ** 2)))

    # D1: conformal interval residuals
    abs_residuals = np.abs(all_actual - all_predicted).tolist()

    # D3: rank-order calibration metrics
    rho, _ = spearmanr(all_predicted, all_actual)
    q75_actual = np.quantile(all_actual, 0.75)
    q75_pred   = np.quantile(all_predicted, 0.75)
    top25_mask = all_actual >= q75_actual
    top25_hit_rate = float(np.mean(all_predicted[top25_mask] >= q75_pred)) if top25_mask.any() else float("nan")

    return {
        "year_results": year_results,
        "r2": agg_r2, "mae": agg_mae, "rmse": agg_rmse,
        "abs_residuals": [round(float(v), 4) for v in abs_residuals],
        "spearman_rho": round(float(rho), 4),
        "top25_hit_rate": round(top25_hit_rate, 4),
    }


def _rolling_cv(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    train_end: int = 2018,   # Train on [start]–train_end
    test_start: int = 2019,  # Test on test_start–[end]
    alpha_range: np.ndarray | None = None,
) -> dict:
    """
    Rolling-window (forward-looking) CV for TE evaluation (Step 7c).
    Most realistic simulation of predicting future draft classes.
    """
    train_df = df[df["draft_year"] <= train_end]
    test_df  = df[df["draft_year"] >= test_start]

    train_sub = train_df[train_df[target].notna()].copy()
    test_sub  = test_df[test_df[target].notna()].copy()

    if len(train_sub) < 10 or len(test_sub) < 2:
        return {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan"),
                "n_train": len(train_sub), "n_test": len(test_sub), "year_results": []}

    used = [f for f in features if f in train_sub.columns]
    X_train = train_sub[used].values
    y_train = train_sub[target].values
    X_test  = test_sub[used].values
    y_test  = test_sub[target].values

    if alpha_range is not None:
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", RidgeCV(alphas=alpha_range, cv=5)),
        ])
    else:
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("mdl", Ridge(alpha=10.0)),
        ])

    pipe.fit(X_train, y_train)
    preds = np.clip(pipe.predict(X_test), 0, None)

    ss_res = float(np.sum((y_test - preds) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mae  = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))

    # Per-year breakdown of test period
    year_results = []
    test_sub = test_sub.copy()
    test_sub["_pred"] = preds
    for yr in sorted(test_sub["draft_year"].unique()):
        sub_yr = test_sub[test_sub["draft_year"] == yr]
        y_yr = sub_yr[target].values
        p_yr = sub_yr["_pred"].values
        ss_r = np.sum((y_yr - p_yr) ** 2)
        ss_t = np.sum((y_yr - y_yr.mean()) ** 2)
        r2_yr = float(1 - ss_r / ss_t) if ss_t > 0 else float("nan")
        year_results.append({
            "year": int(yr), "n_test": len(y_yr),
            "r2": r2_yr, "mae": float(mean_absolute_error(y_yr, p_yr)),
        })

    return {
        "r2": r2, "mae": mae, "rmse": rmse,
        "n_train": len(train_sub), "n_test": len(test_sub),
        "year_results": year_results,
    }


def _kfold_cv(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    n_splits: int = 5,
    alpha_range: np.ndarray | None = None,
) -> dict:
    """
    K-fold CV (Step 7c alternative for TE).
    Better stability than LOYO when some holdout years have <10 obs.
    """
    df_labeled = df[df[target].notna()].copy().reset_index(drop=True)
    if len(df_labeled) < n_splits * 2:
        return {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan"),
                "n_folds": 0, "fold_results": []}

    kf   = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    used = [f for f in features if f in df_labeled.columns]
    X_all = df_labeled[used].values
    y_all = df_labeled[target].values

    fold_results = []
    all_actual, all_predicted = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        if alpha_range is not None:
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
                ("mdl", RidgeCV(alphas=alpha_range, cv=5)),
            ])
        else:
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
                ("mdl", Ridge(alpha=10.0)),
            ])

        pipe.fit(X_train, y_train)
        preds = np.clip(pipe.predict(X_test), 0, None)

        ss_res = np.sum((y_test - preds) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        fold_results.append({
            "fold": fold_idx + 1, "n_test": len(y_test),
            "r2": r2, "mae": float(mean_absolute_error(y_test, preds)),
        })
        all_actual.extend(y_test.tolist())
        all_predicted.extend(preds.tolist())

    all_actual    = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    agg_r2   = float(r2_score(all_actual, all_predicted))
    agg_mae  = float(mean_absolute_error(all_actual, all_predicted))
    agg_rmse = float(np.sqrt(np.mean((all_actual - all_predicted) ** 2)))

    return {
        "r2": agg_r2, "mae": agg_mae, "rmse": agg_rmse,
        "n_folds": n_splits, "fold_results": fold_results,
    }


def _lasso_select(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    max_features: int | None = None,
) -> list[str]:
    """
    Fit LassoCV on median-imputed, standardized X.
    Returns list of features with non-zero coefficients.
    If max_features is set, trims to top-N by |coef| (Part 2 overfitting guard).
    """
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_imp = imputer.fit_transform(X)
    X_std = scaler.fit_transform(X_imp)

    alphas = np.logspace(-4, 2, 60)
    lasso  = LassoCV(alphas=alphas, cv=cv_folds, max_iter=10_000, random_state=42)
    try:
        lasso.fit(X_std, y.values)
    except Exception as e:
        log.warning("LassoCV failed: %s — using all features", e)
        return list(X.columns)

    selected = [f for f, c in zip(X.columns, lasso.coef_) if c != 0.0]
    log.info("  Lasso α=%.4f  selected %d / %d features", lasso.alpha_,
             len(selected), X.shape[1])

    # Part 2: max_features cap — keep top-N by absolute Lasso coefficient
    if max_features is not None and len(selected) > max_features:
        coefs = {f: abs(c) for f, c in zip(X.columns, lasso.coef_) if c != 0.0}
        selected = sorted(selected, key=lambda f: coefs.get(f, 0.0), reverse=True)[:max_features]
        log.info("  max_features cap: trimmed to %d features", len(selected))

    return selected if selected else list(X.columns)


def fit_position(
    position: str,
    df_all: pd.DataFrame,
    target: str,
    models_dir: Path,
    fit_lgbm: bool = True,
    cv_strategy: str = "loyo",  # Step 7c: loyo | rolling | kfold
    feature_candidates: list[str] | None = None,   # Phase I or capital-only: override candidate pool
    model_suffix: str = "",                         # Phase I: "_nocap" appended to artifact names
    clip_outliers: bool = False,                    # B3: QuantileClipper in Phase I pipeline
    max_features: int | None = None,               # Part 2: overfitting cap (None = no cap)
) -> dict:
    """
    Full training pipeline for one position.
    Returns metadata dict with CV scores.
    """
    label = f"{position}{model_suffix}"
    log.info("=== %s (n=%d) ===", label, len(df_all))
    df = df_all[df_all[target].notna()].copy()
    log.info("  %d labeled rows (have %s)", len(df), target)

    candidate_cols = feature_candidates if feature_candidates is not None else _CANDIDATE_FEATURES[position]
    available = [c for c in candidate_cols if c in df.columns]
    missing_cands = [c for c in candidate_cols if c not in df.columns]
    if missing_cands:
        log.debug("  Candidate features not in data: %s", missing_cands)

    # Drop zero-variance columns
    non_const = [c for c in available if df[c].nunique(dropna=True) > 1]
    dropped_const = set(available) - set(non_const)
    if dropped_const:
        log.info("  Dropped zero-variance features: %s", sorted(dropped_const))
    available = non_const

    # Require at least 5 non-null values per feature
    available = [c for c in available if df[c].notna().sum() >= 5]

    X_full = df[available].copy()
    y_full = df[target]

    # --- Outer Lasso feature selection (used for scoring model + nested CV) ---
    log.info("  Running LassoCV on %d candidate features...", len(available))
    selected = _lasso_select(X_full, y_full, max_features=max_features)
    if not selected:
        selected = available
    log.info("  Selected features (%d): %s", len(selected), selected)

    # Per-position alpha range (Steps 4 + 7b)
    alpha_range = _RIDGE_ALPHA_RANGE.get(position)

    # --- Nested LOYO-CV for Ridge (Steps 3 + 4/7b) ---
    log.info("  LOYO-CV (Ridge, nested Lasso per fold, RidgeCV alpha)...")
    cv_ridge = _loyo_cv(
        df, selected, target, "ridge",
        alpha_range=alpha_range,
        # Step 3: nested Lasso uses the outer-selected features as candidates.
        # Using the full available pool (51 features) creates excessive fold-to-fold
        # variability (folds select 4–11 features) that inflates apparent LOYO drop.
        # Using outer-selected features is honest: removes ~0.002 selection bias
        # (confirmed in Section 6 diagnostics) without injecting noise.
        feature_candidates=selected,
        clip_outliers=clip_outliers,  # B3: Phase I only
        max_features=max_features,    # Part 2: overfitting cap inside nested folds
    )
    log.info(
        "  Ridge LOYO: R²=%.3f  MAE=%.2f  RMSE=%.2f  [nested_lasso+RidgeCV]",
        cv_ridge["r2"], cv_ridge["mae"], cv_ridge["rmse"],
    )

    # --- Rolling-window + K-fold CV (Step 7c — always computed for TE; opt-in others) ---
    rolling_cv_result: dict = {}
    kfold_cv_result: dict = {}
    if cv_strategy in ("rolling", "kfold") or position == "TE":
        log.info("  Rolling-window CV (train≤2018, test≥2019)...")
        rolling_cv_result = _rolling_cv(df, selected, target, alpha_range=alpha_range)
        log.info(
            "  Rolling: R²=%.3f  MAE=%.2f  n_train=%d  n_test=%d",
            rolling_cv_result.get("r2", float("nan")),
            rolling_cv_result.get("mae", float("nan")),
            rolling_cv_result.get("n_train", 0),
            rolling_cv_result.get("n_test", 0),
        )
        log.info("  K-fold CV (k=5)...")
        kfold_cv_result = _kfold_cv(df, selected, target, alpha_range=alpha_range)
        log.info(
            "  K-fold: R²=%.3f  MAE=%.2f",
            kfold_cv_result.get("r2", float("nan")),
            kfold_cv_result.get("mae", float("nan")),
        )

    # --- Final Ridge pipeline — RidgeCV selects alpha on all training data ---
    log.info("  Fitting final Ridge with RidgeCV (alpha tuning on full training set)...")
    _final_steps = [("imp", SimpleImputer(strategy="median"))]
    if clip_outliers:
        _final_steps.append(("clip99", QuantileClipper(upper_quantile=0.99)))
        log.info("  QuantileClipper(99th pct) enabled — Phase I outlier guard")
    _final_steps.extend([
        ("scl", StandardScaler()),
        ("mdl", RidgeCV(
            alphas=alpha_range if alpha_range is not None else np.logspace(-1, 3, 50),
            cv=5,
        )),
    ])
    ridge_pipe = Pipeline(_final_steps)
    ridge_pipe.fit(df[selected].values, y_full.values)
    ridge_alpha      = float(ridge_pipe.named_steps["mdl"].alpha_)
    ridge_r2_train   = float(ridge_pipe.score(df[selected].values, y_full.values))
    log.info("  Ridge train R²=%.3f  selected alpha=%.2f", ridge_r2_train, ridge_alpha)

    train_preds_arr = np.clip(ridge_pipe.predict(df[selected].values), 0, None)
    log.info(
        "  Training preds range: %.2f – %.2f (mean=%.2f)",
        train_preds_arr.min(), train_preds_arr.max(), train_preds_arr.mean(),
    )

    # --- LOYO-CV for LightGBM (per-position params, Steps 5 + 7d) ---
    cv_lgbm = {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan"),
               "year_results": []}
    lgbm_pipe   = None
    lgbm_status = "disabled"

    if fit_lgbm:
        lgbm_params = _LGBM_PARAMS_OVERRIDES.get(position, {
            "n_estimators": 300, "learning_rate": 0.05,
            "max_depth": 4, "min_child_samples": max(3, len(df) // 50),
        })

        log.info("  LOYO-CV (LightGBM, params=%s)...", lgbm_params)
        cv_lgbm = _loyo_cv(df, selected, target, "lgbm", lgbm_params)
        log.info(
            "  LGBM  LOYO: R²=%.3f  MAE=%.2f  RMSE=%.2f",
            cv_lgbm["r2"], cv_lgbm["mae"], cv_lgbm["rmse"],
        )

        # Step 7d: formally retire TE LGBM if gap > 0.10
        lgbm_ridge_gap = float(cv_ridge["r2"]) - float(cv_lgbm["r2"])
        if position == "TE" and lgbm_ridge_gap > 0.10:
            log.warning(
                "  TE LGBM LOYO R²=%.3f vs Ridge=%.3f (gap=%.3f > 0.10) — "
                "TE LightGBM RETIRED per Step 7d.",
                cv_lgbm["r2"], cv_ridge["r2"], lgbm_ridge_gap,
            )
            lgbm_status = "retired_gap_too_large"
            lgbm_pipe   = None
        else:
            lgbm_status = "active"
            mc = [_MONOTONE_DIRECTIONS.get(f, 0) for f in selected]
            lgbm_pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("mdl", LGBMRegressor(
                    n_estimators=lgbm_params["n_estimators"],
                    learning_rate=lgbm_params["learning_rate"],
                    max_depth=lgbm_params["max_depth"],
                    min_child_samples=lgbm_params["min_child_samples"],
                    subsample=lgbm_params.get("subsample", 0.8),
                    colsample_bytree=lgbm_params.get("colsample_bytree", 0.8),
                    min_split_gain=lgbm_params.get("min_split_gain", 0.0),
                    monotone_constraints=mc,
                    random_state=42,
                    verbose=-1,
                )),
            ])
            lgbm_pipe.fit(df[selected].values, y_full.values)
            lgbm_r2_train = float(lgbm_pipe.score(df[selected].values, y_full.values))
            log.info("  LGBM  train R²=%.3f", lgbm_r2_train)

    # --- Persist models ---
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge_pipe, models_dir / f"{position}_ridge{model_suffix}.pkl")
    log.info("  Saved: models/%s_ridge%s.pkl", position, model_suffix)

    features_path = models_dir / f"{position}_features{model_suffix}.json"
    features_path.write_text(json.dumps(selected, indent=2))
    log.info("  Saved: models/%s_features%s.json", position, model_suffix)

    if lgbm_pipe is not None:
        joblib.dump(lgbm_pipe, models_dir / f"{position}_lgbm{model_suffix}.pkl")
        log.info("  Saved: models/%s_lgbm%s.pkl", position, model_suffix)

    # --- Feature importance from Ridge coefficients ---
    ridge_coefs = ridge_pipe.named_steps["mdl"].coef_
    coef_table  = sorted(
        zip(selected, ridge_coefs), key=lambda x: abs(x[1]), reverse=True
    )

    ridge_loyo_r2 = float(cv_ridge["r2"])
    return {
        "position":           position,
        "target":             target,
        "n_train":            int(len(df)),
        "n_features":         len(selected),
        "selected_features":  selected,
        "ridge_r2_train":     ridge_r2_train,
        "ridge_alpha":        ridge_alpha,
        # Part 2: overfitting diagnostics
        "train_loyo_gap":     round(ridge_r2_train - ridge_loyo_r2, 4),   # >0.10 is a red flag
        "features_per_obs":   round(len(selected) / len(df), 4),           # >0.08 is a red flag
        "loyo_method":        "nested_lasso",   # Step 3: honest nested Lasso CV
        "ridge_loyo_r2":      ridge_loyo_r2,
        "ridge_loyo_mae":     float(cv_ridge["mae"]),
        "ridge_loyo_rmse":    float(cv_ridge["rmse"]),
        "ridge_loyo_years":   cv_ridge["year_results"],
        "rolling_cv":         rolling_cv_result,  # Step 7c
        "kfold_cv":           kfold_cv_result,    # Step 7c
        "lgbm_loyo_r2":       float(cv_lgbm["r2"]),
        "lgbm_loyo_mae":      float(cv_lgbm["mae"]),
        "lgbm_loyo_rmse":     float(cv_lgbm["rmse"]),
        "lgbm_loyo_years":    cv_lgbm["year_results"],
        "lgbm_status":        lgbm_status,   # Step 7d: active | retired_gap_too_large | disabled
        "ridge_top_features": [
            {"feature": f, "coef": float(c)} for f, c in coef_table[:15]
        ],
        # Training predictions — ORBIT score reference (percentile vs this distribution)
        "train_preds": [round(float(p), 4) for p in train_preds_arr],
        # D1: conformal prediction interval residuals from LOYO-CV
        "loyo_abs_residuals": cv_ridge.get("abs_residuals", []),
        # D3: rank-order calibration metrics
        "loyo_spearman_rho":   cv_ridge.get("spearman_rho", float("nan")),
        "loyo_top25_hit_rate": cv_ridge.get("top25_hit_rate", float("nan")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train dynasty prospect models. Saves to models/."
    )
    parser.add_argument("--position", choices=["WR", "RB", "TE"])
    parser.add_argument("--all", action="store_true", help="Train all positions.")
    parser.add_argument(
        "--target", default="b2s_score",
        choices=["b2s_score", "year1_ppg"],
        help="Training target (default: b2s_score)",
    )
    parser.add_argument(
        "--no-lgbm", action="store_true",
        help="Skip LightGBM and train Ridge only (faster).",
    )
    parser.add_argument(
        "--start-year", type=int, default=2014,
        help="Minimum draft year in training (default: 2014 — first year with full "
             "PFF coverage; pre-2014 rows use median-imputed PFF creating era bias).",
    )
    parser.add_argument(
        "--end-year", type=int, default=None,
        help="Maximum draft year to include in training (default: all ≤2022).",
    )
    parser.add_argument(
        "--cv-strategy", choices=["loyo", "rolling", "kfold"], default="loyo",
        help="Primary CV strategy (default: loyo). 'rolling' and 'kfold' are also "
             "computed for TE automatically regardless of this flag.",
    )
    parser.add_argument(
        "--no-capital", action="store_true",
        help="Train Phase I (no-capital) model using production/efficiency/athleticism "
             "features only. Saves artifacts with '_nocap' suffix (e.g. WR_ridge_nocap.pkl).",
    )
    parser.add_argument(
        "--capital-only", action="store_true",
        help="Train capital-only model (draft position signals only) as a LOYO baseline. "
             "Saves artifacts with '_capital_only' suffix and metadata_capital_only.json. "
             "Use to verify full model beats raw draft capital (Part 1).",
    )
    args = parser.parse_args()

    from config import get_data_dir
    data_dir   = get_data_dir()
    models_dir = _ROOT / "models"
    positions  = (
        ["WR", "RB", "TE"] if args.all
        else ([args.position] if args.position else ["WR"])
    )

    no_capital   = args.no_capital
    capital_only = args.capital_only
    if no_capital and capital_only:
        log.error("--no-capital and --capital-only are mutually exclusive.")
        sys.exit(1)

    if capital_only:
        model_suffix = "_capital_only"
    elif no_capital:
        model_suffix = "_nocap"
    else:
        model_suffix = ""

    all_meta  = {}
    dfs_eng   = {}   # stored per position for capital-only secondary pass

    for pos in positions:
        df_raw = load_training(pos, data_dir, args.start_year, args.end_year)
        df_eng = engineer_features(df_raw)
        dfs_eng[pos] = df_eng

        if capital_only:
            cand = _CAPITAL_FEATURES_ONLY[pos]
        elif no_capital:
            cand = _CANDIDATE_FEATURES_NOCAP[pos]
        else:
            cand = None  # use default _CANDIDATE_FEATURES[pos] inside fit_position

        # Part 2: apply max_features cap for standard runs (not Phase I / capital-only)
        max_feats = _MAX_FEATURES.get(pos) if (not no_capital and not capital_only) else None

        meta = fit_position(
            position=pos,
            df_all=df_eng,
            target=args.target,
            models_dir=models_dir,
            fit_lgbm=(not args.no_lgbm) and (not no_capital) and (not capital_only),
            cv_strategy=args.cv_strategy,
            feature_candidates=cand,
            model_suffix=model_suffix,
            clip_outliers=no_capital,  # B3: QuantileClipper for Phase I only
            max_features=max_feats,
        )
        all_meta[pos] = meta

        rc = meta.get("rolling_cv", {})
        kc = meta.get("kfold_cv", {})
        log.info(
            "  %s%s FINAL — Ridge LOYO R²=%.3f | LGBM R²=%.3f (%s) | α=%.1f%s",
            pos, model_suffix,
            meta["ridge_loyo_r2"],
            meta["lgbm_loyo_r2"],
            meta["lgbm_status"],
            meta["ridge_alpha"],
            (f" | Rolling R²={rc.get('r2', float('nan')):.3f}"
             f" | K-fold R²={kc.get('r2', float('nan')):.3f}")
            if rc or kc else "",
        )

    # Write aggregate metadata — merge with existing file so a single-position run
    # (e.g. --position RB) doesn't clobber WR/TE keys written by a previous run.
    meta_path = models_dir / f"metadata{model_suffix}.json"
    models_dir.mkdir(parents=True, exist_ok=True)
    existing_meta: dict = {}
    if meta_path.exists():
        try:
            existing_meta = json.loads(meta_path.read_text())
        except Exception as exc:
            log.warning("Could not load existing metadata (will overwrite): %s", exc)
    existing_meta.update(all_meta)
    meta_path.write_text(json.dumps(existing_meta, indent=2, default=str))
    log.info("Metadata saved: %s", meta_path)

    # Summary table
    if capital_only:
        mode_label = " [Capital-Only Baseline]"
    elif no_capital:
        mode_label = " [Phase I — No Capital]"
    else:
        mode_label = ""
    print("\n" + "=" * 90)
    print(f"  Model{mode_label}")
    print(f"{'Position':8}  {'N':>4}  {'Feats':>5}  "
          f"{'Train R2':>8}  {'LOYO R2':>8}  {'LGBM R2':>8}  "
          f"{'Gap':>6}  {'F/Obs':>5}  {'alpha':>7}  {'LGBM Status':>20}")
    print("-" * 90)
    for pos, m in all_meta.items():
        gap   = m.get("train_loyo_gap", float("nan"))
        f_obs = m.get("features_per_obs", float("nan"))
        gap_flag   = " !" if (not np.isnan(gap)   and gap   > 0.10) else ""
        f_obs_flag = " !" if (not np.isnan(f_obs) and f_obs > 0.08) else ""
        print(
            f"{pos:8}  {m['n_train']:>4}  {m['n_features']:>5}  "
            f"{m['ridge_r2_train']:>8.3f}  "
            f"{m['ridge_loyo_r2']:>8.3f}  "
            f"{m['lgbm_loyo_r2']:>8.3f}  "
            f"{gap:>5.3f}{gap_flag}  "
            f"{f_obs:>4.3f}{f_obs_flag}  "
            f"{m['ridge_alpha']:>7.1f}  "
            f"{m['lgbm_status']:>20}"
        )
        rc = m.get("rolling_cv", {})
        kc = m.get("kfold_cv", {})
        if rc.get("r2") is not None:
            print(
                f"         Rolling R²={rc.get('r2', float('nan')):.3f} "
                f"(n_train={rc.get('n_train',0)}, n_test={rc.get('n_test',0)})  "
                f"K-fold R²={kc.get('r2', float('nan')):.3f}"
            )
    print("=" * 90)

    # Part 1: Capital-only baseline diagnostic (standard runs only)
    # Shows how much full ORBIT beats raw draft capital — JJ's explicit goal.
    if not no_capital and not capital_only and dfs_eng:
        print("\n  Running capital-only LOYO baseline (Part 1)...")
        cap_results: dict[str, float] = {}
        for pos in positions:
            df_cap = dfs_eng[pos][dfs_eng[pos][args.target].notna()].copy()
            cap_cands = [
                f for f in _CAPITAL_FEATURES_ONLY.get(pos, [])
                if f in df_cap.columns
                and df_cap[f].nunique(dropna=True) > 1
                and df_cap[f].notna().sum() >= 5
            ]
            if not cap_cands:
                log.warning("  %s: no capital features available for baseline", pos)
                cap_results[pos] = float("nan")
                continue
            cap_loyo = _loyo_cv(
                df_cap, cap_cands, args.target, "ridge",
                alpha_range=_RIDGE_ALPHA_RANGE.get(pos),
            )
            cap_results[pos] = float(cap_loyo["r2"])
            log.info(
                "  %s capital-only LOYO R²=%.3f (using %d features)",
                pos, cap_results[pos], len(cap_cands),
            )

        print("\n=== ORBIT vs Capital-Only LOYO R² ===")
        print(f"{'Position':8}  {'Capital-Only':>12}  {'Full ORBIT':>10}  {'Delta':>6}  {'Status':>8}")
        print("-" * 54)
        for pos in positions:
            cap_r2  = cap_results.get(pos, float("nan"))
            full_r2 = all_meta[pos]["ridge_loyo_r2"]
            delta   = full_r2 - cap_r2 if not np.isnan(cap_r2) else float("nan")
            if np.isnan(delta):
                status = "N/A"
            elif delta > _CAPITAL_DELTA_THRESHOLD:
                status = "PASS"
            else:
                status = "FAIL"
            print(
                f"{pos:8}  {cap_r2:>12.3f}  {full_r2:>10.3f}  "
                f"{delta:>+6.3f}  {status:>8}"
            )
        print(f"\n  PASS threshold: Full ORBIT LOYO > Capital-Only LOYO + {_CAPITAL_DELTA_THRESHOLD:.3f}")
        print("=" * 54)


if __name__ == "__main__":
    main()
