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
  6. Save sklearn Pipelines + metadata to models/

Output files:
  models/{POS}_ridge.pkl      — sklearn Pipeline: SimpleImputer → StandardScaler → Ridge
  models/{POS}_lgbm.pkl       — sklearn Pipeline: SimpleImputer → LGBMRegressor
  models/{POS}_features.json  — ordered feature list used by the pipeline
  models/metadata.json        — CV scores, feature counts, training details

Usage:
    python scripts/fit_model.py --all
    python scripts/fit_model.py --position WR
    python scripts/fit_model.py --position WR --no-lgbm      # Ridge only
    python scripts/fit_model.py --all --target top_season_ppg
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # for analyze.py imports

from analyze import engineer_features, load_training

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-position candidate feature sets
# Features are passed through Lasso CV — only those with non-zero coefficients
# are retained for the final models. PFF stubs (all None) are excluded here;
# add them once PFF+ data is ingested.
# ---------------------------------------------------------------------------

_CANDIDATE_FEATURES = {
    "WR": [
        # Draft capital (strongly collinear — Lasso picks the best signal)
        "log_draft_capital", "draft_capital_score", "overall_pick", "draft_round",
        "draft_premium",
        # Pre-draft market consensus
        "consensus_rank", "position_rank",
        # Capital interaction terms
        "capital_x_age", "capital_x_dominator", "rec_rate_x_capital",
        "breakout_x_capital", "college_fpg_x_capital", "dominator_x_breakout",
        # College production — rates
        "best_rec_rate", "log_rec_rate", "best_dominator", "log_dominator",
        "best_reception_share", "best_ppa_pass",
        "best_usage_pass",
        # College production — counting / other
        "college_fantasy_ppg", "best_rec_yards", "best_receptions",
        "career_rec_yards", "career_receptions", "career_yardage",
        # Age / breakout
        "breakout_age", "best_age", "early_declare",
        # Athleticism
        "weight_lbs", "speed_score", "combined_ath", "agility_score",
        "vertical_jump", "broad_jump",
        # Team context
        "teammate_score",
        # Recruiting
        "recruit_rating",
    ],
    "RB": [
        # Draft capital
        "log_draft_capital", "draft_capital_score", "overall_pick", "draft_round",
        # Capital interactions (capital_x_age is #1 for RBs)
        "capital_x_age", "capital_x_dominator",
        # Pre-draft market
        "consensus_rank", "position_rank",
        # College production
        "best_rec_rate", "best_dominator", "best_reception_share",
        "best_usage_pass", "college_fantasy_ppg",
        "best_rush_ypc", "best_yards_per_touch",
        "career_rec_yards", "career_yardage",
        # Age / breakout
        "breakout_age", "best_age", "early_declare",
        # Athleticism
        "weight_lbs", "speed_score", "agility_score", "combined_ath",
        "vertical_jump", "broad_jump",
        # Team context
        "teammate_score",
        # Recruiting
        "recruit_rating",
    ],
    "TE": [
        # Draft capital (dominant signal for TEs)
        "log_draft_capital", "draft_capital_score", "overall_pick", "draft_round",
        "draft_premium",
        # Capital interactions
        "capital_x_age", "capital_x_dominator",
        "breakout_x_capital",
        # Pre-draft market
        "consensus_rank", "position_rank",
        # College production
        "best_rec_rate", "best_dominator", "best_reception_share",
        "career_rec_per_target", "college_fantasy_ppg",
        "best_ppa_pass",
        # Age / breakout
        "breakout_age", "best_age", "early_declare",
        # Athleticism — especially important for TEs
        "weight_lbs", "speed_score", "combined_ath", "agility_score",
        "forty_time", "vertical_jump", "broad_jump",
        # Team context
        "teammate_score",
        # Recruiting
        "recruit_rating",
    ],
}

# ---------------------------------------------------------------------------
# LGBM monotone constraints: +1 (higher value = better), -1 (lower = better)
# Applied per feature, in the same order as the feature list used for fitting.
# Only for selected features (rebuilt at fit time).
# ---------------------------------------------------------------------------
_MONOTONE_DIRECTIONS = {
    # Draft capital variants — positive (more capital = better outcome)
    "log_draft_capital": 1,   "draft_capital_score": 1,
    "draft_premium": 1,       "capital_x_age": 1,
    "capital_x_dominator": 1, "rec_rate_x_capital": 1,
    "breakout_x_capital": 1,  "college_fpg_x_capital": 1,
    "dominator_x_breakout": 1,
    # Pick number — negative (lower pick = better outcome)
    "overall_pick": -1,       "draft_round": -1,
    # Pre-draft consensus — negative (lower rank = higher expectation = better)
    "consensus_rank": -1,     "position_rank": -1,
    # Production — positive
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
    # Athleticism negative direction (lower = more agile / faster)
    "agility_score": -1,       "forty_time": -1,
    # Recruiting — positive
    "recruit_rating": 1,
    # Team context — positive
    "teammate_score": 1,
    # Age — ambiguous (no hard constraint)
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _loyo_cv(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    model_type: str,  # "ridge" or "lgbm"
    lgbm_params: dict | None = None,
) -> dict:
    """
    Leave-One-Year-Out cross-validation.
    Trains on all years except the test year, predicts on the holdout year.
    Returns per-year results + aggregate metrics.
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

        # Filter to features present in both splits (some engineered features
        # could be fully NaN in a small test year)
        used = [f for f in features if f in train_sub.columns]
        X_train = train_sub[used].values
        y_train = train_sub[target].values
        X_test  = test_sub[used].values
        y_test  = test_sub[target].values

        if model_type == "ridge":
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
                ("mdl", Ridge(alpha=10.0)),
            ])
        else:
            mc = [_MONOTONE_DIRECTIONS.get(f, 0) for f in used]
            params = lgbm_params or {}
            pipe = Pipeline([
                # keep_empty_features=True: don't drop all-NaN columns so that
                # the column count always matches monotone_constraints length
                ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("mdl", LGBMRegressor(
                    n_estimators=params.get("n_estimators", 300),
                    learning_rate=params.get("learning_rate", 0.05),
                    max_depth=params.get("max_depth", 4),
                    min_child_samples=params.get("min_child_samples", 5),
                    subsample=0.8,
                    colsample_bytree=0.8,
                    monotone_constraints=mc,
                    random_state=42,
                    verbose=-1,
                )),
            ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        # Clip negative predictions (B2S can't be < 0)
        preds = np.clip(preds, 0, None)

        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(np.mean((y_test - preds) ** 2)))
        ss_res = np.sum((y_test - preds) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        year_results.append({
            "year": int(test_year),
            "n_test": len(y_test),
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
        })
        all_actual.extend(y_test.tolist())
        all_predicted.extend(preds.tolist())

    if not all_actual:
        return {"year_results": year_results, "r2": float("nan"),
                "mae": float("nan"), "rmse": float("nan")}

    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    agg_r2   = float(r2_score(all_actual, all_predicted))
    agg_mae  = float(mean_absolute_error(all_actual, all_predicted))
    agg_rmse = float(np.sqrt(np.mean((all_actual - all_predicted) ** 2)))

    return {
        "year_results": year_results,
        "r2":   agg_r2,
        "mae":  agg_mae,
        "rmse": agg_rmse,
    }


def _lasso_select(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
) -> list[str]:
    """
    Fit LassoCV on median-imputed, standardized X.
    Returns list of features with non-zero coefficients.
    """
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_imp = imputer.fit_transform(X)
    X_std = scaler.fit_transform(X_imp)

    alphas = np.logspace(-4, 2, 60)
    lasso = LassoCV(alphas=alphas, cv=cv_folds, max_iter=10_000, random_state=42)
    try:
        lasso.fit(X_std, y.values)
    except Exception as e:
        log.warning("LassoCV failed: %s — using all features", e)
        return list(X.columns)

    selected = [f for f, c in zip(X.columns, lasso.coef_) if c != 0.0]
    log.info("  Lasso α=%.4f  selected %d / %d features", lasso.alpha_,
             len(selected), X.shape[1])
    return selected if selected else list(X.columns)


def fit_position(
    position: str,
    df_all: pd.DataFrame,
    target: str,
    models_dir: Path,
    fit_lgbm: bool = True,
) -> dict:
    """
    Full training pipeline for one position.
    Returns metadata dict with CV scores.
    """
    log.info("=== %s (n=%d) ===", position, len(df_all))
    df = df_all[df_all[target].notna()].copy()
    log.info("  %d labeled rows (have %s)", len(df), target)

    # Engineering is already applied upstream (engineer_features)
    candidate_cols = _CANDIDATE_FEATURES[position]
    available = [c for c in candidate_cols if c in df.columns]
    missing_cands = [c for c in candidate_cols if c not in df.columns]
    if missing_cands:
        log.debug("  Candidate features not in data: %s", missing_cands)

    # Drop zero-variance columns (e.g. PFF stubs that are all NaN → median=NaN → 0)
    non_const = [c for c in available if df[c].nunique(dropna=True) > 1]
    dropped_const = set(available) - set(non_const)
    if dropped_const:
        log.info("  Dropped zero-variance features: %s", sorted(dropped_const))
    available = non_const

    # Require at least 5 non-null values per feature
    available = [c for c in available if df[c].notna().sum() >= 5]

    X_full = df[available].copy()
    y_full = df[target]

    # --- Lasso feature selection ---
    log.info("  Running LassoCV on %d candidate features...", len(available))
    selected = _lasso_select(X_full, y_full)
    if not selected:
        selected = available
    log.info("  Selected features (%d): %s", len(selected), selected)

    # --- LOYO-CV for Ridge ---
    log.info("  LOYO-CV (Ridge)...")
    cv_ridge = _loyo_cv(df, selected, target, "ridge")
    log.info(
        "  Ridge LOYO: R²=%.3f  MAE=%.2f  RMSE=%.2f",
        cv_ridge["r2"], cv_ridge["mae"], cv_ridge["rmse"],
    )

    # --- Final Ridge pipeline (fit on all data) ---
    ridge_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("mdl", Ridge(alpha=10.0)),
    ])
    ridge_pipe.fit(df[selected].values, y_full.values)
    ridge_r2_train = ridge_pipe.score(df[selected].values, y_full.values)
    log.info("  Ridge train R²=%.3f", ridge_r2_train)

    # --- LOYO-CV for LightGBM ---
    cv_lgbm = {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan"),
               "year_results": []}
    lgbm_pipe = None
    if fit_lgbm:
        log.info("  LOYO-CV (LightGBM)...")
        lgbm_params = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_samples": max(3, len(df) // 50),
        }
        cv_lgbm = _loyo_cv(df, selected, target, "lgbm", lgbm_params)
        log.info(
            "  LGBM  LOYO: R²=%.3f  MAE=%.2f  RMSE=%.2f",
            cv_lgbm["r2"], cv_lgbm["mae"], cv_lgbm["rmse"],
        )

        # Final LightGBM pipeline (fit on all data)
        mc = [_MONOTONE_DIRECTIONS.get(f, 0) for f in selected]
        lgbm_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("mdl", LGBMRegressor(
                n_estimators=lgbm_params["n_estimators"],
                learning_rate=lgbm_params["learning_rate"],
                max_depth=lgbm_params["max_depth"],
                min_child_samples=lgbm_params["min_child_samples"],
                subsample=0.8,
                colsample_bytree=0.8,
                monotone_constraints=mc,
                random_state=42,
                verbose=-1,
            )),
        ])
        lgbm_pipe.fit(df[selected].values, y_full.values)
        lgbm_r2_train = lgbm_pipe.score(df[selected].values, y_full.values)
        log.info("  LGBM  train R²=%.3f", lgbm_r2_train)

    # --- Persist ---
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge_pipe, models_dir / f"{position}_ridge.pkl")
    log.info("  Saved: models/%s_ridge.pkl", position)

    features_path = models_dir / f"{position}_features.json"
    features_path.write_text(json.dumps(selected, indent=2))
    log.info("  Saved: models/%s_features.json", position)

    if lgbm_pipe is not None:
        joblib.dump(lgbm_pipe, models_dir / f"{position}_lgbm.pkl")
        log.info("  Saved: models/%s_lgbm.pkl", position)

    # --- Feature importance from Ridge coefficients ---
    imputer_fit = SimpleImputer(strategy="median").fit(df[selected])
    scaler_fit  = StandardScaler().fit(imputer_fit.transform(df[selected]))
    ridge_coefs = ridge_pipe.named_steps["mdl"].coef_
    coef_table = sorted(
        zip(selected, ridge_coefs), key=lambda x: abs(x[1]), reverse=True
    )

    return {
        "position":          position,
        "target":            target,
        "n_train":           int(len(df)),
        "n_features":        len(selected),
        "selected_features": selected,
        "ridge_r2_train":    float(ridge_r2_train),
        "ridge_loyo_r2":     float(cv_ridge["r2"]),
        "ridge_loyo_mae":    float(cv_ridge["mae"]),
        "ridge_loyo_rmse":   float(cv_ridge["rmse"]),
        "ridge_loyo_years":  cv_ridge["year_results"],
        "lgbm_loyo_r2":      float(cv_lgbm["r2"]),
        "lgbm_loyo_mae":     float(cv_lgbm["mae"]),
        "lgbm_loyo_rmse":    float(cv_lgbm["rmse"]),
        "lgbm_loyo_years":   cv_lgbm["year_results"],
        "ridge_top_features": [
            {"feature": f, "coef": float(c)} for f, c in coef_table[:15]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train dynasty prospect models. Saves to models/."
    )
    parser.add_argument("--position", choices=["WR", "RB", "TE"])
    parser.add_argument("--all", action="store_true", help="Train all positions.")
    parser.add_argument(
        "--target", default="b2s_score",
        choices=["b2s_score", "top_season_ppg", "year1_ppg"],
        help="Training target (default: b2s_score)",
    )
    parser.add_argument(
        "--no-lgbm", action="store_true",
        help="Skip LightGBM and train Ridge only (faster).",
    )
    parser.add_argument(
        "--start-year", type=int, default=None,
        help="Minimum draft year to include in training (default: all).",
    )
    parser.add_argument(
        "--end-year", type=int, default=None,
        help="Maximum draft year to include in training (default: all).",
    )
    args = parser.parse_args()

    from config import get_data_dir
    data_dir   = get_data_dir()
    models_dir = _ROOT / "models"
    positions  = ["WR", "RB", "TE"] if args.all else ([args.position] if args.position else ["WR"])

    all_meta = {}

    for pos in positions:
        df_raw = load_training(pos, data_dir, args.start_year, args.end_year)
        df_eng = engineer_features(df_raw)

        meta = fit_position(
            position=pos,
            df_all=df_eng,
            target=args.target,
            models_dir=models_dir,
            fit_lgbm=not args.no_lgbm,
        )
        all_meta[pos] = meta

        log.info(
            "  %s FINAL — Ridge LOYO R²=%.3f | LGBM LOYO R²=%.3f",
            pos,
            meta["ridge_loyo_r2"],
            meta["lgbm_loyo_r2"],
        )

    # Write aggregate metadata
    meta_path = models_dir / "metadata.json"
    models_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(all_meta, indent=2, default=str))
    log.info("Metadata saved: %s", meta_path)

    # Summary table
    print("\n" + "=" * 72)
    print(f"{'Position':8}  {'N':>4}  {'Feats':>5}  "
          f"{'Ridge Train':>11}  {'Ridge LOYO':>10}  {'LGBM LOYO':>9}")
    print("-" * 72)
    for pos, m in all_meta.items():
        print(
            f"{pos:8}  {m['n_train']:>4}  {m['n_features']:>5}  "
            f"{m['ridge_r2_train']:>11.3f}  "
            f"{m['ridge_loyo_r2']:>10.3f}  "
            f"{m['lgbm_loyo_r2']:>9.3f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
