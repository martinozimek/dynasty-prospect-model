"""
Train position-specific prospect models and run cross-validation.

This script is intentionally left as a framework to be filled in after
EDA (notebooks/01_feature_correlations.ipynb) establishes which features
matter and what model structure makes sense.

Workflow:
    1. Run build_training_set.py to produce data/training_{WR,RB,TE}.csv
    2. Open notebooks for EDA — establish feature-B2S correlations per position
    3. Come back here to fit and validate the selected model

Usage (when ready):
    python scripts/fit_model.py --position WR
    python scripts/fit_model.py --position RB --model lgbm
    python scripts/fit_model.py --all
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_data_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature sets — update after EDA
# TODO: refine these after notebooks/01_feature_correlations.ipynb
# ---------------------------------------------------------------------------

FEATURES_WR = [
    "best_rec_rate",        # SOS-adjusted rec yards per team pass attempt
    "best_dominator",       # rec yards / team total rec yards
    "best_age",             # age at best season (breakout age proxy)
    "career_rush_yards",    # dual-threat signal
    "early_declare",        # entered draft with eligibility remaining
    "weight_lbs",           # size
    "speed_score",          # athleticism
    "draft_capital_score",  # post-draft market signal
    "recruit_rating",       # pre-college market signal
    "consensus_rank",       # pre-draft market expectation
    "teammate_score",       # offensive context quality
]

FEATURES_RB = [
    "best_rec_rate",        # receiving ability (separates from pure rushers)
    "best_reception_share", # team target share
    "best_age",             # breakout age
    "weight_lbs",
    "speed_score",
    "draft_capital_score",
    "teammate_score",
    "consensus_rank",
]

FEATURES_TE = [
    "draft_capital_score",  # dominant predictor for TEs
    "speed_score",
    "career_rec_per_target",   # YPRR proxy
    "best_age",
    "weight_lbs",
    "consensus_rank",
]

TARGET = "b2s_score"

FEATURE_MAP = {"WR": FEATURES_WR, "RB": FEATURES_RB, "TE": FEATURES_TE}


def _load_training(position: str, data_dir: Path) -> pd.DataFrame:
    path = data_dir / f"training_{position}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {path}\n"
            "Run scripts/build_training_set.py first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %d rows for %s from %s", len(df), position, path)
    return df


def fit_position(position: str, data_dir: Path, model_type: str = "linear") -> None:
    """
    Placeholder: fit model for one position.
    Fill this in after EDA establishes the right feature set.
    """
    df = _load_training(position, data_dir)
    features = FEATURE_MAP[position]

    # Drop rows missing the target
    df = df[df[TARGET].notna()].copy()
    logger.info("  %s: %d labeled rows (have B2S score)", position, len(df))

    # Report feature coverage
    for feat in features:
        if feat not in df.columns:
            logger.warning("  Feature '%s' not in training CSV — add to build_training_set.py", feat)
            continue
        n_missing = df[feat].isna().sum()
        pct = 100 * n_missing / len(df)
        if pct > 20:
            logger.warning("  %s: '%s' missing %.0f%% — consider imputation strategy", position, feat, pct)
        else:
            logger.info("  %s: '%s' %.0f%% populated", position, feat, 100 - pct)

    # TODO: implement after EDA
    # Suggested steps:
    #   1. Impute missing features (median for numeric, 0 for binary flags)
    #   2. Fit OLS baseline (sklearn LinearRegression)
    #   3. Leave-one-year-out cross-validation (train on years N-1, test on year N)
    #   4. Fit LightGBM with monotonic_cst where appropriate
    #   5. Calibrate output to 0-100 scale using B2S distribution
    #   6. Save model to model/{position}_model.pkl
    logger.info("  %s: TODO — fit model after EDA phase.", position)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train dynasty prospect models. Run after EDA notebooks."
    )
    parser.add_argument("--position", type=str, choices=["WR", "RB", "TE"])
    parser.add_argument("--all", action="store_true", help="Fit all three positions.")
    parser.add_argument(
        "--model", type=str, choices=["linear", "lgbm"], default="linear",
        help="Model type (default: linear — start here).",
    )
    args = parser.parse_args()

    data_dir = get_data_dir()
    positions = ["WR", "RB", "TE"] if args.all else ([args.position] if args.position else ["WR"])

    for pos in positions:
        logger.info("=== %s ===", pos)
        fit_position(pos, data_dir, args.model)


if __name__ == "__main__":
    main()
