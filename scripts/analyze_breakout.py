"""
Breakout Score Research Script — Phase A.

Computes univariate R² (vs b2s_score) for:
  - Current best_breakout_score (CFBD rec_rate × SOS × age26)
  - PFF-based variants: bs_pff_T = yprr × SOS × max(0, T − age) for T in [24..28]
  - Component decompositions (rec_rate+SOS, rec_rate+age, yprr+SOS, yprr+age, full)
  - Binary threshold flags (yprr ≥ thr for thr in [1.5, 1.8, 2.0, 2.2, 2.5])
  - Ordinal tiers

Outputs driven results determine _BREAKOUT_AGE_THRESHOLD values in:
  - scripts/build_training_set.py
  - scripts/score_class.py

Usage:
    python scripts/analyze_breakout.py

Outputs:
    output/breakout_analysis_{WR,RB,TE}.csv
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from analyze import compute_univariate_r2, load_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TARGET = "b2s_score"
START_YEAR = 2014   # training window: PFF coverage complete from 2014
END_YEAR = 2022     # 3+ NFL seasons complete by end of 2024

AGE_THRESHOLDS = [24, 25, 26, 27, 28]
YPRR_THRESHOLDS = [1.5, 1.8, 2.0, 2.2, 2.5]

# Base metrics from existing training CSV columns (Phase A.1 priority)
BASE_METRICS = [
    "best_breakout_score",     # current formula: rec_rate × SOS × max(0,26-age)
    "best_yprr",               # PFF season-locked to best rec_rate season
    "best_rec_rate",           # rec_yards/team_pass_att (CFBD)
    "best_routes_per_game",    # PFF route volume in best season
    "best_receiving_grade",    # PFF receiving grade in best season
    "best_age",                # raw age at best season — for comparison only
    "age_at_draft",            # raw age at draft — for comparison only
    "best_sp_plus",            # SOS alone
    "college_fantasy_ppg",
    "recruit_rating",
]


def _sos_mult(sp_plus: pd.Series) -> pd.Series:
    """Replicate SOS multiplier: max(0.70, min(1.30, 1 + (SP+ − 5) / 100))."""
    raw = 1.0 + (sp_plus - 5.0) / 100.0
    return raw.clip(lower=0.70, upper=1.30)


def _compute_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Add breakout score variants as new columns in-memory (no DB writes)."""
    df = df.copy()

    # SOS multiplier — fill missing SP+ with 5.0 (FBS average → mult=1.0)
    sp = df["best_sp_plus"].fillna(5.0)
    sos = _sos_mult(sp)

    # Age multiplier variants at each threshold T
    for T in AGE_THRESHOLDS:
        df[f"_age_mult_{T}"] = (T - df["best_age"]).clip(lower=0)

    # ------------------------------------------------------------------
    # A.2 Component decompositions (priority 3)
    # ------------------------------------------------------------------
    # CFBD base × SOS only
    df["rec_rate_x_sos"] = df["best_rec_rate"] * sos
    # PFF base × SOS only
    df["yprr_x_sos"] = df["best_yprr"] * sos

    for T in AGE_THRESHOLDS:
        am = df[f"_age_mult_{T}"]
        # CFBD base variants
        df[f"rec_rate_x_age{T}"]       = df["best_rec_rate"] * am
        df[f"rec_rate_x_sos_x_age{T}"] = df["best_rec_rate"] * sos * am
        # PFF base variants
        df[f"yprr_x_age{T}"]           = df["best_yprr"] * am
        df[f"yprr_x_sos_x_age{T}"]     = df["best_yprr"] * sos * am
        # Primary target: PFF × SOS × age(T) — the Phase B candidate formula
        df[f"bs_pff_{T}"]              = df["best_yprr"] * sos * am

    # ------------------------------------------------------------------
    # A.3 Threshold / tier analysis (priority 6)
    # ------------------------------------------------------------------
    for thr in YPRR_THRESHOLDS:
        col = str(thr).replace(".", "_")
        df[f"yprr_above_{col}"] = (df["best_yprr"] >= thr).astype(float)
        # Approximate BS equivalent: multiply by ~3 (typical age_mult range)
        df[f"bs_above_{col}"]   = (df["best_breakout_score"] >= thr * 3.0).astype(float)

    # Ordinal YPRR tier (0=below average, 1=good, 2=elite)
    df["yprr_tier3"] = pd.cut(
        df["best_yprr"],
        bins=[-np.inf, 1.5, 2.2, np.inf],
        labels=[0, 1, 2],
    ).astype(float)

    # Ordinal breakout_score tier (0/1/2 by tercile)
    bs_lo = df["best_breakout_score"].quantile(0.33)
    bs_hi = df["best_breakout_score"].quantile(0.67)
    df["bs_tier3"] = pd.cut(
        df["best_breakout_score"],
        bins=[-np.inf, bs_lo, bs_hi, np.inf],
        labels=[0, 1, 2],
    ).astype(float)

    # Drop temp helper columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    return df


def _candidate_features(df: pd.DataFrame) -> list[str]:
    """Return all numeric feature columns — excludes identity, target, and JSON blobs."""
    skip = {
        TARGET, "b2s_score",
        "player_id", "cfb_player_id", "nfl_player_id",
        "player_name", "position", "draft_year", "best_team",
        "seasons_json", "match_score",
    }
    return [
        c for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c])
    ]


def run_position(pos: str, data_dir: Path, out_dir: Path) -> None:
    df = load_training(pos, data_dir, start_year=START_YEAR, end_year=END_YEAR)
    n = len(df)
    log.info(
        f"{pos}: {n} rows  target mean={df[TARGET].mean():.2f}  "
        f"std={df[TARGET].std():.2f}  PFF yprr coverage="
        f"{df['best_yprr'].notna().sum()}/{n} "
        f"({100 * df['best_yprr'].notna().mean():.0f}%)"
    )

    df = _compute_variants(df)
    features = _candidate_features(df)
    log.info(f"  Computing univariate R² for {len(features)} features ...")

    results = compute_univariate_r2(df, features, TARGET, n_bootstrap=500)
    results = results.sort_values("r2", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # A.4 Output — ranked table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 68}")
    print(f"=== {pos} Phase I Univariate R2  (n={n},  target={TARGET}) ===")
    print(f"{'=' * 68}")
    print(f"{'Feature':<36}  {'R2':>6}  {'Spearman':>10}  {'n_obs':>6}")
    print("-" * 64)
    for _, row in results.iterrows():
        r2_s  = f"{row['r2']:.3f}"          if not pd.isna(row["r2"])           else "   NaN"
        sp_s  = f"{row['spearman_rho']:.3f}" if not pd.isna(row["spearman_rho"]) else "   NaN"
        n_obs = int(row["n_obs"])            if not pd.isna(row["n_obs"])        else 0
        print(f"{row['feature']:<36}  {r2_s:>6}  {sp_s:>10}  {n_obs:>6}")

    # Summary callouts
    pff_rows = results[results["feature"].str.startswith("bs_pff_")]
    if not pff_rows.empty:
        best_t   = pff_rows.iloc[0]
        cur_row  = results[results["feature"] == "best_breakout_score"]
        cur_r2   = cur_row.iloc[0]["r2"] if not cur_row.empty else float("nan")
        gain     = best_t["r2"] - cur_r2
        sign     = "+" if gain >= 0 else ""
        print(
            f"\n  >> Best PFF variant : {best_t['feature']:<20}  "
            f"R²={best_t['r2']:.3f}  (vs current best_breakout_score R²={cur_r2:.3f},  "
            f"delta={sign}{gain:.3f})"
        )
        print(f"  >> Optimal T for {pos}: {best_t['feature'].split('_')[-1]}")

    # Save CSV
    out_path = out_dir / f"breakout_analysis_{pos}.csv"
    results.to_csv(out_path, index=False)
    log.info(f"  Saved: {out_path}")


def main() -> None:
    data_dir = _ROOT / "data"
    out_dir  = _ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    for pos in ["WR", "RB", "TE"]:
        run_position(pos, data_dir, out_dir)


if __name__ == "__main__":
    main()
