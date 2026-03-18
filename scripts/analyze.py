#!/usr/bin/env python3
"""
scripts/analyze.py — Dynasty Prospect Model: Regression Analysis Framework

Systematically correlates pre-draft college/combine/draft features to NFL
fantasy performance (B2S Score) using position-by-position regression analysis.
Produces a multi-page PDF report with charts, tables, and statistical results.

Usage:
    python scripts/analyze.py                               # all positions, 2011-2022
    python scripts/analyze.py --position WR                 # WR only
    python scripts/analyze.py --start-year 2015             # recent era
    python scripts/analyze.py --end-year 2019               # older era
    python scripts/analyze.py --hit-threshold 9.0           # binary hit analysis
    python scripts/analyze.py --output output/reports/x.pdf
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# QuantileClipper — Phase I pipeline outlier guard (defined here so both
# fit_model.py and score_class.py can resolve it when loading pickled models)
# ---------------------------------------------------------------------------

class QuantileClipper(BaseEstimator, TransformerMixin):
    """
    Clip each feature at its training-set 99th percentile.
    Applied in Phase I (no-capital) models to prevent extreme athleticism
    outliers from dominating when breakout_score is NaN.
    """
    def __init__(self, upper_quantile: float = 0.99):
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.upper_ = np.nanpercentile(X, self.upper_quantile * 100, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, a_min=None, a_max=self.upper_)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# matplotlib global style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
})
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_POSITIONS = ("WR", "RB", "TE")
_TARGETS = ("b2s_score", "year1_ppg", "year2_ppg", "year3_ppg")
_DEFAULT_TARGET = "b2s_score"

_NON_FEATURES = {
    "nfl_name", "cfb_name", "cfb_player_id", "position",
    "draft_year", "match_score", "b2s_score",
    "year1_ppg", "year2_ppg", "year3_ppg", "qualifying_nfl_seasons",
    "best_team", "recruit_year", "best_season_year",
}

_BASE_FEATURES = [
    # Core ORBIT metrics
    "best_rec_rate", "best_dominator", "best_reception_share",
    "best_age", "best_breakout_score", "best_total_yards_rate",
    "best_games", "best_sp_plus",
    "best_ppa_pass", "best_ppa_overall", "best_ppa_rush",
    "best_usage", "best_usage_pass", "best_usage_rush",
    # Best season per-play rates
    "best_catch_rate", "best_td_rate", "best_ypr", "best_ypt",
    "best_rush_ypc", "best_yards_per_touch",
    "college_fantasy_ppg",
    # Best season counting stats
    "best_rec_yards", "best_receptions", "best_targets",
    "best_rec_tds", "best_rush_tds",
    # Career totals
    "career_seasons", "career_rec_yards", "career_rush_yards",
    "career_targets", "career_receptions", "career_rush_attempts",
    "career_rec_tds", "career_rush_tds", "career_total_tds",
    "career_rec_per_target", "early_declare",
    # Combine / athleticism
    "weight_lbs", "forty_time", "speed_score", "height_inches",
    "vertical_jump", "broad_jump", "three_cone", "shuttle", "bench_press",
    "agility_score",
    # Draft position
    "draft_capital_score", "draft_round", "overall_pick",
    # Recruiting
    "recruit_rating", "recruit_stars", "recruit_rank_national",
    # Pre-draft market expectation
    "consensus_rank", "position_rank",
    # Team context
    "teammate_score",
    # Conference tier
    "power4_conf",
    # TD share in best season
    "best_rec_td_pct",
    # PFF stubs — zero coverage until PFF+ data ingested; retained for future use
    "best_yprr", "best_routes_per_game", "best_receiving_grade",
    "best_contested_catch_rate", "best_drop_rate", "best_target_sep",
    # Draft tier features (ep 1086 — capital works in tiers)
    "draft_tier", "is_top16_rb",
]

_ERA_WINDOWS = [
    ("2011-2015", 2011, 2015),
    ("2016-2019", 2016, 2019),
    ("2020-2022", 2020, 2022),
    ("All",       2011, 2022),
]

_JJ_FEATURES = [
    "best_dominator", "best_age", "best_breakout_score", "draft_capital_score",
    "speed_score", "career_rec_per_target", "best_rec_rate", "recruit_rating",
    "college_fantasy_ppg",
]

_JJ_CLAIMS = {
    "best_dominator":        ">20% = strong signal (rec_yds% + rec_td%)/2",
    "best_age":              "Younger at peak = better; WR especially",
    "best_breakout_score":   "rec_rate × max(0, 26−age); rewards young efficient receivers",
    "draft_capital_score":   "Strongest single predictor of NFL success",
    "speed_score":           "Most predictive for RBs; (wt*200)/40^4",
    "career_rec_per_target": "YPRR proxy; yards per target career",
    "best_rec_rate":         "Rec yds / team pass att — ORBIT rec rate",
    "recruit_rating":        "Elite recruits signal ceiling",
    "college_fantasy_ppg":   "PPR fantasy PPG in best college season",
}

_FEATURE_LABELS = {
    # Core ORBIT
    "best_rec_rate":              "Rec Rate (yd/team_att) — ORBIT",
    "best_dominator":             "Dominator Rating (rec_yd%+rec_td%)/2",
    "best_reception_share":       "Reception Share",
    "best_age":                   "Age at Best College Season",
    "breakout_age":               "Breakout Age (first dom>=threshold) [diagnostic]",
    "best_breakout_score":        "Breakout Score (rec_rate × SOS × max(0,26−age)) WR",
    "best_total_yards_rate":      "Total Yards Rate ((rec+rush)/team_pass_att × SOS × age) RB",
    "best_games":                 "Games Played (Best Season)",
    "best_sp_plus":               "Team SP+ Rating (SOS)",
    "best_ppa_pass":              "PPA Pass (EPA per pass play)",
    "best_ppa_overall":           "PPA Overall (EPA per play)",
    "best_ppa_rush":              "PPA Rush (EPA per rush play)",
    "best_usage":                 "Usage Rate (Best Season)",
    "best_usage_pass":            "Pass Usage Rate (Best Season)",
    "best_usage_rush":            "Rush Usage Rate (Best Season)",
    # Best season rates
    "best_catch_rate":            "Catch Rate (rec/target)",
    "best_td_rate":               "TD Rate (rec_td/target)",
    "best_ypr":                   "Yards Per Reception (best season)",
    "best_ypt":                   "Yards Per Target (best season)",
    "best_rush_ypc":              "Rush Yards Per Carry (best season)",
    "best_yards_per_touch":       "Yards Per Touch (rec+rush)",
    "college_fantasy_ppg":        "College Fantasy PPG (PPR equiv.)",
    # Best season counting
    "best_rec_yards":             "Receiving Yards (Best Season)",
    "best_receptions":            "Receptions (Best Season)",
    "best_targets":               "Targets (Best Season)",
    "best_rec_tds":               "Receiving TDs (Best Season)",
    "best_rush_tds":              "Rushing TDs (Best Season)",
    # Career totals
    "career_seasons":             "Career Seasons Played",
    "career_rec_yards":           "Career Receiving Yards",
    "career_rush_yards":          "Career Rushing Yards",
    "career_targets":             "Career Targets",
    "career_receptions":          "Career Receptions",
    "career_rush_attempts":       "Career Rush Attempts",
    "career_rec_tds":             "Career Receiving TDs",
    "career_rush_tds":            "Career Rushing TDs",
    "career_total_tds":           "Career Total TDs",
    "career_rec_per_target":      "Career Yards Per Target (proxy YPRR)",
    "early_declare":              "Early Declare (<=3 Seasons)",
    # Combine
    "weight_lbs":                 "Weight (lbs)",
    "forty_time":                 "40-Yard Dash (sec)",
    "speed_score":                "Speed Score ((wt*200)/40^4)",
    "height_inches":              "Height (inches)",
    "vertical_jump":              "Vertical Jump (in)",
    "broad_jump":                 "Broad Jump (in)",
    "three_cone":                 "3-Cone Drill (sec)",
    "shuttle":                    "20-Yd Shuttle (sec)",
    "bench_press":                "Bench Press Reps (225 lbs)",
    "agility_score":              "Agility Composite (3-cone+shuttle)/2",
    # Draft
    "draft_capital_score":        "Draft Capital Score (0-100)",
    "draft_round":                "Draft Round",
    "overall_pick":               "Overall Pick Number",
    # Recruiting
    "recruit_rating":             "Recruit Rating (247Sports)",
    "recruit_stars":              "Recruit Stars",
    "recruit_rank_national":      "National Recruit Ranking",
    # Pre-draft market
    "consensus_rank":             "Consensus Big Board Rank",
    "position_rank":              "Position Group Board Rank",
    # Team context
    "teammate_score":             "Teammate Draft Capital",
    # Conference tier
    "power4_conf":                "Power-4 Conference (1=SEC/Big Ten/ACC/Big 12)",
    # TD share
    "best_rec_td_pct":            "Rec TD Share (player TDs / team TDs, best season)",
    # Derived
    "dominator_x_rate":          "Dominator x Rec Rate",
    "age_x_dominator":           "Age x Dominator",
    "capital_x_dominator":       "Draft Capital x Dominator",
    "capital_x_age":             "Draft Capital x Age",
    "rec_rate_x_capital":        "Rec Rate x Draft Capital",
    "log_rec_rate":              "log(Rec Rate + 0.01)",
    "log_dominator":             "log(Dominator + 0.001)",
    "log_draft_capital":         "log(Draft Capital + 1)",
    "rec_rate_sq":               "Rec Rate^2",
    "dominator_sq":              "Dominator^2",
    "age_sq":                    "Age^2 (non-linear penalty)",
    "weight_x_speed":            "Weight x Speed Score",
    "combined_ath":              "Speed x Draft Position Composite",
    "ypr":                       "Career Yards Per Target (YPR)",
    "draft_premium":             "(100 - Big Board Rank) / 100",
    "career_yardage":            "Career Rec + Rush Yards",
    "breakout_x_capital":        "Breakout Age Inverted x Draft Capital [deprecated]",
    "college_fpg_x_capital":     "College Fantasy PPG x Draft Capital",
    "dominator_x_breakout":      "Dominator x Breakout Youth [deprecated]",
    "breakout_score_x_capital":  "Breakout Score x Draft Capital",
    "breakout_score_x_dominator": "Breakout Score x Dominator Rating",
    "total_yards_rate_x_capital": "Total Yards Rate x Draft Capital (RB)",
    "combined_ath_x_capital":    "Combined Ath x Draft Capital (TE)",
    "agility_score":             "Agility Composite (3-cone+shuttle)/2",
    # PFF stubs
    "best_yprr":                 "YPRR — Yards Per Route Run [PFF+]",
    "best_routes_per_game":      "Routes Per Game (Best Season) [PFF+]",
    "best_receiving_grade":      "PFF Receiving Grade 0-100 [PFF+]",
    "best_contested_catch_rate": "Contested Catch Rate [PFF+]",
    "best_drop_rate":            "Drop Rate (lower=better) [PFF+]",
    "best_target_sep":           "Target Separation Yards [PFF+]",
    "yprr_x_capital":            "YPRR x Draft Capital [PFF+]",
    # Draft tier features (ep 1086)
    "draft_tier":                "Draft Tier Ordinal (4=top16, 3=17-50, 2=51-100, 1=day3)",
    "is_top16_rb":               "Top-Half R1 RB Flag (pick<=16 & RB) — R²=0.526 per JJ",
    # Age at draft day (JJ threshold: under-22)
    "age_at_draft":              "Age at Draft Day (JJ under-22 threshold)",
    # Phase I no-capital interaction terms
    "breakout_score_x_yprr":    "Breakout Score × YPRR (Phase I WR double-signal)",
    "rec_rate_x_routes":        "Rec Rate × Routes/Game (Phase I WR volume+efficiency)",
    "total_yards_x_youth":      "Total Yards Rate × Youth 26-age (Phase I RB)",
    "breakout_score_x_grade":   "Breakout Score × PFF Grade (Phase I TE)",
}

_MISSING_CAUSES = {
    "recruit_rating":     "CFBD quota exhausted; resume March 2026",
    "recruit_stars":      "Same as recruit_rating",
    "consensus_rank":     "No free source pre-2016 (structural gap)",
    "position_rank":      "Same as consensus_rank",
    "speed_score":        "Combine non-attenders (not random)",
    "forty_time":         "Combine non-attenders (not random)",
    "weight_lbs":         "Combine non-attenders",
    "height_inches":      "Combine non-attenders",
    "vertical_jump":      "Combine non-attenders",
    "broad_jump":         "Combine non-attenders",
    "three_cone":         "Combine non-attenders (many skill players skip)",
    "shuttle":            "Combine non-attenders",
    "bench_press":        "Many skill players skip bench press at combine",
    "best_ppa_pass":      "CFBD coverage sparse 2011-2013",
    "best_ppa_overall":   "CFBD coverage sparse 2011-2013",
    "best_ppa_rush":      "CFBD coverage sparse 2011-2013",
    "best_usage":         "CFBD coverage sparse 2011-2013",
    "best_usage_pass":    "CFBD coverage sparse 2011-2013",
    "best_usage_rush":    "CFBD coverage sparse 2011-2013",
    "breakout_age":       "None if player never crossed dominator threshold [diagnostic only]",
    "best_breakout_score":    "None if all qualifying seasons have missing age or rec_rate",
    "best_total_yards_rate":  "None if team_pass_att or age missing in all qualifying seasons",
    "best_rush_ypc":      "WR/TE: few rush attempts — mostly None",
    "agility_score":      "Derived from three_cone + shuttle; same coverage as those fields",
    "recruit_rank_national": "Same as recruit_rating — CFBD quota gap 2011-2017",
    # PFF+ stubs — priority addition (see data planning table)
    "best_yprr":             "PFF+ required — $120/yr; college historical back to 2014",
    "best_routes_per_game":  "PFF+ required",
    "best_receiving_grade":  "PFF+ required",
    "best_contested_catch_rate": "PFF+ required",
    "best_drop_rate":        "PFF+ required",
    "best_target_sep":       "PFF+ required",
    # Permanently unavailable (paid/NFL-only) — documented in report
    "air_yards":          "NFL tracking data — unavailable free",
    "snap_share":         "NFL data — unavailable free",
    "adot":               "NFL charting data — unavailable free",
    "juke_rate":          "PlayerProfiler paid metric",
    "yards_created":      "PlayerProfiler paid metric",
    "target_separation":  "NFL Next Gen Stats (restricted)",
}


# =============================================================================
# TIER 1 — Data Loading and Feature Engineering
# =============================================================================

def load_training(
    position: str,
    data_dir: Path,
    start_year: int | None = None,
    end_year: int | None = None,
) -> pd.DataFrame:
    """Load training CSV for a position, optionally filtered by draft_year."""
    path = data_dir / f"training_{position}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {path}\n"
            "Run scripts/build_training_set.py first."
        )
    df = pd.read_csv(path)
    n_raw = len(df)
    if start_year:
        df = df[df["draft_year"] >= start_year]
    if end_year:
        df = df[df["draft_year"] <= end_year]
    log.info(
        f"  {position}: {len(df)} rows (from {n_raw}), "
        f"draft_year {df['draft_year'].min()}-{df['draft_year'].max()}"
    )
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived/interaction features as new columns."""
    df = df.copy()
    # Core ORBIT interaction terms
    df["dominator_x_rate"]    = df["best_dominator"] * df["best_rec_rate"]
    df["age_x_dominator"]     = df["best_age"] * df["best_dominator"]
    df["capital_x_dominator"] = df["draft_capital_score"] * df["best_dominator"]
    df["capital_x_age"]       = df["draft_capital_score"] * df["best_age"]
    df["rec_rate_x_capital"]  = df["best_rec_rate"] * df["draft_capital_score"]
    # Log transforms
    df["log_rec_rate"]        = np.log(df["best_rec_rate"].clip(lower=0.001))
    df["log_dominator"]       = np.log(df["best_dominator"].clip(lower=0.001))
    df["log_draft_capital"]   = np.log(df["draft_capital_score"].clip(lower=0.1) + 1)
    # JJ-style pick capital: log(1000 / (pick + 1)) — Phase C RB +0.068 LOYO delta
    if "overall_pick" in df.columns:
        df["log_pick_capital"] = np.log(1000.0 / (df["overall_pick"].clip(lower=1) + 1))
        if "best_age" in df.columns:
            df["pick_capital_x_age"] = df["log_pick_capital"] * df["best_age"]
        if "best_total_yards_rate" in df.columns:
            df["total_yards_rate_x_pick_capital"] = (
                df["best_total_yards_rate"] * df["log_pick_capital"]
            )
    # 83/17 blended capital: actual pick + consensus rank projection (JJ RB-specific)
    if "draft_capital_score" in df.columns and "consensus_rank" in df.columns:
        _consensus_cap = 100.0 * np.exp(
            -0.023 * (df["consensus_rank"].clip(lower=1) - 1)
        )
        df["blended_capital_rb"] = (
            0.83 * df["draft_capital_score"] + 0.17 * _consensus_cap
        )
        df["log_blended_capital"] = np.log(
            df["blended_capital_rb"].clip(lower=0.1) + 1
        )
    # Polynomial
    df["rec_rate_sq"]         = df["best_rec_rate"] ** 2
    df["dominator_sq"]        = df["best_dominator"] ** 2
    df["age_sq"]              = df["best_age"] ** 2
    # Athleticism composites
    df["weight_x_speed"]      = df["weight_lbs"] * df["speed_score"]
    df["combined_ath"] = (
        df["speed_score"].fillna(0)
        * (100 - df["overall_pick"]).clip(lower=0) / 100
    )
    # Career metrics
    df["ypr"] = np.where(
        df["career_targets"] > 0,
        df["career_rec_yards"] / df["career_targets"],
        np.nan,
    )
    df["career_yardage"] = df["career_rec_yards"] + df["career_rush_yards"]
    # Pre-draft market signal
    df["draft_premium"] = (100 - df["consensus_rank"].clip(upper=300)) / 100
    # Combined athleticism x capital (TE primary — athleticism necessary-but-not-sufficient)
    df["combined_ath_x_capital"] = df["combined_ath"] * df["log_draft_capital"]
    # Breakout score interaction terms (WR primary — JJ's confirmed formula)
    if "best_breakout_score" in df.columns:
        df["breakout_score_x_capital"]   = df["best_breakout_score"] * df["log_draft_capital"]
        df["breakout_score_x_dominator"] = df["best_breakout_score"] * df["best_dominator"]
    # Total yards rate interaction (RB primary — JJ's adjusted yards per team play)
    if "best_total_yards_rate" in df.columns:
        df["total_yards_rate_x_capital"] = df["best_total_yards_rate"] * df["log_draft_capital"]
    # teammate_score × capital interaction (Phase C: +0.057 LOYO delta for RB)
    if "teammate_score" in df.columns:
        df["teammate_score_x_capital"] = df["teammate_score"] * df["log_draft_capital"]
    # College fantasy PPG interaction
    if "college_fantasy_ppg" in df.columns:
        df["college_fpg_x_capital"] = df["college_fantasy_ppg"] * df["draft_capital_score"]
    # Agility composite: average of 3-cone + shuttle (seconds; lower = more agile)
    # Stored as raw average — lower = better, so direction is negative vs. fantasy output
    if "three_cone" in df.columns and "shuttle" in df.columns:
        df["agility_score"] = (df["three_cone"] + df["shuttle"]) / 2
    # PFF interaction terms — active when PFF+ data is ingested
    if "best_yprr" in df.columns:
        df["yprr_x_capital"] = df["best_yprr"] * df["draft_capital_score"]
    # PFF split interaction terms
    if "best_deep_target_rate" in df.columns:
        df["deep_target_x_capital"] = df["best_deep_target_rate"] * df["draft_capital_score"]
    if "best_deep_yprr" in df.columns:
        df["deep_yprr_x_capital"] = df["best_deep_yprr"] * df["draft_capital_score"]
    if "best_man_zone_delta" in df.columns:
        df["man_delta_x_capital"] = df["best_man_zone_delta"] * df["draft_capital_score"]
    if "best_slot_target_rate" in df.columns:
        df["slot_rate_x_capital"] = df["best_slot_target_rate"] * df["draft_capital_score"]
    # Draft tier encoding (ordinal: 4=top16, 3=picks17-50, 2=picks51-100, 1=day3)
    # JJ ep 1086: capital works in tiers, not as precise rankings — top-half R1 RBs R²=0.526
    if "overall_pick" in df.columns:
        def _pick_to_tier(pick):
            if pd.isna(pick):
                return np.nan
            if pick <= 16:
                return 4
            if pick <= 50:
                return 3
            if pick <= 100:
                return 2
            return 1
        df["draft_tier"] = df["overall_pick"].apply(_pick_to_tier)
    # is_top16_rb: binary flag capturing steep-then-flat non-linearity for RBs
    # Top-half first round RBs (picks 1-16) have LOYO R²=0.526 — near-blind follow
    if "overall_pick" in df.columns:
        pos_col = df["position"] if "position" in df.columns else pd.Series("", index=df.index)
        df["is_top16_rb"] = (
            (df["overall_pick"] <= 16) & (pos_col == "RB")
        ).astype(float)
    # Phase I no-capital interaction terms — double-confirmation signals without draft capital
    # WR: production AND efficiency agree → highest-conviction Phase I signal
    if "best_breakout_score" in df.columns and "best_yprr" in df.columns:
        df["breakout_score_x_yprr"] = df["best_breakout_score"].fillna(0) * df["best_yprr"].fillna(0)
    # WR: volume-adjusted efficiency (rec_rate measures team-context dominance; routes = usage)
    if "best_rec_rate" in df.columns and "best_routes_per_game" in df.columns:
        df["rec_rate_x_routes"] = df["best_rec_rate"].fillna(0) * df["best_routes_per_game"].fillna(0)
    # RB: total yards × youth (mirrors breakout_score structure for RBs)
    if "best_total_yards_rate" in df.columns and "best_age" in df.columns:
        df["total_yards_x_youth"] = df["best_total_yards_rate"].fillna(0) * (
            (26.0 - df["best_age"].clip(upper=26)).fillna(0)
        )
    # TE: production quality × PFF receiving grade (athleticism model; grade captures separation)
    if "best_breakout_score" in df.columns and "best_receiving_grade" in df.columns:
        df["breakout_score_x_grade"] = df["best_breakout_score"].fillna(0) * df["best_receiving_grade"].fillna(0)
    return df


def get_feature_lists(df: pd.DataFrame) -> tuple[list, list]:
    """Return (base_features, engineered_features) present in df."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    available = [c for c in numeric_cols if c not in _NON_FEATURES]
    base = [f for f in _BASE_FEATURES if f in available]
    eng = [c for c in available if c not in _BASE_FEATURES]
    return base, eng


def prepare_xy(
    df: pd.DataFrame,
    features: list,
    target: str,
    min_n: int = 10,
) -> tuple:
    """Prepare imputed X, y for multivariate models (median imputation within call)."""
    sub = df[df[target].notna()].copy()
    used = [f for f in features if f in df.columns and sub[f].notna().sum() >= min_n]
    X = sub[used].copy()
    for col in X.columns:
        median = X[col].median()
        X[col] = X[col].fillna(median)
    y = sub[target]
    return X, y, used


# =============================================================================
# TIER 2 — Statistical Analysis
# =============================================================================

def compute_univariate_r2(
    df: pd.DataFrame,
    features: list,
    target: str,
    n_bootstrap: int = 500,
) -> pd.DataFrame:
    """
    Univariate R² for each feature vs target using pairwise complete obs.
    Bootstrap 95% CI on R² (500-rep default).
    """
    records = []
    rng = np.random.default_rng(42)
    target_vals = df[target].values

    for feat in features:
        if feat not in df.columns:
            continue
        feat_vals = df[feat].values
        mask = ~(np.isnan(feat_vals) | np.isnan(target_vals))
        n = int(mask.sum())
        if n < 5:
            records.append({
                "feature": feat, "n_obs": n,
                "r2": np.nan, "r2_ci_lo": np.nan, "r2_ci_hi": np.nan,
                "pearson_r": np.nan, "spearman_rho": np.nan,
                "p_value": np.nan, "direction": "?",
            })
            continue
        x = feat_vals[mask]
        y = target_vals[mask]

        try:
            pr, pp = stats.pearsonr(x, y)
        except Exception:
            pr, pp = np.nan, np.nan
        try:
            sr, _ = stats.spearmanr(x, y)
        except Exception:
            sr = np.nan

        r2 = float(pr ** 2) if not np.isnan(pr) else np.nan
        direction = "+" if (not np.isnan(pr) and pr >= 0) else "-"

        # Bootstrap CI on R²
        r2_boot = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            xb, yb = x[idx], y[idx]
            if np.std(xb) < 1e-10 or np.std(yb) < 1e-10:
                continue
            try:
                rb, _ = stats.pearsonr(xb, yb)
                r2_boot.append(rb ** 2)
            except Exception:
                pass
        if len(r2_boot) >= 10:
            ci_lo = float(np.percentile(r2_boot, 2.5))
            ci_hi = float(np.percentile(r2_boot, 97.5))
        else:
            ci_lo = ci_hi = np.nan

        records.append({
            "feature": feat, "n_obs": n,
            "r2": r2,
            "r2_ci_lo": ci_lo, "r2_ci_hi": ci_hi,
            "pearson_r": float(pr) if not np.isnan(pr) else np.nan,
            "spearman_rho": float(sr) if not np.isnan(sr) else np.nan,
            "p_value": float(pp) if not np.isnan(pp) else np.nan,
            "direction": direction,
        })

    result = pd.DataFrame(records)
    result = result.sort_values("r2", ascending=False, na_position="last").reset_index(drop=True)
    result["rank"] = result.index + 1
    return result


def compute_correlation_matrix(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Pearson correlation matrix, pairwise complete (min_periods=10)."""
    available = [f for f in features if f in df.columns]
    return df[available].corr(method="pearson", min_periods=10)


def fit_ols_full(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fit statsmodels OLS with standardized features."""
    scaler = StandardScaler()
    X_std = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    X_const = sm.add_constant(X_std)
    try:
        model = sm.OLS(y, X_const).fit()
        coef_df = pd.DataFrame({
            "feature": X.columns.tolist(),
            "coef":    model.params[1:].values,
            "std_err": model.bse[1:].values,
            "t_stat":  model.tvalues[1:].values,
            "p_value": model.pvalues[1:].values,
            "ci_lo":   model.conf_int().iloc[1:, 0].values,
            "ci_hi":   model.conf_int().iloc[1:, 1].values,
        }).sort_values("p_value")
        return {
            "coef_table": coef_df,
            "r2": model.rsquared,
            "adj_r2": model.rsquared_adj,
            "f_stat": model.fvalue,
            "f_pvalue": model.f_pvalue,
            "n_obs": int(model.nobs),
            "n_features": len(X.columns),
            "aic": model.aic,
            "bic": model.bic,
        }
    except Exception as e:
        log.warning(f"OLS failed: {e}")
        return {
            "coef_table": pd.DataFrame(), "r2": np.nan, "adj_r2": np.nan,
            "f_stat": np.nan, "f_pvalue": np.nan, "n_obs": 0, "n_features": 0,
            "aic": np.nan, "bic": np.nan,
        }


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor per feature."""
    records = []
    X_const = sm.add_constant(X)
    cols = list(X.columns)
    for i, col in enumerate(cols):
        try:
            vif = variance_inflation_factor(X_const.values, i + 1)
        except Exception:
            vif = np.inf
        records.append({"feature": col, "vif": round(float(vif), 2)})
    return pd.DataFrame(records).sort_values("vif", ascending=False)


def fit_lasso(X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict:
    """LassoCV with StandardScaler."""
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    alphas = np.logspace(-4, 2, 60)
    model = LassoCV(alphas=alphas, cv=cv_folds, max_iter=5000, random_state=42)
    try:
        model.fit(X_std, y)
        coef_df = pd.DataFrame({
            "feature":  X.columns.tolist(),
            "coef":     model.coef_,
            "abs_coef": np.abs(model.coef_),
        }).sort_values("abs_coef", ascending=False)
        selected = coef_df[coef_df["coef"] != 0]["feature"].tolist()
        return {
            "best_alpha": model.alpha_,
            "coef_table": coef_df,
            "selected_features": selected,
            "r2_train": model.score(X_std, y),
        }
    except Exception as e:
        log.warning(f"Lasso failed: {e}")
        return {"best_alpha": np.nan, "coef_table": pd.DataFrame(),
                "selected_features": [], "r2_train": np.nan}


def fit_ridge(X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> dict:
    """RidgeCV with StandardScaler."""
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    model = RidgeCV(alphas=alphas, cv=cv_folds)
    try:
        model.fit(X_std, y)
        coef_df = pd.DataFrame({
            "feature":  X.columns.tolist(),
            "coef":     model.coef_,
            "abs_coef": np.abs(model.coef_),
        }).sort_values("abs_coef", ascending=False)
        return {
            "best_alpha": model.alpha_,
            "coef_table": coef_df,
            "r2_train": model.score(X_std, y),
        }
    except Exception as e:
        log.warning(f"Ridge failed: {e}")
        return {"best_alpha": np.nan, "coef_table": pd.DataFrame(), "r2_train": np.nan}


def leave_one_year_out_cv(
    df: pd.DataFrame,
    features: list,
    target: str,
) -> dict:
    """Leave-one-year-out cross-validation using Ridge (robust to small test sets)."""
    years = sorted(df["draft_year"].unique())
    year_results = []
    all_actual, all_predicted, all_years_label = [], [], []

    for test_year in years:
        train = df[df["draft_year"] != test_year]
        test  = df[df["draft_year"] == test_year]

        X_train, y_train, used = prepare_xy(train, features, target)
        if len(y_train) < 10 or not used:
            continue

        test_sub = test[test[target].notna()].copy()
        if len(test_sub) < 2:
            year_results.append({
                "year": test_year, "n_test": len(test_sub),
                "r2_test": np.nan, "mae_test": np.nan, "rmse_test": np.nan,
            })
            continue

        X_test = test_sub[used].copy()
        for col in used:
            X_test[col] = X_test[col].fillna(X_train[col].median())
        y_test = test_sub[target].values

        scaler = StandardScaler()
        Xs_train = scaler.fit_transform(X_train)
        Xs_test  = scaler.transform(X_test)
        m = Ridge(alpha=10.0).fit(Xs_train, y_train)
        preds = m.predict(Xs_test)

        residuals = y_test - preds
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
        mae  = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        year_results.append({
            "year": test_year, "n_test": len(y_test),
            "r2_test": r2, "mae_test": mae, "rmse_test": rmse,
        })
        all_actual.extend(y_test.tolist())
        all_predicted.extend(preds.tolist())
        all_years_label.extend([test_year] * len(y_test))

    if len(all_actual) < 2:
        return {
            "year_results": year_results, "overall_r2": np.nan,
            "overall_mae": np.nan, "overall_rmse": np.nan,
            "all_actual": [], "all_predicted": [], "all_years": [],
        }
    act  = np.array(all_actual)
    pred = np.array(all_predicted)
    ss_res = np.sum((act - pred) ** 2)
    ss_tot = np.sum((act - act.mean()) ** 2)
    return {
        "year_results": year_results,
        "overall_r2":   float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "overall_mae":  float(np.mean(np.abs(act - pred))),
        "overall_rmse": float(np.sqrt(np.mean((act - pred) ** 2))),
        "all_actual":   all_actual,
        "all_predicted": all_predicted,
        "all_years":    all_years_label,
    }


def compute_era_sensitivity(
    df: pd.DataFrame,
    features: list,
    target: str,
) -> list:
    """Univariate and multivariate R² by era window."""
    results = []
    for label, start, end in _ERA_WINDOWS:
        sub = df[(df["draft_year"] >= start) & (df["draft_year"] <= end)].copy()
        feat_r2 = {}
        for feat in features:
            if feat not in sub.columns:
                continue
            mask = sub[feat].notna() & sub[target].notna()
            if mask.sum() < 5:
                feat_r2[feat] = np.nan
                continue
            try:
                pr, _ = stats.pearsonr(sub.loc[mask, feat], sub.loc[mask, target])
                feat_r2[feat] = float(pr ** 2)
            except Exception:
                feat_r2[feat] = np.nan

        mv_r2 = adj_r2 = np.nan
        if len(sub) >= 10:
            X, y, used = prepare_xy(sub, features, target)
            if len(y) >= 10 and used:
                try:
                    res = fit_ols_full(X, y)
                    mv_r2  = res["r2"]
                    adj_r2 = res["adj_r2"]
                except Exception:
                    pass
        results.append({
            "era": label, "n_obs": len(sub),
            "multivariate_r2": mv_r2, "adj_r2": adj_r2,
            "feature_r2": feat_r2,
        })
    return results


def compute_hit_rate(df: pd.DataFrame, target: str, threshold: float) -> dict:
    """Binary hit-rate analysis: hit = (target >= threshold)."""
    sub = df[df[target].notna()].copy()
    sub["_hit"] = (sub[target] >= threshold).astype(int)
    n_hits  = int(sub["_hit"].sum())
    n_total = len(sub)

    records = []
    for feat in _BASE_FEATURES:
        if feat not in sub.columns:
            continue
        mask = sub[feat].notna()
        if mask.sum() < 10:
            continue
        x = sub.loc[mask, feat]
        h = sub.loc[mask, "_hit"]
        try:
            corr, pval = stats.pointbiserialr(h, x)
        except Exception:
            corr, pval = np.nan, np.nan
        median = x.median()
        hr_above = float(h[x >= median].mean()) if (x >= median).sum() > 0 else np.nan
        hr_below = float(h[x <  median].mean()) if (x <  median).sum() > 0 else np.nan
        delta = (hr_above - hr_below) if (hr_above is not None and hr_below is not None
                                           and not np.isnan(hr_above) and not np.isnan(hr_below)) else np.nan
        records.append({
            "feature": feat,
            "correlation": float(corr) if not np.isnan(corr) else np.nan,
            "p_value":     float(pval) if not np.isnan(pval) else np.nan,
            "hit_rate_above_median": hr_above,
            "hit_rate_below_median": hr_below,
            "delta": delta,
        })

    feat_df = pd.DataFrame(records).sort_values("delta", ascending=False, na_position="last")
    return {
        "threshold": threshold,
        "n_hits":    n_hits,
        "n_total":   n_total,
        "hit_rate":  n_hits / n_total if n_total > 0 else 0.0,
        "feature_results": feat_df,
    }


def compute_missing_data_summary(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Missing data counts and rates per feature."""
    n = len(df)
    records = []
    for feat in features:
        if feat not in df.columns:
            continue
        n_miss = int(df[feat].isna().sum())
        pct    = n_miss / n * 100
        status = "OK" if pct < 10 else ("WARN" if pct < 40 else "LOW")
        cause  = _MISSING_CAUSES.get(feat, "")
        records.append({
            "feature":      feat,
            "n_total":      n,
            "n_missing":    n_miss,
            "n_available":  n - n_miss,
            "pct_missing":  round(pct, 1),
            "status":       status,
            "coverage_note": cause,
        })
    return pd.DataFrame(records).sort_values("pct_missing", ascending=False)


# =============================================================================
# TIER 3 — PDF Rendering
# =============================================================================

def _label(feat: str) -> str:
    return _FEATURE_LABELS.get(feat, feat)


def _era_color(yr: int) -> str:
    if yr <= 2015:
        return "#4472C4"
    elif yr <= 2019:
        return "#ED7D31"
    return "#70AD47"


def _fig_cover(
    pdf: PdfPages,
    positions: list,
    target: str,
    start_year: int | None,
    end_year: int | None,
    timestamp: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    yr_range = f"{start_year or 2011}-{end_year or 2022}"

    ax.text(0.5, 0.91, "Dynasty Prospect Model", fontsize=22, fontweight="bold",
            ha="center", va="center", transform=ax.transAxes)
    ax.text(0.5, 0.84, "College-to-NFL Regression Analysis",
            fontsize=16, ha="center", va="center", transform=ax.transAxes, color="#444")
    ax.text(0.5, 0.77,
            f"Target: {target}  |  Draft classes: {yr_range}  |  Positions: {', '.join(positions)}",
            fontsize=11, ha="center", va="center", transform=ax.transAxes, color="#666")

    method_text = (
        "METHODOLOGY\n\n"
        f"Training data : {yr_range} NFL draft classes (WR/RB/TE)\n"
        "College data  : CFBD API — season totals, usage, PPA, recruiting\n"
        "NFL outcomes  : nflverse player_stats (PPR fantasy scoring)\n"
        "Pre-draft data: nflmockdraftdatabase.com consensus board (2016+)\n\n"
        "Target — B2S Score:\n"
        "  WR/RB: avg PPR PPG of best-2 seasons, first 3 NFL years (>=8 games)\n"
        "  TE:    best single-season PPG, first 3 NFL years (>=8 games)\n\n"
        "Analysis includes:\n"
        "  * Univariate R2 scan — pairwise complete, 500-rep bootstrap CI\n"
        "  * Multivariate OLS — standardized features (coefficients comparable)\n"
        "  * Lasso / Ridge regularized regression with CV alpha selection\n"
        "  * Leave-one-year-out cross-validation (temporal honesty)\n"
        "  * Era sensitivity analysis (2011-15 / 2016-19 / 2020-22 / All)\n"
        "  * Derived/interaction feature engineering (15 new features)\n"
        "  * JJ Zachariason benchmark — his metric claims vs our data\n"
        "  * Missing data inventory and improvement roadmap"
    )
    ax.text(0.10, 0.68, method_text, fontsize=9, va="top", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#F5F5F5", edgecolor="#CCC"),
            fontfamily="monospace")

    ax.text(0.5, 0.03,
            f"Generated {timestamp}  |  martinozimek/dynasty-prospect-model",
            fontsize=8, ha="center", va="bottom", transform=ax.transAxes, color="#999")

    pdf.savefig(fig)
    plt.close(fig)


def _fig_data_overview(
    pdf: PdfPages,
    position: str,
    df: pd.DataFrame,
    target: str,
    missing_summary: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(f"{position} — Data Overview", fontsize=13, fontweight="bold", y=0.99)

    # Top-left: draft class distribution
    ax = axes[0, 0]
    yr_counts = df["draft_year"].value_counts().sort_index()
    bars = ax.barh(yr_counts.index.astype(str), yr_counts.values, color="#4472C4", alpha=0.8)
    ax.set_xlabel("Players")
    ax.set_title(f"Draft Class Distribution (n={len(df)})")
    for bar, val in zip(bars, yr_counts.values):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=7)

    # Top-right: target histogram
    ax = axes[0, 1]
    tvals = df[target].dropna()
    ax.hist(tvals, bins=25, color="#ED7D31", alpha=0.75, edgecolor="white")
    ax.axvline(tvals.mean(),   color="red",    linestyle="--", linewidth=1.2,
               label=f"Mean={tvals.mean():.2f}")
    ax.axvline(tvals.median(), color="orange", linestyle="--", linewidth=1.2,
               label=f"Median={tvals.median():.2f}")
    ax.set_xlabel(target)
    ax.set_ylabel("Count")
    ax.set_title(f"{target} Distribution")
    ax.legend(fontsize=7)
    zero_pct = (tvals == 0).mean() * 100
    ax.text(0.98, 0.95,
            f"n={len(tvals)}\nSD={tvals.std():.2f}\nZero={zero_pct:.1f}%",
            ha="right", va="top", transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Bottom-left: missing data
    ax = axes[1, 0]
    top_miss = missing_summary[missing_summary["pct_missing"] > 0].head(14)
    colors_m = ["#E74C3C" if p > 50 else "#F39C12" if p > 10 else "#27AE60"
                for p in top_miss["pct_missing"]]
    ax.barh(top_miss["feature"].apply(_label), top_miss["pct_missing"],
            color=colors_m, alpha=0.85)
    ax.set_xlabel("% Missing")
    ax.set_title("Feature Missingness (top 14)")
    ax.set_xlim(0, 108)
    for i, (_, row) in enumerate(top_miss.iterrows()):
        ax.text(row["pct_missing"] + 1, i, f"{row['pct_missing']:.0f}%",
                va="center", fontsize=7)

    # Bottom-right: qualifying NFL seasons
    ax = axes[1, 1]
    if "qualifying_nfl_seasons" in df.columns:
        q_counts = df["qualifying_nfl_seasons"].value_counts().sort_index()
        bars2 = ax.bar(q_counts.index.astype(str), q_counts.values,
                       color="#7030A0", alpha=0.8)
        ax.set_xlabel("Qualifying NFL Seasons (>=8 games)")
        ax.set_ylabel("Players")
        ax.set_title("NFL Season Qualification")
        for bar, val in zip(bars2, q_counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(val), ha="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "qualifying_nfl_seasons not available",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _fig_univariate_table(
    pdf: PdfPages,
    position: str,
    r2_df: pd.DataFrame,
    target: str,
    top_n: int = 20,
) -> None:
    top = r2_df.dropna(subset=["r2"]).head(top_n)
    if len(top) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.suptitle(f"{position} — Univariate R2 Ranking vs {target}", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.axis("off")
    table_data = []
    for _, row in top.iterrows():
        ci_lo = row.get("r2_ci_lo", np.nan)
        ci_hi = row.get("r2_ci_hi", np.nan)
        ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]" if not (np.isnan(ci_lo) or np.isnan(ci_hi)) else "--"
        pr = row.get("pearson_r", np.nan)
        pv = row.get("p_value", np.nan)
        table_data.append([
            int(row["rank"]),
            _label(row["feature"])[:30],
            int(row["n_obs"]),
            f"{row['r2']:.4f}",
            ci_str,
            f"{pr:+.3f}" if not np.isnan(pr) else "--",
            f"{pv:.3f}" if not np.isnan(pv) else "--",
        ])
    tbl = ax.table(
        cellText=table_data,
        colLabels=["Rank", "Feature", "N", "R2", "95% CI", "r", "p-val"],
        loc="upper center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.25)
    for i, (_, row) in enumerate(top.iterrows()):
        r2  = row["r2"]
        pv  = row.get("p_value", 1.0)
        bg  = "#C6EFCE" if (r2 >= 0.05 and not np.isnan(pv) and pv < 0.05) \
              else "#FFEB9C" if r2 >= 0.02 else "white"
        for j in range(7):
            tbl[(i + 1, j)].set_facecolor(bg)
    ax.text(0.5, -0.01,
            "Green = R2>=0.05 & p<0.05  |  Yellow = R2>=0.02  |  95% CI from 500-rep bootstrap",
            ha="center", va="top", transform=ax.transAxes, fontsize=7, color="#666")

    ax2 = axes[1]
    y_pos = range(len(top) - 1, -1, -1)
    colors_bar = [
        "#4472C4" if (not np.isnan(row.get("p_value", 1.0)) and row.get("p_value", 1.0) < 0.05)
        else "#BDD7EE"
        for _, row in top.iterrows()
    ]
    bars = ax2.barh(list(y_pos), top["r2"].values, color=colors_bar, alpha=0.9, height=0.7)
    ci_lo_arr = (top["r2"] - top["r2_ci_lo"]).clip(lower=0).values
    ci_hi_arr = (top["r2_ci_hi"] - top["r2"]).clip(lower=0).values
    valid_ci = ~(top["r2_ci_lo"].isna() | top["r2_ci_hi"].isna())
    if valid_ci.any():
        ax2.errorbar(
            top["r2"].values, list(y_pos),
            xerr=[ci_lo_arr, ci_hi_arr],
            fmt="none", color="#555", capsize=2, linewidth=0.8,
        )
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels([_label(f)[:22] for f in top["feature"]], fontsize=7)
    ax2.set_xlabel("R2")
    ax2.set_title("R2 (blue = p<0.05)")
    ax2.set_xlim(0, max(top["r2"].max() * 1.3, 0.05))
    for bar, val in zip(bars, top["r2"].values):
        ax2.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=6.5)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _fig_scatter_grid(
    pdf: PdfPages,
    position: str,
    df: pd.DataFrame,
    r2_df: pd.DataFrame,
    target: str,
    top_n: int = 6,
) -> None:
    top_feats = r2_df.dropna(subset=["r2"]).head(top_n)["feature"].tolist()
    if not top_feats:
        return

    n_cols = 3
    n_rows = int(np.ceil(len(top_feats) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 8.5))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(f"{position} — Top Feature Scatter Plots vs {target}",
                 fontsize=13, fontweight="bold")

    for i, feat in enumerate(top_feats):
        ax = axes[i]
        mask = df[feat].notna() & df[target].notna()
        sub  = df[mask]
        x    = sub[feat].values
        y    = sub[target].values
        c_pts = [_era_color(yr) for yr in sub["draft_year"]]

        ax.scatter(x, y, c=c_pts, alpha=0.5, s=20, linewidths=0)
        if len(x) >= 3:
            try:
                m, b = np.polyfit(x, y, 1)
                xline = np.linspace(x.min(), x.max(), 100)
                ax.plot(xline, m * xline + b, color="#C00", linewidth=1.2, zorder=5)
            except Exception:
                pass

        r2_row = r2_df[r2_df["feature"] == feat]
        r2_val = r2_row["r2"].values[0] if len(r2_row) else np.nan
        p_val  = r2_row["p_value"].values[0] if len(r2_row) else np.nan
        ax.text(0.04, 0.96,
                f"R2={r2_val:.4f}\nn={len(x)}\np={p_val:.3f}" if not np.isnan(p_val)
                else f"R2={r2_val:.4f}\nn={len(x)}",
                ha="left", va="top", transform=ax.transAxes, fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        ax.set_xlabel(_label(feat)[:30], fontsize=7)
        ax.set_ylabel(target if i % n_cols == 0 else "", fontsize=7)
        ax.set_title(f"#{i + 1} {feat}", fontsize=8)

    for j in range(len(top_feats), len(axes)):
        axes[j].axis("off")

    legend_elements = [
        Patch(facecolor="#4472C4", label="2011-2015"),
        Patch(facecolor="#ED7D31", label="2016-2019"),
        Patch(facecolor="#70AD47", label="2020-2022"),
    ]
    fig.legend(handles=legend_elements, loc="lower right", fontsize=8, ncol=3,
               bbox_to_anchor=(0.99, 0.01))
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _fig_correlation_heatmap(
    pdf: PdfPages, position: str, corr_matrix: pd.DataFrame
) -> None:
    n = len(corr_matrix)
    if n < 2:
        return
    fig_h = max(7, min(10, n * 0.38))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    fig.suptitle(f"{position} — Feature Correlation Heatmap (Pearson r)",
                 fontsize=13, fontweight="bold")

    mask   = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    labels = [_label(c)[:24] for c in corr_matrix.columns]
    annot  = n <= 20
    sns.heatmap(
        corr_matrix, mask=mask, ax=ax,
        cmap="RdBu_r", vmin=-1, vmax=1,
        annot=annot, fmt=".2f" if annot else "",
        annot_kws={"size": 6} if annot else {},
        xticklabels=labels, yticklabels=labels,
        square=True, linewidths=0.3 if annot else 0,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=6.5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6.5)
    ax.text(0, -0.04, "|r| > 0.7 = potential multicollinearity  |  Lower triangle only  |  min 10 obs",
            transform=ax.transAxes, fontsize=8, color="#666")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _fig_ols_summary(
    pdf: PdfPages, position: str, ols_result: dict, vif_df: pd.DataFrame, target: str
) -> None:
    coef_df = ols_result.get("coef_table", pd.DataFrame())
    if coef_df.empty:
        return
    r2  = ols_result.get("r2", np.nan)
    ar2 = ols_result.get("adj_r2", np.nan)
    f   = ols_result.get("f_stat", np.nan)
    fp  = ols_result.get("f_pvalue", np.nan)
    n   = ols_result.get("n_obs", 0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.suptitle(
        f"{position} — OLS Regression vs {target}\n"
        f"R2={r2:.4f}  Adj-R2={ar2:.4f}  F={f:.2f} (p={fp:.4f})  n={n}"
        f"  [Features standardized for comparability]",
        fontsize=10, fontweight="bold"
    )

    merged = coef_df.merge(vif_df, on="feature", how="left")

    ax = axes[0]
    ax.axis("off")
    table_data = []
    for _, row in merged.iterrows():
        p    = row.get("p_value", 1.0)
        vif  = row.get("vif", np.nan)
        table_data.append([
            _label(row["feature"])[:26],
            f"{row['coef']:+.4f}",
            f"{row['std_err']:.4f}",
            f"{row['t_stat']:+.2f}",
            f"{p:.3f}",
            f"{vif:.1f}" if not np.isnan(vif) else "--",
        ])
    tbl = ax.table(
        cellText=table_data,
        colLabels=["Feature", "Coef", "SE", "t", "p-val", "VIF"],
        loc="upper center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.2)
    for i, (_, row) in enumerate(merged.iterrows()):
        p   = row.get("p_value", 1.0)
        vif = row.get("vif", np.nan)
        bg  = "#C6EFCE" if p < 0.05 else "#FFEB9C" if p < 0.10 else "white"
        for j in range(6):
            tbl[(i + 1, j)].set_facecolor(bg)
        if not np.isnan(vif) and vif > 10:
            tbl[(i + 1, 5)].set_facecolor("#FFC7CE")
    ax.text(0.5, -0.01, "Green=p<0.05  Yellow=p<0.10  Red VIF=>10",
            ha="center", va="top", transform=ax.transAxes, fontsize=7, color="#666")

    ax2 = axes[1]
    y_pos = range(len(merged))
    sig   = merged["p_value"] < 0.05
    c_f   = ["#4472C4" if s else "#BDD7EE" for s in sig]
    ax2.scatter(merged["coef"], list(y_pos), c=c_f, s=30, zorder=5)
    ax2.errorbar(
        merged["coef"], list(y_pos),
        xerr=1.96 * merged["std_err"],
        fmt="none", color="#888", linewidth=1, capsize=3,
    )
    ax2.axvline(0, color="red", linewidth=0.8, linestyle="--")
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels([_label(f)[:20] for f in merged["feature"]], fontsize=7)
    ax2.set_xlabel("Std. Coefficient (+/-1.96 SE)", fontsize=8)
    ax2.set_title("Forest Plot (blue = p<0.05)")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _fig_regularized(
    pdf: PdfPages, position: str, lasso_result: dict, ridge_result: dict, target: str
) -> None:
    lasso_coef = lasso_result.get("coef_table", pd.DataFrame())
    ridge_coef = ridge_result.get("coef_table", pd.DataFrame())
    if lasso_coef.empty:
        return

    la     = lasso_result.get("best_alpha", np.nan)
    ra     = ridge_result.get("best_alpha", np.nan)
    lr     = lasso_result.get("r2_train", np.nan)
    rr     = ridge_result.get("r2_train", np.nan)
    n_sel  = len(lasso_result.get("selected_features", []))
    n_tot  = len(lasso_coef)

    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    fig.suptitle(f"{position} — Regularized Regression vs {target}",
                 fontsize=13, fontweight="bold")

    # Left: Lasso feature importance
    ax = axes[0]
    lc = lasso_coef.head(25)
    c_lasso = ["#4472C4" if c != 0 else "#D0D0D0" for c in lc["coef"]]
    y_pos   = range(len(lc) - 1, -1, -1)
    ax.barh(list(y_pos), lc["abs_coef"].values, color=c_lasso, alpha=0.9, height=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([_label(f)[:22] for f in lc["feature"]], fontsize=7)
    ax.set_xlabel("|Coefficient| (standardized features)")
    ax.set_title(f"Lasso Feature Selection\nalpha={la:.4f}  {n_sel}/{n_tot} selected  R2={lr:.4f}")
    ax.text(0.98, 0.02,
            f"Blue = selected ({n_sel})\nGray = zeroed ({n_tot - n_sel})",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Right: Ridge vs Lasso scatter
    ax2 = axes[1]
    if not ridge_coef.empty:
        m = lasso_coef.merge(ridge_coef, on="feature", suffixes=("_lasso", "_ridge"))
        ax2.scatter(m["coef_lasso"], m["coef_ridge"], alpha=0.7, s=40, color="#ED7D31")
        lim = max(m["coef_lasso"].abs().max(), m["coef_ridge"].abs().max()) * 1.15
        ax2.plot([-lim, lim], [-lim, lim], "r--", linewidth=0.8, label="y=x (no shrinkage)")
        ax2.set_xlabel("Lasso coefficient")
        ax2.set_ylabel("Ridge coefficient")
        ax2.set_title(f"Ridge vs Lasso\nRidge alpha={ra:.2f}  R2={rr:.4f}")
        ax2.legend(fontsize=8)
        top5 = lasso_result.get("selected_features", [])[:5]
        for _, row in m[m["feature"].isin(top5)].iterrows():
            ax2.annotate(_label(row["feature"])[:16],
                         (row["coef_lasso"], row["coef_ridge"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=6)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _fig_jj_benchmark(
    pdf: PdfPages, position: str, r2_df: pd.DataFrame, target: str
) -> None:
    mean_r2 = r2_df["r2"].dropna().mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.suptitle(f"{position} — JJ Zachariason Metric Benchmark vs {target}",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.axis("off")
    table_data = []
    for feat in _JJ_FEATURES:
        row = r2_df[r2_df["feature"] == feat]
        r2_val = row.iloc[0]["r2"]      if len(row) else np.nan
        p_val  = row.iloc[0]["p_value"] if len(row) else np.nan
        n_obs  = row.iloc[0]["n_obs"]   if len(row) else np.nan
        rank   = str(int(row.iloc[0].get("rank", 0))) if len(row) else "--"
        claim  = _JJ_CLAIMS.get(feat, "--")
        table_data.append([
            _label(feat)[:28],
            claim[:40],
            f"{r2_val:.4f}" if not np.isnan(r2_val) else "--",
            rank,
            str(int(n_obs)) if (not isinstance(n_obs, float) or not np.isnan(n_obs)) else "--",
            f"{p_val:.3f}"  if not np.isnan(p_val)  else "--",
        ])
    tbl = ax.table(
        cellText=table_data,
        colLabels=["Feature", "JZ Claim", "Our R2", "Rank", "N", "p-val"],
        loc="upper center", cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 2.0)
    for i, feat in enumerate(_JJ_FEATURES):
        row = r2_df[r2_df["feature"] == feat]
        r2v = row.iloc[0]["r2"] if len(row) else np.nan
        bg  = "#C6EFCE" if (not np.isnan(r2v) and r2v >= 0.05) \
              else "#FFEB9C" if (not np.isnan(r2v) and r2v >= 0.01) else "#FFC7CE"
        for j in range(6):
            tbl[(i + 1, j)].set_facecolor(bg)
    ax.text(0.5, 0.05,
            "Green=R2>=0.05  |  Yellow=R2>=0.01  |  Red=R2<0.01 (weak signal in our data)",
            ha="center", va="bottom", transform=ax.transAxes, fontsize=8, color="#666")

    ax2 = axes[1]
    jj_r2s = []
    jj_lbs = []
    for feat in _JJ_FEATURES:
        row = r2_df[r2_df["feature"] == feat]
        jj_r2s.append(row.iloc[0]["r2"] if len(row) else 0.0)
        jj_lbs.append(_label(feat)[:20])
    y_pos = range(len(jj_r2s) - 1, -1, -1)
    ax2.barh(list(y_pos), jj_r2s, color="#4472C4", alpha=0.8, height=0.6)
    ax2.axvline(mean_r2, color="red", linestyle="--", linewidth=1.2,
                label=f"Mean all features R2={mean_r2:.4f}")
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(jj_lbs, fontsize=8)
    ax2.set_xlabel("Univariate R2")
    ax2.set_title("JJ Metrics vs Universe Average")
    ax2.legend(fontsize=8)
    for yp, v in zip(y_pos, jj_r2s):
        if not np.isnan(v):
            ax2.text(v + 0.001, yp, f"{v:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _fig_era_sensitivity(
    pdf: PdfPages, position: str, era_results: list, top_features: list, target: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    fig.suptitle(f"{position} — Era Sensitivity Analysis", fontsize=13, fontweight="bold")

    era_labels = [e["era"] for e in era_results]

    ax = axes[0]
    for feat in top_features[:6]:
        vals = [e["feature_r2"].get(feat, np.nan) for e in era_results]
        ns   = [e["n_obs"] for e in era_results]
        style = "-o" if all(n >= 15 for n in ns) else "--o"
        ax.plot(era_labels, vals, style, linewidth=1.5, markersize=5,
                label=_label(feat)[:20])
        for xl, val, n in zip(era_labels, vals, ns):
            if not np.isnan(val) and n < 15:
                ax.annotate(f"n={n}", (xl, val),
                            textcoords="offset points", xytext=(0, 7),
                            fontsize=6, ha="center", color="#C00")
    ax.set_xlabel("Era")
    ax.set_ylabel("R2")
    ax.set_title("Top Feature R2 by Era\n(-- = thin era <15 obs)")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.set_ylim(bottom=0)

    ax2 = axes[1]
    mv_r2  = [e.get("multivariate_r2", np.nan) for e in era_results]
    adj_r2 = [e.get("adj_r2", np.nan)          for e in era_results]
    ns2    = [e["n_obs"] for e in era_results]
    x_pos  = range(len(era_labels))
    width  = 0.35
    b1 = ax2.bar([x - width / 2 for x in x_pos], mv_r2,  width, label="R2",     color="#4472C4", alpha=0.8)
    b2 = ax2.bar([x + width / 2 for x in x_pos], adj_r2, width, label="Adj-R2", color="#ED7D31", alpha=0.8)
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels(era_labels)
    ax2.set_ylabel("R2")
    ax2.set_title("Multivariate OLS R2 by Era")
    ax2.legend(fontsize=8)
    for b, n in zip(b1, ns2):
        ax2.text(b.get_x() + b.get_width(), b.get_height() + 0.005,
                 f"n={n}", fontsize=7, ha="center")
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _fig_feature_engineering(
    pdf: PdfPages,
    position: str,
    eng_r2_df: pd.DataFrame,
    base_r2_df: pd.DataFrame,
    target: str,
) -> None:
    eng_top = eng_r2_df.dropna(subset=["r2"]).head(15)
    if eng_top.empty:
        return
    best_base_r2 = base_r2_df["r2"].dropna().max() if not base_r2_df.empty else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.suptitle(f"{position} — Feature Engineering Results vs {target}",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.axis("off")
    table_data = []
    for _, row in eng_top.iterrows():
        r2    = row["r2"]
        delta = (r2 - best_base_r2) if not np.isnan(best_base_r2) else np.nan
        table_data.append([
            _label(row["feature"])[:30],
            f"{r2:.4f}",
            f"{best_base_r2:.4f}" if not np.isnan(best_base_r2) else "--",
            f"{delta:+.4f}"       if not np.isnan(delta)         else "--",
            str(int(row["n_obs"])),
        ])
    tbl = ax.table(
        cellText=table_data,
        colLabels=["Derived Feature", "R2", "Best Base R2", "Delta", "N"],
        loc="upper center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.55)
    for i, (_, row) in enumerate(eng_top.iterrows()):
        delta = (row["r2"] - best_base_r2) if not np.isnan(best_base_r2) else np.nan
        bg = "#C6EFCE" if (not np.isnan(delta) and delta > 0.01) \
             else "#FFEB9C" if row["r2"] >= 0.02 else "white"
        for j in range(5):
            tbl[(i + 1, j)].set_facecolor(bg)

    ax2 = axes[1]
    y_pos = range(len(eng_top) - 1, -1, -1)
    ax2.barh(list(y_pos), eng_top["r2"].values,
             color="#70AD47", alpha=0.85, height=0.6, label="Derived feature")
    if not np.isnan(best_base_r2):
        ax2.axvline(best_base_r2, color="red", linestyle="--", linewidth=1.2,
                    label=f"Best base R2={best_base_r2:.4f}")
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels([_label(f)[:22] for f in eng_top["feature"]], fontsize=7)
    ax2.set_xlabel("Univariate R2")
    ax2.set_title("Derived vs Best Base Feature")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _fig_cv_results(
    pdf: PdfPages, position: str, cv_result: dict, target: str
) -> None:
    yr = cv_result.get("year_results", [])
    if not yr:
        return

    overall_r2 = cv_result.get("overall_r2", np.nan)
    rmse       = cv_result.get("overall_rmse", np.nan)
    mae        = cv_result.get("overall_mae", np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
    fig.suptitle(
        f"{position} — Leave-One-Year-Out Cross-Validation vs {target}\n"
        f"Pooled: R2={overall_r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}",
        fontsize=12, fontweight="bold"
    )

    ax = axes[0]
    years = [r["year"] for r in yr]
    r2s   = [r.get("r2_test", np.nan) for r in yr]
    ns    = [r.get("n_test", 0)       for r in yr]
    c_cv  = ["#4472C4" if (not np.isnan(v) and v >= 0) else "#E74C3C" for v in r2s]
    bars  = ax.bar(years, r2s, color=c_cv, alpha=0.8)
    if not np.isnan(overall_r2):
        ax.axhline(overall_r2, color="red", linestyle="--", linewidth=1.2,
                   label=f"Pooled R2={overall_r2:.4f}")
    ax.set_xlabel("Draft Year (held out)")
    ax.set_ylabel("Test-set R2")
    ax.set_title("Per-Year CV R2  (Blue=positive, Red=negative)")
    ax.legend(fontsize=8)
    for bar, n_yr, v in zip(bars, ns, r2s):
        offset = (v or 0) + 0.01 if not np.isnan(v or 0) else 0.01
        note   = f"n={n_yr}" + ("*" if n_yr < 10 else "")
        c_note = "#C00" if n_yr < 10 else "#333"
        ax.text(bar.get_x() + bar.get_width() / 2, offset, note,
                ha="center", fontsize=6.5, color=c_note)

    ax2 = axes[1]
    actual    = cv_result.get("all_actual", [])
    predicted = cv_result.get("all_predicted", [])
    all_yrs   = cv_result.get("all_years", [])
    if actual and predicted:
        c_pts = [_era_color(y) for y in all_yrs]
        ax2.scatter(actual, predicted, c=c_pts, alpha=0.45, s=18, linewidths=0)
        lim = max(max(actual), max(predicted)) * 1.05
        ax2.plot([0, lim], [0, lim], "r--", linewidth=0.9, label="Perfect prediction")
        ax2.set_xlabel(f"Actual {target}")
        ax2.set_ylabel(f"Predicted {target}")
        ax2.set_title("Predicted vs Actual (LOYO-CV)")
        ax2.legend(fontsize=8)
        ax2.text(0.04, 0.96, f"R2={overall_r2:.4f}\nRMSE={rmse:.2f}\nMAE={mae:.2f}",
                 ha="left", va="top", transform=ax2.transAxes, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _fig_hit_rate(
    pdf: PdfPages, position: str, hit_result: dict, target: str, threshold: float
) -> None:
    feat_df = hit_result.get("feature_results", pd.DataFrame())
    if feat_df.empty:
        return
    top = feat_df.dropna(subset=["delta"]).head(15)

    n_hits  = hit_result.get("n_hits", 0)
    n_total = hit_result.get("n_total", 0)
    hr      = hit_result.get("hit_rate", 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 8.5), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.suptitle(
        f"{position} — Binary Hit-Rate Analysis (threshold={threshold})\n"
        f"Hits: {n_hits}/{n_total} ({hr:.1%} base rate)",
        fontsize=12, fontweight="bold"
    )

    ax = axes[0]
    ax.axis("off")
    table_data = []
    for _, row in top.iterrows():
        ha = row.get("hit_rate_above_median", np.nan)
        hb = row.get("hit_rate_below_median", np.nan)
        d  = row.get("delta", np.nan)
        pv = row.get("p_value", np.nan)
        table_data.append([
            _label(row["feature"])[:28],
            f"{ha:.1%}" if not np.isnan(ha) else "--",
            f"{hb:.1%}" if not np.isnan(hb) else "--",
            f"{d:+.3f}" if not np.isnan(d)  else "--",
            f"{pv:.3f}" if not np.isnan(pv) else "--",
        ])
    tbl = ax.table(
        cellText=table_data,
        colLabels=["Feature", "Hit % >Med", "Hit % <Med", "Delta", "p-val"],
        loc="upper center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    for i, (_, row) in enumerate(top.iterrows()):
        d  = row.get("delta", 0)
        pv = row.get("p_value", 1.0)
        bg = "#C6EFCE" if (not np.isnan(d) and d > 0.10 and not np.isnan(pv) and pv < 0.05) \
             else "#FFEB9C" if (not np.isnan(d) and d > 0.05) else "white"
        for j in range(5):
            tbl[(i + 1, j)].set_facecolor(bg)

    ax2 = axes[1]
    y_pos   = range(len(top) - 1, -1, -1)
    c_hit   = ["#27AE60" if d > 0 else "#E74C3C" for d in top["delta"]]
    ax2.barh(list(y_pos), top["delta"].values, color=c_hit, alpha=0.85, height=0.7)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels([_label(f)[:22] for f in top["feature"]], fontsize=7)
    ax2.set_xlabel("Hit Rate Delta (above - below median)")
    ax2.set_title("Best Discriminating Features")
    ax2.text(0.98, 0.02, f"Threshold: {target} >= {threshold}",
             ha="right", va="bottom", transform=ax2.transAxes, fontsize=8)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _fig_missing_inventory(
    pdf: PdfPages, positions: list, missing_by_pos: dict
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle("Missing Data Inventory & Improvement Roadmap",
                 fontsize=13, fontweight="bold", y=0.99)

    ax_l = fig.add_axes([0.04, 0.54, 0.44, 0.41])
    ax_l.axis("off")
    narrative = (
        "DATA GAPS — WHAT IS MISSING\n\n"
        "1. recruit_rating (~80% missing)\n"
        "   Blocked: CFBD monthly API quota exhausted 2026-02-22.\n"
        "   Fix: resume in March 2026 with populate_db.py\n"
        "        --recruit-years 2011 2012 2013 2014 2015 2016 2017\n\n"
        "2. consensus_rank (~41% missing)\n"
        "   Structural: no free source exists pre-2016.\n"
        "   Fix: accept gap, or scrape archived ESPN boards manually.\n\n"
        "3. speed_score / forty_time (~21-25% missing)\n"
        "   Cause: combine non-attenders (NOT random — lower-upside\n"
        "   players more likely to skip). Consider indicator variable.\n\n"
        "4. best_ppa_pass / best_usage (early CFBD gap)\n"
        "   Cause: CFBD did not track PPA/usage in 2011-2013 seasons.\n\n"
        "5. College routes run (YPRR) — entirely unavailable free\n"
        "   career_rec_per_target is a weak proxy only.\n"
        "   Fix: PFF+ (~$200/yr) for true YPRR back to 2014.\n\n"
        "6. NFL-level metrics — permanently unavailable free:\n"
        "   Air yards, snap share, ADOT, deep targets, YAC,\n"
        "   target separation: NFL tracking/charting data.\n"
        "   Juke rate, yards created, evaded tackles, explosive\n"
        "   rating, breakaway runs: PlayerProfiler paid (~$20/mo).\n"
        "   Contested catch rating: PFF+ paid.\n\n"
        "7. A.J. Brown (WR 2019, b2s=15.80) and ~5 others missing\n"
        "   Wrong link in CFBLink. Needs manual cfb_player_id fix.\n\n"
        "8. 2026 combine data not yet ingested\n"
        "   Fix: populate_nfl.py --combine-years 2026 after combine."
    )
    ax_l.text(0, 1, narrative, va="top", ha="left", fontsize=7.5,
              fontfamily="monospace", transform=ax_l.transAxes)

    ax_r = fig.add_axes([0.52, 0.54, 0.44, 0.41])
    ax_r.axis("off")
    recommendations = (
        "WHAT WOULD IMPROVE THE ANALYSIS\n\n"
        "HIGH IMPACT:\n"
        "  * recruit_rating 2011-2017 (resume CFBD March 2026)\n"
        "    -> coverage from ~6% to ~100%\n"
        "    -> could be top-3 predictor for WR\n\n"
        "MEDIUM IMPACT:\n"
        "  * Pre-2016 consensus boards (manual ESPN scrape)\n"
        "    -> consensus_rank from ~59% to ~100%\n"
        "  * 2026 NFL combine data (ingest before scoring class)\n"
        "    -> speed_score for 2026 draft prospects\n\n"
        "LOW IMPACT / OPTIONAL:\n"
        "  * PFF+ YPRR data (~$200/yr)\n"
        "    -> Particularly improves TE model\n"
        "  * Manual CFBLink fixes (A.J. Brown + ~5 others)\n"
        "    -> Restores star players to WR training set\n"
        "  * PlayerProfiler athleticism tiers (~$20/mo)\n"
        "    -> Juke rate, yards created, breakaway runs\n"
        "    -> Could add meaningful signal for RBs\n"
        "  * PFF+ (~$200/yr) for college YPRR + contested catch\n"
        "    -> Strong TE signal expected from YPRR\n\n"
        "FUTURE MODEL ENHANCEMENTS:\n"
        "  * Format-specific B2S targets (te_premium_1.0)\n"
        "  * Penalty-adjusted PPG (fumbles_lost, INTs)\n"
        "  * NFL aging curves (B2S window extension to year 4-5)\n"
        "  * Team offensive context features (OC, QB quality)\n"
        "  * Non-linear model (LightGBM) with monotonic constraints"
    )
    ax_r.text(0, 1, recommendations, va="top", ha="left", fontsize=7.5,
              fontfamily="monospace", transform=ax_r.transAxes)

    # Bottom: combined missing table
    ax_t = fig.add_axes([0.04, 0.04, 0.92, 0.46])
    ax_t.axis("off")
    all_rows = []
    for pos in positions:
        ms = missing_by_pos.get(pos, pd.DataFrame())
        if ms.empty:
            continue
        for _, row in ms[ms["pct_missing"] > 5].head(6).iterrows():
            cause = _MISSING_CAUSES.get(row["feature"], "--")
            all_rows.append([
                pos,
                _label(row["feature"])[:26],
                str(int(row["n_available"])),
                f"{row['pct_missing']:.0f}%",
                row["status"],
                cause[:52],
            ])

    if all_rows:
        tbl = ax_t.table(
            cellText=all_rows,
            colLabels=["Pos", "Feature", "N Available", "% Missing", "Status", "Cause"],
            loc="upper center", cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1, 1.3)
        for i, row in enumerate(all_rows):
            status = row[4]
            bg = "#C6EFCE" if status == "OK" else "#FFEB9C" if status == "WARN" else "#FFC7CE"
            for j in range(6):
                tbl[(i + 1, j)].set_facecolor(bg)

    pdf.savefig(fig)
    plt.close(fig)


# =============================================================================
# TIER 4 — CSV Export and Orchestration
# =============================================================================

def _export_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"  CSV: {path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dynasty Prospect Model — Regression Analysis + PDF Report"
    )
    p.add_argument("--position", choices=list(_POSITIONS), default=None,
                   help="Single position to analyze (default: all three)")
    p.add_argument("--start-year", type=int, default=2014,
                   help="First draft class year (default: 2011)")
    p.add_argument("--end-year", type=int, default=None,
                   help="Last draft class year (default: 2022)")
    p.add_argument("--target", choices=list(_TARGETS), default=_DEFAULT_TARGET,
                   help=f"Target variable (default: {_DEFAULT_TARGET})")
    p.add_argument("--hit-threshold", type=float, default=None,
                   help="PPG threshold for binary hit-rate analysis (e.g. 9.0)")
    p.add_argument("--output", type=str, default=None,
                   help="Output PDF path (auto-generated timestamp if not set)")
    p.add_argument("--no-csv", action="store_true",
                   help="Suppress CSV exports")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    positions  = [args.position] if args.position else list(_POSITIONS)
    start_year = args.start_year
    end_year   = args.end_year
    target     = args.target
    yr_tag     = f"{start_year or 2011}_{end_year or 2022}"
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_display = datetime.now().strftime("%Y-%m-%d %H:%M")

    out_path = (
        Path(args.output) if args.output
        else _ROOT / "output" / "reports" / f"analysis_{yr_tag}_{ts}.pdf"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    (_ROOT / "output" / "analysis").mkdir(parents=True, exist_ok=True)

    log.info(f"Dynasty Prospect Model — Regression Analysis")
    log.info(f"Positions: {positions}  |  Target: {target}  |  Years: {yr_tag}")
    log.info(f"Output: {out_path}")

    data_dir = _ROOT / "data"
    results  = {}

    for pos in positions:
        log.info(f"\n{'='*52}\nAnalyzing {pos}...")
        df_raw = load_training(pos, data_dir, start_year, end_year)
        df     = engineer_features(df_raw)
        base_feats, eng_feats = get_feature_lists(df)
        all_feats = base_feats + eng_feats

        missing = compute_missing_data_summary(df, all_feats)

        log.info(f"  Univariate R2 ({len(all_feats)} features, 500-rep bootstrap)...")
        r2_all  = compute_univariate_r2(df, all_feats, target, n_bootstrap=500)
        r2_base = r2_all[r2_all["feature"].isin(base_feats)].copy()
        r2_eng  = r2_all[r2_all["feature"].isin(eng_feats)].copy()

        top25       = r2_all.dropna(subset=["r2"]).head(25)["feature"].tolist()
        corr_matrix = compute_correlation_matrix(df, top25)

        log.info("  Multivariate OLS + VIF + Lasso + Ridge...")
        X, y, used_feats = prepare_xy(df, base_feats, target)
        ols_result   = fit_ols_full(X, y)
        vif_df       = compute_vif(X)
        lasso_result = fit_lasso(X, y)
        ridge_result = fit_ridge(X, y)

        log.info("  Leave-one-year-out CV...")
        cv_result = leave_one_year_out_cv(df, base_feats, target)

        top5_base   = r2_base.dropna(subset=["r2"]).head(5)["feature"].tolist()
        log.info("  Era sensitivity...")
        era_results = compute_era_sensitivity(df, top5_base, target)

        hit_result = None
        if args.hit_threshold is not None:
            log.info(f"  Hit-rate analysis (threshold={args.hit_threshold})...")
            hit_result = compute_hit_rate(df, target, args.hit_threshold)

        log.info(
            f"  Done. OLS R2={ols_result.get('r2', 0):.4f}  "
            f"LOYO R2={cv_result.get('overall_r2', 0):.4f}"
        )
        results[pos] = {
            "df": df, "missing": missing,
            "r2_all": r2_all, "r2_base": r2_base, "r2_eng": r2_eng,
            "corr_matrix": corr_matrix,
            "ols": ols_result, "vif": vif_df,
            "lasso": lasso_result, "ridge": ridge_result,
            "cv": cv_result, "era": era_results,
            "top5_base": top5_base, "hit": hit_result,
        }

    # --- Render PDF ---
    log.info(f"\nRendering PDF ({len(positions)} positions)...")
    page_count = 0
    with PdfPages(out_path) as pdf:
        _fig_cover(pdf, positions, target, start_year, end_year, ts_display)
        page_count += 1

        for pos in positions:
            r = results[pos]
            log.info(f"  {pos}: rendering pages...")
            _fig_data_overview(pdf, pos, r["df"], target, r["missing"]);       page_count += 1
            _fig_univariate_table(pdf, pos, r["r2_all"], target);              page_count += 1
            _fig_scatter_grid(pdf, pos, r["df"], r["r2_all"], target);         page_count += 1
            _fig_correlation_heatmap(pdf, pos, r["corr_matrix"]);              page_count += 1
            _fig_ols_summary(pdf, pos, r["ols"], r["vif"], target);            page_count += 1
            _fig_regularized(pdf, pos, r["lasso"], r["ridge"], target);        page_count += 1
            _fig_jj_benchmark(pdf, pos, r["r2_all"], target);                  page_count += 1
            _fig_era_sensitivity(pdf, pos, r["era"], r["top5_base"], target);  page_count += 1
            _fig_feature_engineering(pdf, pos, r["r2_eng"], r["r2_base"], target); page_count += 1
            _fig_cv_results(pdf, pos, r["cv"], target);                        page_count += 1
            if r["hit"] is not None:
                _fig_hit_rate(pdf, pos, r["hit"], target, args.hit_threshold); page_count += 1

        _fig_missing_inventory(
            pdf, positions, {p: results[p]["missing"] for p in positions}
        )
        page_count += 1

    log.info(f"PDF written: {out_path}  ({page_count} pages)")

    # --- Export CSVs ---
    if not args.no_csv:
        for pos in positions:
            r = results[pos]
            _export_csv(
                r["r2_all"],
                _ROOT / "output" / "analysis" / f"univariate_{pos}_{yr_tag}.csv",
            )
            _export_csv(
                r["r2_eng"],
                _ROOT / "output" / "analysis" / f"feature_engineering_{pos}_{yr_tag}.csv",
            )

    # --- Summary ---
    print(f"\n{'=' * 62}")
    print(f"ANALYSIS COMPLETE — target: {target}  |  era: {yr_tag}")
    print(f"{'=' * 62}")
    for pos in positions:
        r     = results[pos]
        or2   = r["ols"].get("r2", np.nan)
        cvr2  = r["cv"].get("overall_r2", np.nan)
        top1  = r["r2_all"].iloc[0]["feature"] if len(r["r2_all"]) else "--"
        top1r = r["r2_all"].iloc[0]["r2"]      if len(r["r2_all"]) else np.nan
        print(f"  {pos}: OLS R2={or2:.4f}  LOYO R2={cvr2:.4f}  "
              f"Top feature: {top1} (R2={top1r:.4f})")
    print(f"\nReport : {out_path}")
    print(f"Pages  : {page_count}")


if __name__ == "__main__":
    main()
