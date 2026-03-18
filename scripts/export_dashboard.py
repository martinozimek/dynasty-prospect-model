"""
export_dashboard.py  —  Dynasty Prospect Model Dashboard Generator
Reads scored CSVs + training data + model metadata, writes a self-contained HTML dashboard.

Usage:
    python scripts/export_dashboard.py                    # default: 2026
    python scripts/export_dashboard.py --year 2025
    python scripts/export_dashboard.py --all-years
    python scripts/export_dashboard.py --year 2026 --include-historical
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

# Allow importing analyze.py (shared transformers + engineer_features)
sys.path.insert(0, str(Path(__file__).parent))
try:
    import joblib
    from analyze import engineer_features, QuantileClipper  # noqa: F401 — needed for pickle resolution
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SCORES_DIR = ROOT / "output" / "scores"
DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"
DASH_DIR   = ROOT / "dashboard"

# ── Feature configuration ─────────────────────────────────────────────────────
FEATURE_LABELS = {
    "best_age": "Draft Age",
    "log_draft_capital": "Draft Capital",
    "draft_capital_score": "Draft Capital Score",
    "draft_tier": "Draft Tier",
    "overall_pick": "Draft Pick #",
    "breakout_score_x_capital": "Breakout × Capital",
    "capital_x_age": "Capital × Age",
    "early_declare": "Early Declare",
    "best_routes_per_game": "Routes / Game",
    "best_man_yprr": "Man Coverage YPRR",
    "best_zone_yprr": "Zone Coverage YPRR",
    "best_man_zone_delta": "Man–Zone Split",
    "best_slot_yprr": "Slot YPRR",
    "best_deep_yprr": "Deep YPRR",
    "best_yprr": "Overall YPRR",
    "best_breakout_score": "Breakout Score",
    "best_dominator": "Dominator Rating",
    "best_rec_rate": "Reception Rate",
    "best_receiving_grade": "PFF Receiving Grade",
    "best_drop_rate": "Drop Rate",
    "best_deep_target_rate": "Deep Target Rate",
    "best_slot_target_rate": "Slot Target Rate",
    "speed_score": "Speed Score",
    "vertical_jump": "Vertical Jump",
    "broad_jump": "Broad Jump",
    "forty_time": "40-Yard Dash",
    "agility_score": "Agility Score",
    "weight_lbs": "Weight",
    "height_inches": "Height",
    "best_rush_ypc": "Rush YPC",
    "best_usage_pass": "Pass Usage %",
    "best_total_yards_rate": "Total Yards Rate",
    "college_fantasy_ppg": "College Fantasy PPG",
    "position_rank": "Position Rank",
    "consensus_rank": "Consensus Big Board",
    "slot_rate_x_capital": "Slot Rate × Capital",
    "total_yards_rate_x_capital": "Total Yards × Capital",
    "capital_x_dominator": "Capital × Dominator",
    "combined_ath_x_capital": "Athleticism × Capital",
    "draft_premium": "Draft Premium",
    "recruit_rating": "Recruit Rating",
    "power4_conf": "Power 4 Conference",
    # Phase C features (2026-03-16)
    "log_pick_capital": "JJ Pick Capital (log)",
    "pick_capital_x_age": "Pick Capital × Age",
    "total_yards_rate_x_pick_capital": "Total Yards Rate × Pick Cap",
    "teammate_score_x_capital": "Teammate Score × Capital",
    "blended_capital_rb": "Blended Capital (RB 83/17)",
    "log_blended_capital": "Blended Capital (log)",
    "best_total_yards_rate_v2": "Total Yards Rate v2 (total plays)",
}

# Features where LOWER = BETTER (invert percentile)
INVERSE_FEATURES = {
    "best_age", "forty_time", "overall_pick", "position_rank",
    "consensus_rank", "best_drop_rate", "agility_score", "three_cone", "shuttle"
}

# Binary features (show Yes/No, no percentile bar)
BINARY_FEATURES = {"early_declare", "power4_conf"}

# All feature columns to extract from score CSVs
BEST_FEATURE_COLS = [
    "best_dominator", "best_rec_rate", "college_fantasy_ppg", "breakout_age",
    "best_age", "best_breakout_score", "best_total_yards_rate", "best_usage_pass",
    "best_usage_rush", "best_rush_ypc", "early_declare", "power4_conf",
    "best_yprr", "best_receiving_grade", "best_routes_per_game", "best_drop_rate",
    "best_target_sep", "best_contested_catch_rate", "best_deep_yprr",
    "best_deep_target_rate", "best_behind_los_rate", "best_slot_yprr",
    "best_slot_target_rate", "best_screen_rate", "best_man_yprr",
    "best_zone_yprr", "best_man_zone_delta", "weight_lbs", "height_inches",
    "speed_score", "forty_time", "broad_jump", "vertical_jump", "three_cone",
    "shuttle", "bench_press", "recruit_rating", "recruit_stars",
    "recruit_rank_national", "consensus_rank", "position_rank",
    "draft_capital_score", "overall_pick",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_float(v):
    """Return float or None for NaN/missing."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def compute_pct(value, distribution, inverse=False):
    """Percentile of value in distribution. Returns None if not computable."""
    if value is None:
        return None
    clean = [x for x in distribution if x is not None and not math.isnan(x)]
    if not clean:
        return None
    pct = percentileofscore(clean, value, kind="rank")
    if inverse:
        pct = 100.0 - pct
    return round(pct, 1)


def load_metadata(path: Path) -> dict:
    """Load metadata JSON, handling NaN as null."""
    text = path.read_text(encoding="utf-8")
    # JSON doesn't support NaN — replace bare NaN tokens
    text = text.replace(": NaN", ": null").replace(":NaN", ":null")
    return json.loads(text)


def load_training(pos: str) -> pd.DataFrame:
    path = DATA_DIR / f"training_{pos}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def load_scores(year: int) -> pd.DataFrame | None:
    path = SCORES_DIR / f"scores_{year}_ridge.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def build_feature_distributions(training_dfs: dict) -> dict:
    """For each position, for each feature, collect list of non-null values."""
    dists = {}
    for pos, df in training_dfs.items():
        dists[pos] = {}
        for col in BEST_FEATURE_COLS:
            if col in df.columns:
                vals = df[col].dropna().tolist()
                # Convert to plain Python floats / remove non-numeric
                clean = []
                for v in vals:
                    f = safe_float(v)
                    if f is not None:
                        clean.append(f)
                dists[pos][col] = clean
            else:
                dists[pos][col] = []
    return dists


def build_feature_percentiles(dists: dict) -> dict:
    """Compute p10/p25/p50/p75/p90 for each feature in each position."""
    percs = {}
    for pos, fd in dists.items():
        percs[pos] = {}
        for feat, vals in fd.items():
            if len(vals) < 5:
                continue
            a = np.array(vals)
            percs[pos][feat] = {
                "p10": round(float(np.percentile(a, 10)), 4),
                "p25": round(float(np.percentile(a, 25)), 4),
                "p50": round(float(np.percentile(a, 50)), 4),
                "p75": round(float(np.percentile(a, 75)), 4),
                "p90": round(float(np.percentile(a, 90)), 4),
            }
    return percs


def row_to_player(row: pd.Series, pos: str, dists: dict, year: int) -> dict:
    """Convert a scored CSV row to a player dict for JSON embedding."""
    def g(col, default=None):
        if col not in row.index:
            return default
        v = row[col]
        if pd.isna(v) if not isinstance(v, str) else False:
            return default
        return v

    # Feature values + percentiles
    features = {}
    pos_dist = dists.get(pos, {})
    for col in BEST_FEATURE_COLS:
        val = safe_float(g(col))
        if col in BINARY_FEATURES:
            # Store as 0/1 integer
            if val is not None:
                features[col] = int(val)
            else:
                raw = g(col)
                features[col] = (1 if str(raw).strip().lower() in ("1", "true", "yes") else 0) if raw is not None else None
        else:
            if val is not None:
                dist_vals = pos_dist.get(col, [])
                inverse = col in INVERSE_FEATURES
                pct = compute_pct(val, dist_vals, inverse=inverse)
                features[col] = {"value": round(val, 4), "pct": pct}
            else:
                features[col] = {"value": None, "pct": None}

    # Parse seasons_json for college career table
    raw_sj = g("seasons_json")
    seasons = []
    if raw_sj and isinstance(raw_sj, str):
        try:
            seasons = json.loads(raw_sj)
        except Exception:
            pass

    player = {
        "name": str(g("player_name", "")),
        "college": str(g("most_recent_team") or g("best_team", "")),
        "position": pos,
        "draft_year": year,
        "best_season_year": safe_float(g("best_season_year")),
        "pos_rank": safe_float(g("pos_rank")),
        "height_inches": safe_float(g("height_inches")),
        "weight_lbs": safe_float(g("weight_lbs")),
        "forty_time": safe_float(g("forty_time")),
        "speed_score": safe_float(g("speed_score")),
        "broad_jump": safe_float(g("broad_jump")),
        "vertical_jump": safe_float(g("vertical_jump")),
        "three_cone": safe_float(g("three_cone")),
        "shuttle": safe_float(g("shuttle")),
        "bench_press": safe_float(g("bench_press")),
        "recruit_stars": safe_float(g("recruit_stars")),
        "recruit_rank_national": safe_float(g("recruit_rank_national")),
        "consensus_rank": safe_float(g("consensus_rank")),
        "position_rank": safe_float(g("position_rank")),
        "overall_pick": safe_float(g("overall_pick")),
        "capital_is_projected": bool(g("capital_is_projected", True)),
        "draft_capital_score": safe_float(g("draft_capital_score")),
        "projected_b2s": safe_float(g("projected_b2s")),
        "orbit_score": safe_float(g("orbit_score")),
        "b2s_lo80": safe_float(g("b2s_lo80")),
        "b2s_hi80": safe_float(g("b2s_hi80")),
        "b2s_lo90": safe_float(g("b2s_lo90")),
        "b2s_hi90": safe_float(g("b2s_hi90")),
        "model": str(g("model", "ridge")),
        "post_draft": bool(g("post_draft", False)),
        "phase1_orbit": safe_float(g("phase1_orbit")),
        "capital_delta": safe_float(g("capital_delta")),
        "risk": str(g("risk", "")) if g("risk") is not None else None,
        "b2s_score": None,  # future class — no actual outcome
        # Top-level convenience fields for table sorting (also in features dict)
        "best_age": safe_float(g("best_age")),
        "best_breakout_score": safe_float(g("best_breakout_score")),
        # Data quality flags — warn user when key Phase I features are imputed
        "age_imputed": safe_float(g("best_age")) is None or safe_float(g("best_age")) == 18.5,
        "breakout_imputed": safe_float(g("best_breakout_score")) is None,
        "age_suspect": (safe_float(g("best_age")) or 0) > 26 or safe_float(g("best_age")) == 18.5,
        "features": features,
        "seasons": seasons,
    }
    return player


def _load_hist_models() -> dict:
    """Load fitted sklearn Pipelines for historical ORBIT computation. Returns {} on failure."""
    if not _MODELS_AVAILABLE:
        return {}
    models = {}
    for pos in ("WR", "RB", "TE"):
        cap_path = MODELS_DIR / f"{pos}_ridge.pkl"
        nc_path  = MODELS_DIR / f"{pos}_ridge_nocap.pkl"
        try:
            models[pos] = {
                "cap":   joblib.load(cap_path)   if cap_path.exists()  else None,
                "nocap": joblib.load(nc_path)    if nc_path.exists()   else None,
            }
        except Exception as exc:
            print(f"  [warn] Could not load {pos} models: {exc}")
    return models


def _hist_orbit(df: pd.DataFrame, pipe, feat_list: list, train_preds: list) -> pd.Series:
    """Run historical training rows through a fitted pipeline → ORBIT series."""
    if pipe is None or not feat_list or not train_preds:
        return pd.Series([None] * len(df))
    try:
        X = df[feat_list].values.astype(float)
        preds = pipe.predict(X)
        orbits = [
            min(100.0, round(float(percentileofscore(train_preds, p, kind="rank")), 1))
            for p in preds
        ]
        return pd.Series(orbits, index=df.index)
    except Exception as exc:
        print(f"  [warn] Historical ORBIT computation failed: {exc}")
        return pd.Series([None] * len(df))


def build_historical(training_dfs: dict, dists: dict, meta: dict, meta_nocap: dict) -> dict:
    """Build historical player records from training CSVs (2011-2022, known outcomes)."""
    hist_models = _load_hist_models()

    hist = {}
    for pos, df in training_dfs.items():
        if df.empty:
            hist[pos] = []
            continue

        # ── Engineer capital features (log_draft_capital, interaction terms) ──
        df_eng = df.copy()
        if _MODELS_AVAILABLE:
            try:
                df_eng = engineer_features(df_eng)
            except Exception as exc:
                print(f"  [warn] engineer_features failed for {pos}: {exc}")

        # ── Compute ORBIT scores in bulk ──────────────────────────────────────
        pos_models = hist_models.get(pos, {})
        cap_feats   = meta.get(pos, {}).get("selected_features", [])
        nocap_feats = meta_nocap.get(pos, {}).get("selected_features", [])
        cap_train   = meta.get(pos, {}).get("train_preds", [])
        nocap_train = meta_nocap.get(pos, {}).get("train_preds", [])

        orbit_series   = _hist_orbit(df_eng, pos_models.get("cap"),   cap_feats,   cap_train)
        phase1_series  = _hist_orbit(df_eng, pos_models.get("nocap"), nocap_feats, nocap_train)

        records = []
        for i, (_, row) in enumerate(df.iterrows()):
            def g(col, default=None, _row=row):
                if col not in _row.index:
                    return default
                v = _row[col]
                if pd.isna(v) if not isinstance(v, str) else False:
                    return default
                return v

            features = {}
            for col in BEST_FEATURE_COLS:
                val = safe_float(g(col))
                if col in BINARY_FEATURES:
                    features[col] = int(val) if val is not None else None
                else:
                    if val is not None:
                        dist_vals = dists.get(pos, {}).get(col, [])
                        inverse = col in INVERSE_FEATURES
                        pct = compute_pct(val, dist_vals, inverse=inverse)
                        features[col] = {"value": round(val, 4), "pct": pct}
                    else:
                        features[col] = {"value": None, "pct": None}

            draft_cap = safe_float(g("draft_capital_score"))
            b2s = safe_float(g("b2s_score"))
            orb = orbit_series.iloc[i] if i < len(orbit_series) else None
            p1  = phase1_series.iloc[i] if i < len(phase1_series) else None

            records.append({
                "name": str(g("nfl_name") or g("cfb_name") or ""),
                "college": str(g("best_team", "")),
                "position": pos,
                "draft_year": int(g("draft_year", 0)) if g("draft_year") is not None else None,
                "height_inches": safe_float(g("height_inches")),
                "weight_lbs": safe_float(g("weight_lbs")),
                "forty_time": safe_float(g("forty_time")),
                "speed_score": safe_float(g("speed_score")),
                "broad_jump": safe_float(g("broad_jump")),
                "vertical_jump": safe_float(g("vertical_jump")),
                "overall_pick": safe_float(g("overall_pick")),
                "draft_capital_score": draft_cap,
                "b2s_score": b2s,
                "orbit_score": safe_float(orb),
                "phase1_orbit": safe_float(p1),
                "capital_delta": round(float(orb) - float(p1), 1) if orb is not None and p1 is not None else None,
                "features": features,
            })
        hist[pos] = records
    return hist


def build_season_data() -> dict:
    """Load per-season production data from cfb-prospect-db for the year-out chart.
    Returns dict keyed by position with background dots, elite avg line, and prospect seasons.
    """
    import sqlite3
    db = ROOT.parent / "cfb-prospect-db" / "ff.db"
    if not db.exists():
        return {}
    con = sqlite3.connect(str(db))
    df = pd.read_sql("""
        SELECT
            s.player_id,
            p.full_name,
            p.position,
            p.declared_draft_year,
            s.season_year,
            s.games_played,
            s.rec_yards_per_team_pass_att,
            s.dominator_rating
        FROM cfb_player_seasons s
        JOIN players p ON p.id = s.player_id
        LEFT JOIN (
            SELECT player_id, MIN(recruit_year) AS recruit_year
            FROM recruiting GROUP BY player_id
        ) r ON r.player_id = s.player_id
        WHERE p.position IN ('WR','RB','TE')
          AND s.games_played >= 6
    """, con)
    picks = pd.read_sql(
        "SELECT player_id FROM nfl_draft_picks WHERE draft_round <= 2", con
    )
    con.close()

    elite_ids = set(picks["player_id"])
    min_sy = df.groupby("player_id")["season_year"].min().rename("min_sy")
    df = df.join(min_sy, on="player_id")
    df["year_out"] = (df["season_year"] - df["min_sy"] + 1).astype(int)
    df = df[(df["year_out"] >= 1) & (df["year_out"] <= 5)]
    df["is_elite"] = df["player_id"].isin(elite_ids)

    Y_LIMS = {
        "WR": ("rec_yards_per_team_pass_att", "Rec Yds / Team Pass Att", 0.0, 4.0),
        "RB": ("dominator_rating",            "Dominator Rating",         0.0, 0.25),
        "TE": ("rec_yards_per_team_pass_att", "Rec Yds / Team Pass Att", 0.0, 2.5),
    }
    PROSPECT_YEARS = [2024, 2025, 2026]

    result = {}
    for pos, (metric, label, y_lo, y_hi) in Y_LIMS.items():
        pos_df = df[(df["position"] == pos)].dropna(subset=[metric]).copy()
        pos_df = pos_df[(pos_df[metric] >= y_lo) & (pos_df[metric] <= y_hi)]

        # Elite average per year_out (R1-R2 picks only, from non-prospect years)
        bg_df = pos_df[~pos_df["declared_draft_year"].isin(PROSPECT_YEARS)]
        elite_avg = {}
        for yo in range(1, 6):
            sub = bg_df[(bg_df["year_out"] == yo) & bg_df["is_elite"]]
            if len(sub) >= 3:
                elite_avg[str(yo)] = round(float(sub[metric].mean()), 4)

        # Background dots (non-prospect years only, sampled to ≤5k points to keep HTML small)
        bg_pts = [
            {"x": int(r["year_out"]), "y": round(float(r[metric]), 4)}
            for _, r in bg_df.iterrows()
        ]
        if len(bg_pts) > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(bg_pts), 5000, replace=False)
            bg_pts = [bg_pts[i] for i in sorted(idx)]

        # Prospect seasons per year (named, for highlighting)
        prospects = {}
        for yr in PROSPECT_YEARS:
            yr_df = pos_df[pos_df["declared_draft_year"] == yr]
            pts = []
            # Aggregate to best season per player to avoid clutter
            for pid, grp in yr_df.groupby("player_id"):
                best = grp.loc[grp[metric].idxmax()]
                pts.append({
                    "x": int(best["year_out"]),
                    "y": round(float(best[metric]), 4),
                    "name": str(best["full_name"]),
                })
            prospects[str(yr)] = pts

        result[pos] = {
            "metric_label": label,
            "y_max": y_hi,
            "bg": bg_pts,
            "elite_avg": elite_avg,
            "prospects": prospects,
        }
    return result


def build_model_perf(meta: dict, meta_nocap: dict) -> dict:
    """Build model performance summary for each position."""
    perf = {}
    for pos in ("WR", "RB", "TE"):
        m = meta.get(pos, {})
        mn = meta_nocap.get(pos, {})
        perf[pos] = {
            "loyo_r2": m.get("ridge_loyo_r2"),
            "spearman_rho": m.get("loyo_spearman_rho"),
            "top25_hit_rate": m.get("loyo_top25_hit_rate"),
            "n_train": m.get("n_train"),
            "selected_features": m.get("selected_features", []),
            "nocap_features": mn.get("selected_features", []),
            "ridge_coefs": {
                f["feature"]: round(f["coef"], 4)
                for f in m.get("ridge_top_features", [])
            },
            "nocap_ridge_coefs": {
                f["feature"]: round(f["coef"], 4)
                for f in mn.get("ridge_top_features", [])
            },
            "loyo_by_year": [
                {"year": y["year"], "r2": round(y["r2"], 3), "n": y["n_test"]}
                for y in m.get("ridge_loyo_years", [])
            ],
        }
    return perf


# ── Main build ────────────────────────────────────────────────────────────────

def build_dashboard_data(years: list[int], include_historical: bool = False) -> dict:
    """Assemble the full DASHBOARD_DATA object."""

    # Load training sets
    training_dfs = {pos: load_training(pos) for pos in ("WR", "RB", "TE")}

    # Feature distributions (for percentile computation)
    dists = build_feature_distributions(training_dfs)
    feat_pcts = build_feature_percentiles(dists)

    # Load metadata
    meta_path = MODELS_DIR / "metadata.json"
    meta_nocap_path = MODELS_DIR / "metadata_nocap.json"
    meta = load_metadata(meta_path) if meta_path.exists() else {}
    meta_nocap = load_metadata(meta_nocap_path) if meta_nocap_path.exists() else {}

    model_perf = build_model_perf(meta, meta_nocap)

    # Players per year per position
    players_by_year = {}
    for year in years:
        df = load_scores(year)
        if df is None:
            print(f"  [warn] No scores found for {year}, skipping.")
            continue
        year_players = {"WR": [], "RB": [], "TE": []}
        for pos in ("WR", "RB", "TE"):
            subset = df[df["position"] == pos].copy()
            subset = subset.sort_values("pos_rank", na_position="last")
            for _, row in subset.iterrows():
                year_players[pos].append(row_to_player(row, pos, dists, year))
        players_by_year[str(year)] = year_players
        total = sum(len(v) for v in year_players.values())
        print(f"  {year}: {total} players loaded "
              f"(WR={len(year_players['WR'])}, RB={len(year_players['RB'])}, TE={len(year_players['TE'])})")

    # Historical (training set players with outcomes)
    historical = {}
    if include_historical:
        historical = build_historical(training_dfs, dists, meta, meta_nocap)
        for pos, recs in historical.items():
            n_orb = sum(1 for r in recs if r.get("orbit_score") is not None)
            print(f"  Historical {pos}: {len(recs)} records ({n_orb} with ORBIT scores)")

    # Season-level production data for year-out chart
    season_data = build_season_data()

    # Slim down distributions to only features actually used somewhere
    used_features = set()
    for pos_perf in model_perf.values():
        used_features.update(pos_perf["selected_features"])
        used_features.update(pos_perf["nocap_features"])
    # Also include all best_* columns for percentile display
    for col in BEST_FEATURE_COLS:
        used_features.add(col)

    slim_dists = {}
    for pos, fd in dists.items():
        slim_dists[pos] = {
            k: [round(v, 4) for v in vals]
            for k, vals in fd.items()
            if k in used_features and vals
        }

    import datetime
    today = datetime.date.today().isoformat()

    data = {
        "meta": {
            "generated": today,
            "years": [str(y) for y in years],
            "primary_year": str(years[-1]),
            "title": "Dynasty Prospect Model",
        },
        "model_performance": model_perf,
        "feature_labels": FEATURE_LABELS,
        "inverse_features": list(INVERSE_FEATURES),
        "binary_features": list(BINARY_FEATURES),
        "feature_percentiles": feat_pcts,
        "feature_distributions": slim_dists,
        "players": players_by_year,
        "historical": historical,
        "season_data": season_data,
    }
    return data


# ── HTML Template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dynasty Prospect Model — {title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels@2.2.0/dist/chartjs-plugin-datalabels.min.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #0f1117;
  --surface: #1a1d27;
  --surface2: #242736;
  --surface3: #2e3347;
  --border: #2d3148;
  --text: #e2e8f0;
  --text-muted: #94a3b8;
  --text-dim: #64748b;
  --accent: #4f9cf9;
  --accent-dim: #1e3a5f;
  --green: #22c55e;
  --green-dim: #14532d;
  --yellow: #f59e0b;
  --yellow-dim: #451a03;
  --red: #ef4444;
  --red-dim: #450a0a;
  --blue: #3b82f6;
  --blue-dim: #1e3a5f;
  --panel-width: 440px;
  --header-h: 64px;
  --tabs-h: 48px;
}

html, body {
  height: 100%; background: var(--bg); color: var(--text);
  font-family: 'Inter', system-ui, sans-serif; font-size: 14px;
  overflow: hidden;
}

/* ── Layout ── */
#app { display: flex; flex-direction: column; height: 100vh; }

header {
  height: var(--header-h); background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 24px;
  flex-shrink: 0; gap: 12px; position: relative; z-index: 50;
}
.header-logo { font-size: 18px; font-weight: 800; letter-spacing: -0.5px; color: var(--text); }
.header-logo span { color: var(--accent); }
.header-badge {
  font-size: 11px; font-weight: 600; background: var(--accent-dim);
  color: var(--accent); padding: 3px 8px; border-radius: 999px; letter-spacing: 0.5px;
}
.header-spacer { flex: 1; }
.header-meta { font-size: 12px; color: var(--text-dim); }

.tab-bar {
  height: var(--tabs-h); background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 24px; gap: 4px;
  flex-shrink: 0; position: relative; z-index: 40;
}
.tab-btn {
  background: none; border: none; cursor: pointer;
  color: var(--text-muted); font-size: 13px; font-weight: 500;
  padding: 8px 16px; border-radius: 8px;
  transition: all 0.15s ease; font-family: inherit;
}
.tab-btn:hover { background: var(--surface2); color: var(--text); }
.tab-btn.active { background: var(--surface2); color: var(--accent); }

.main-area {
  flex: 1; overflow: hidden; display: flex; position: relative;
}

.tab-content {
  flex: 1; overflow-y: auto; padding: 24px;
  display: none;
}
.tab-content.active { display: block; }

/* ── Controls bar ── */
.controls {
  display: flex; align-items: center; gap: 12px; margin-bottom: 20px; flex-wrap: wrap;
}
.pos-btn-group, .year-btn-group { display: flex; gap: 4px; }
.seg-btn {
  background: var(--surface2); border: 1px solid var(--border);
  color: var(--text-muted); font-size: 12px; font-weight: 600;
  padding: 6px 14px; border-radius: 8px; cursor: pointer;
  transition: all 0.15s; font-family: inherit; letter-spacing: 0.3px;
}
.seg-btn:hover { border-color: var(--accent); color: var(--text); }
.seg-btn.active { background: var(--accent); border-color: var(--accent); color: #fff; }

.controls-spacer { flex: 1; }
.search-box {
  background: var(--surface2); border: 1px solid var(--border);
  color: var(--text); font-size: 13px; font-family: inherit;
  padding: 7px 14px; border-radius: 8px; width: 220px;
  outline: none;
}
.search-box:focus { border-color: var(--accent); }
.search-box::placeholder { color: var(--text-dim); }

/* ── Table ── */
.table-wrap { overflow: auto; border-radius: 12px; border: 1px solid var(--border); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
thead th {
  background: var(--surface2); color: var(--text-muted);
  font-weight: 600; font-size: 11px; letter-spacing: 0.5px; text-transform: uppercase;
  padding: 10px 14px; text-align: left; position: sticky; top: 0; z-index: 2;
  cursor: pointer; user-select: none; white-space: nowrap;
  border-bottom: 1px solid var(--border);
}
thead th:hover { color: var(--text); }
thead th .sort-icon { margin-left: 4px; opacity: 0.5; }
thead th.sorted .sort-icon { opacity: 1; color: var(--accent); }
tbody tr {
  cursor: pointer; border-bottom: 1px solid var(--border);
  transition: background 0.1s;
}
tbody tr:last-child { border-bottom: none; }
tbody tr:hover { background: var(--surface2); }
tbody tr.selected { background: var(--accent-dim); }
tbody td { padding: 10px 14px; color: var(--text); vertical-align: middle; }
tbody td.muted { color: var(--text-muted); }

/* ── ORBIT bar ── */
.orbit-cell { display: flex; align-items: center; gap: 8px; }
.orbit-num {
  font-weight: 700; font-size: 14px; min-width: 32px; text-align: right;
}
.orbit-bar-track {
  flex: 1; max-width: 60px; height: 6px; background: var(--surface3);
  border-radius: 3px; overflow: hidden;
}
.orbit-bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
.orbit-green { color: var(--green); }
.orbit-yellow { color: var(--yellow); }
.orbit-red { color: var(--red); }

/* ── Delta ── */
.delta-pos { color: var(--red); font-weight: 600; }
.delta-neg { color: var(--green); font-weight: 600; }
.delta-neutral { color: var(--text-muted); }

/* ── Risk badge ── */
.risk-badge {
  display: inline-block; font-size: 10px; font-weight: 700;
  padding: 3px 8px; border-radius: 999px; letter-spacing: 0.5px;
  text-transform: uppercase;
}
.risk-low { background: var(--green-dim); color: var(--green); }
.risk-neutral { background: var(--surface3); color: var(--text-muted); }
.risk-high { background: var(--red-dim); color: var(--red); }

/* ── Player panel ── */
#player-panel {
  width: var(--panel-width); background: var(--surface);
  border-left: 1px solid var(--border);
  display: flex; flex-direction: column;
  transform: translateX(var(--panel-width));
  transition: transform 0.3s ease;
  position: relative; flex-shrink: 0;
  overflow: hidden;
}
#player-panel.open { transform: translateX(0); }

.panel-close {
  position: absolute; top: 14px; right: 14px;
  background: var(--surface2); border: 1px solid var(--border);
  color: var(--text-muted); width: 30px; height: 30px;
  border-radius: 50%; cursor: pointer; font-size: 16px;
  display: flex; align-items: center; justify-content: center;
  z-index: 10; transition: all 0.15s;
}
.panel-close:hover { color: var(--text); border-color: var(--text); }

.panel-scroll { overflow-y: auto; flex: 1; padding: 20px; }

.panel-header { margin-bottom: 16px; padding-right: 36px; }
.panel-name { font-size: 20px; font-weight: 800; color: var(--text); line-height: 1.2; }
.panel-sub { font-size: 13px; color: var(--text-muted); margin-top: 4px; }

/* Score cards */
.score-cards {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 8px; margin-bottom: 16px;
}
.score-card {
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px; text-align: center;
}
.score-card-label { font-size: 10px; font-weight: 600; color: var(--text-dim); letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 4px; }
.score-card-val { font-size: 22px; font-weight: 800; line-height: 1; }
.score-card-sub { font-size: 11px; color: var(--text-muted); margin-top: 4px; }

/* CI range */
.ci-bar-wrap { margin: 12px 0; background: var(--surface2); border-radius: 10px; padding: 12px; }
.ci-title { font-size: 10px; font-weight: 600; color: var(--text-dim); letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 8px; }
.ci-row { display: flex; align-items: center; gap: 8px; font-size: 12px; margin-bottom: 6px; }
.ci-label { color: var(--text-muted); min-width: 30px; }
.ci-track { flex: 1; height: 8px; background: var(--surface3); border-radius: 4px; position: relative; }
.ci-range { position: absolute; height: 100%; background: var(--accent); border-radius: 4px; opacity: 0.6; }
.ci-point { position: absolute; top: -2px; width: 12px; height: 12px; background: var(--accent); border-radius: 50%; transform: translateX(-50%); }
.ci-vals { color: var(--text-muted); font-size: 11px; white-space: nowrap; }

/* Physical grid */
.section-title {
  font-size: 11px; font-weight: 700; color: var(--text-dim);
  letter-spacing: 0.8px; text-transform: uppercase;
  margin-bottom: 10px; margin-top: 18px;
  display: flex; align-items: center; gap: 8px;
}
.section-title::after {
  content: ''; flex: 1; height: 1px; background: var(--border);
}
.phys-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
.phys-item {
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px;
}
.phys-label { font-size: 10px; color: var(--text-dim); font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; }
.phys-val { font-size: 16px; font-weight: 700; color: var(--text); margin-top: 3px; }

/* Feature rows */
.feat-list { display: flex; flex-direction: column; gap: 8px; }
.feat-row {
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 12px;
}
.feat-imputed { opacity: 0.6; border-style: dashed; }
.feat-row-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
.feat-label { font-size: 12px; font-weight: 500; color: var(--text); }
.feat-val { font-size: 12px; font-weight: 700; color: var(--text); }
.feat-tier { font-size: 10px; font-weight: 700; letter-spacing: 0.5px; }
.tier-elite { color: var(--green); }
.tier-good { color: var(--blue); }
.tier-avg { color: var(--yellow); }
.tier-below { color: var(--yellow); }
.tier-weak { color: var(--red); }
.tier-na { color: var(--text-dim); }

.pct-track { height: 6px; background: var(--surface3); border-radius: 3px; overflow: hidden; }
.pct-fill { height: 100%; border-radius: 3px; }
.pct-green { background: var(--green); }
.pct-blue { background: var(--blue); }
.pct-amber { background: var(--yellow); }
.pct-red { background: var(--red); }
.pct-gray { background: var(--text-dim); }

/* ── Scatter plot tab ── */
#scatter-canvas-wrap {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px;
  height: calc(100vh - var(--header-h) - var(--tabs-h) - 110px);
}

/* ── Historical tab ── */
.hist-meta { color: var(--text-muted); font-size: 13px; margin-bottom: 16px; }
.outcome-beat { color: var(--green); font-weight: 600; }
.outcome-near { color: var(--text-muted); }
.outcome-miss { color: var(--red); font-weight: 600; }

/* ── About tab ── */
.about-wrap { max-width: 860px; }
.about-section { margin-bottom: 32px; }
.about-section h2 { font-size: 18px; font-weight: 700; margin-bottom: 12px; color: var(--text); }
.about-section p { color: var(--text-muted); line-height: 1.7; margin-bottom: 10px; }
.perf-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 16px; }
.perf-card {
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px;
}
.perf-card h3 { font-size: 16px; font-weight: 700; color: var(--accent); margin-bottom: 12px; }
.perf-stat { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
.perf-stat:last-child { border-bottom: none; }
.perf-stat-label { color: var(--text-muted); }
.perf-stat-val { font-weight: 700; color: var(--text); }
.feat-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.feat-chip {
  font-size: 11px; padding: 3px 10px; border-radius: 999px;
  background: var(--surface3); color: var(--text-muted);
  border: 1px solid var(--border);
}
.disclaimer {
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px; color: var(--text-muted);
  font-size: 12px; line-height: 1.6;
}

/* ── Analysis tab ── */
.analysis-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.analysis-grid.triple { grid-template-columns: 1fr 1fr 1fr; }
.chart-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px;
}
.chart-card-title {
  font-size: 13px; font-weight: 700; color: var(--text-muted);
  margin-bottom: 12px; letter-spacing: 0.3px;
}
.chart-card-canvas { height: 320px; position: relative; }
.analysis-no-data { color: var(--text-dim); text-align: center; padding: 40px; font-size: 13px; }

/* ── Scatter axis selects ── */
.axis-select-wrap { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--text-muted); }
.axis-select {
  background: var(--surface2); border: 1px solid var(--border);
  color: var(--text); font-size: 12px; font-family: inherit;
  padding: 5px 10px; border-radius: 6px; cursor: pointer; outline: none;
}
.axis-select:focus { border-color: var(--accent); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surface3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border); }

/* ── Responsive ── */
@media (max-width: 900px) {
  :root { --panel-width: 100vw; }
  .score-cards { grid-template-columns: repeat(2, 1fr); }
}

.no-data { color: var(--text-dim); text-align: center; padding: 40px; font-size: 14px; }
.empty-dash { color: var(--text-dim); }

/* ── College career season table ── */
.season-table-wrap { overflow-x: auto; margin-top: 6px; }
.season-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.season-table thead th {
  background: var(--surface3); color: var(--text-dim); font-weight: 600;
  font-size: 10px; letter-spacing: 0.4px; text-transform: uppercase;
  padding: 6px 8px; text-align: right; border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
.season-table thead th:first-child,
.season-table thead th:nth-child(2) { text-align: left; }
.season-table tbody tr:nth-child(even) { background: rgba(255,255,255,0.02); }
.season-table tbody td {
  padding: 5px 8px; color: var(--text); text-align: right;
  border-bottom: 1px solid rgba(45,49,72,0.4);
}
.season-table tbody td:first-child { text-align: left; font-weight: 600; }
.season-table tbody td:nth-child(2) { text-align: left; color: var(--text-muted); }
</style>
</head>
<body>
<div id="app">
  <!-- Header -->
  <header>
    <div class="header-logo">Dynasty<span>Prospect</span></div>
    <div class="header-badge">MODEL v2</div>
    <div class="header-spacer"></div>
    <div class="header-meta" id="header-meta"></div>
  </header>

  <!-- Tabs -->
  <div class="tab-bar">
    <button class="tab-btn active" data-tab="rankings">Rankings</button>
    <button class="tab-btn" data-tab="scatter">Scatter Plot</button>
    <button class="tab-btn" data-tab="historical">Historical</button>
    <button class="tab-btn" data-tab="analysis">Analysis</button>
    <button class="tab-btn" data-tab="about">About</button>
  </div>

  <div class="main-area">
    <!-- ── Rankings Tab ── -->
    <div id="tab-rankings" class="tab-content active">
      <div class="controls">
        <div class="pos-btn-group" id="pos-btns">
          <button class="seg-btn active" data-pos="WR">WR</button>
          <button class="seg-btn" data-pos="RB">RB</button>
          <button class="seg-btn" data-pos="TE">TE</button>
        </div>
        <div class="year-btn-group" id="year-btns"></div>
        <div class="controls-spacer"></div>
        <input class="search-box" id="search-box" type="text" placeholder="Search players…">
      </div>
      <div class="table-wrap">
        <table id="rankings-table">
          <thead id="rankings-thead"></thead>
          <tbody id="rankings-tbody"></tbody>
        </table>
      </div>
    </div>

    <!-- ── Scatter Tab ── -->
    <div id="tab-scatter" class="tab-content">
      <div class="controls">
        <div class="pos-btn-group" id="scatter-pos-btns">
          <button class="seg-btn active" data-pos="WR">WR</button>
          <button class="seg-btn" data-pos="RB">RB</button>
          <button class="seg-btn" data-pos="TE">TE</button>
        </div>
        <div class="year-btn-group" id="scatter-year-btns"></div>
        <div class="controls-spacer"></div>
        <div class="axis-select-wrap">
          <span>X:</span>
          <select class="axis-select" id="scatter-x-select">
            <option value="phase1_orbit">Phase I ORBIT</option>
            <option value="orbit_score">ORBIT Score</option>
            <option value="capital_delta">Capital Δ</option>
            <option value="projected_b2s">Proj B2S</option>
            <option value="draft_capital_score">Draft Capital</option>
            <option value="position_rank">Pos Rank</option>
            <option value="best_age">Draft Age</option>
            <option value="best_breakout_score">Breakout Score</option>
            <option value="best_rec_rate">Reception Rate</option>
            <option value="best_yprr">YPRR</option>
          </select>
          <span>Y:</span>
          <select class="axis-select" id="scatter-y-select">
            <option value="orbit_score">ORBIT Score</option>
            <option value="phase1_orbit">Phase I ORBIT</option>
            <option value="capital_delta">Capital Δ</option>
            <option value="projected_b2s">Proj B2S</option>
            <option value="draft_capital_score">Draft Capital</option>
            <option value="position_rank">Pos Rank</option>
            <option value="best_age">Draft Age</option>
            <option value="best_breakout_score">Breakout Score</option>
            <option value="best_rec_rate">Reception Rate</option>
            <option value="best_yprr">YPRR</option>
          </select>
        </div>
      </div>
      <div id="scatter-canvas-wrap">
        <canvas id="scatter-canvas"></canvas>
      </div>
    </div>

    <!-- ── Historical Tab ── -->
    <div id="tab-historical" class="tab-content">
      <div class="controls">
        <div class="pos-btn-group" id="hist-pos-btns">
          <button class="seg-btn active" data-pos="WR">WR</button>
          <button class="seg-btn" data-pos="RB">RB</button>
          <button class="seg-btn" data-pos="TE">TE</button>
        </div>
      </div>
      <div class="hist-meta" id="hist-meta"></div>
      <div class="table-wrap">
        <table id="hist-table">
          <thead id="hist-thead"></thead>
          <tbody id="hist-tbody"></tbody>
        </table>
      </div>
    </div>

    <!-- ── Analysis Tab ── -->
    <div id="tab-analysis" class="tab-content">
      <div class="controls" style="margin-bottom:16px">
        <div class="pos-btn-group" id="analysis-pos-btns">
          <button class="seg-btn active" data-pos="WR">WR</button>
          <button class="seg-btn" data-pos="RB">RB</button>
          <button class="seg-btn" data-pos="TE">TE</button>
        </div>
        <div class="year-btn-group" id="analysis-year-btns"></div>
      </div>
      <!-- Row 1: Breakout Age vs B2S | ORBIT vs Pos Rank -->
      <div class="analysis-grid" style="margin-bottom:20px">
        <div class="chart-card">
          <div class="chart-card-title">Breakout Age vs B2S — Historical</div>
          <div class="chart-card-canvas"><canvas id="analysis-age-b2s"></canvas></div>
        </div>
        <div class="chart-card">
          <div class="chart-card-title">ORBIT Score vs Position Rank — Current Class</div>
          <div class="chart-card-canvas"><canvas id="analysis-orbit-rank"></canvas></div>
        </div>
      </div>
      <!-- Row 2: Capital Calibration -->
      <div class="chart-card">
        <div class="chart-card-title" id="chart3-title">Draft Capital vs Actual B2S — Historical</div>
        <div class="chart-card-canvas" style="height:280px"><canvas id="analysis-cap-b2s"></canvas></div>
      </div>
    </div>

    <!-- ── About Tab ── -->
    <div id="tab-about" class="tab-content">
      <div class="about-wrap">
        <div class="about-section">
          <h2>Dynasty Prospect Model</h2>
          <p>A Ridge regression model trained on 2011–2022 NFL draft classes to predict <strong>Best 2 of 3 Seasons (B2S)</strong> PPR fantasy points per game for WR, RB, and TE prospects. The model uses pre-draft college performance, athleticism, and draft capital signals.</p>
          <p><strong>ORBIT Score</strong> (0–100) is the full model composite — college production, athleticism, and draft capital. <strong>Talent ORBIT (Phase I)</strong> removes draft capital entirely, inspired by JJ Zachariason's ZAP framework. <strong>Capital Δ</strong> = Full ORBIT minus Talent ORBIT: positive means the draft slot is elevating the score (<em>Capital Bet</em>); negative means talent exceeds the draft slot (<em>ZAP Signal</em> — the NFL may be sleeping on this player). For RBs, top-half R1 capital is very reliable (R²=0.53) so Capital Bet is expected; for WRs, R1 capital is weak (R²=0.08) making Capital Bet a genuine flag.</p>
          <p><strong>B2S</strong>: Best 2 of first 3 NFL seasons PPR PPG (minimum 8 games each). Higher is better; NFL starters typically score 10–15 B2S PPG.</p>
        </div>
        <div class="about-section">
          <h2>Model Performance (LOYO Cross-Validation)</h2>
          <p>Leave-One-Year-Out (LOYO) cross-validation: train on all years except one, test on the held-out year. Repeated for each draft year 2014–2022.</p>
          <div class="perf-grid" id="perf-grid"></div>
        </div>
        <div class="about-section">
          <h2>Selected Features</h2>
          <div id="feat-section"></div>
        </div>
        <div class="about-section">
          <h2>Disclaimer</h2>
          <div class="disclaimer">
            This model is a research tool for dynasty fantasy football analysis. Predictions are probabilistic — even an ORBIT 95 prospect can bust. Draft capital projections (pre-draft) are estimates based on consensus big board rankings and carry additional uncertainty. Model was trained on 2011–2022 draft classes only; out-of-sample performance may differ. Not financial advice.
          </div>
        </div>
      </div>
    </div>

    <!-- ── Player Detail Panel ── -->
    <div id="player-panel">
      <button class="panel-close" id="panel-close">✕</button>
      <div class="panel-scroll" id="panel-scroll"></div>
    </div>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════════════════
// EMBEDDED DATA
// ═══════════════════════════════════════════════════════════
const DASHBOARD_DATA = {DASHBOARD_DATA_JSON};

// ═══════════════════════════════════════════════════════════
// APP STATE
// ═══════════════════════════════════════════════════════════
const state = {
  activeTab: 'rankings',
  pos: 'WR',
  year: DASHBOARD_DATA.meta.primary_year,
  activeYears: new Set([DASHBOARD_DATA.meta.primary_year]),
  scatterPos: 'WR',
  scatterYear: DASHBOARD_DATA.meta.primary_year,
  scatterX: 'phase1_orbit',
  scatterY: 'orbit_score',
  histPos: 'WR',
  histSort: { key: 'b2s_score', dir: -1 },
  analysisPos: 'WR',
  analysisYear: DASHBOARD_DATA.meta.primary_year,
  analysisYears: new Set([DASHBOARD_DATA.meta.primary_year]),
  sortCol: 'pos_rank',
  sortDir: 1,
  search: '',
  selectedPlayer: null,
  scatterChart: null,
  analysisCharts: {},
};

// ═══════════════════════════════════════════════════════════
// UTILS
// ═══════════════════════════════════════════════════════════
const FL = DASHBOARD_DATA.feature_labels || {};
const INVERSE = new Set(DASHBOARD_DATA.inverse_features || []);
const BINARY = new Set(DASHBOARD_DATA.binary_features || []);

// Register datalabels plugin (only used in scatter/analysis charts — opt-in per chart)
if (typeof ChartDataLabels !== 'undefined') {
  Chart.register(ChartDataLabels);
}

// Axis accessor: resolves top-level player fields OR nested features[key].value
function axisValue(player, key) {
  if (player[key] !== null && player[key] !== undefined) return player[key];
  const f = (player.features || {})[key];
  if (f && typeof f === 'object' && f.value !== null && f.value !== undefined) return f.value;
  return null;
}

const AXIS_LABELS = {
  phase1_orbit: 'Phase I ORBIT (Talent Only)',
  orbit_score: 'ORBIT Score (Full Model)',
  capital_delta: 'Capital Δ (ORBIT − Phase I)',
  projected_b2s: 'Projected B2S PPG',
  draft_capital_score: 'Draft Capital Score',
  position_rank: 'Position Rank (Big Board)',
  best_age: 'Draft Age',
  best_breakout_score: 'Breakout Score',
  best_rec_rate: 'Reception Rate',
  best_yprr: 'YPRR',
};

function fmt(v, decimals=1, suffix='') {
  if (v === null || v === undefined || (typeof v === 'number' && isNaN(v))) return '—';
  return Number(v).toFixed(decimals) + suffix;
}

function fmtHeight(inches) {
  if (inches === null || inches === undefined) return '—';
  const ft = Math.floor(inches / 12);
  const inn = Math.round(inches % 12);
  return `${ft}'${inn}"`;
}

function orbitColor(z) {
  if (z === null || z === undefined) return 'var(--text-muted)';
  if (z >= 80) return 'var(--green)';
  if (z >= 50) return 'var(--yellow)';
  return 'var(--red)';
}

function orbitClass(z) {
  if (z === null || z === undefined) return '';
  if (z >= 80) return 'orbit-green';
  if (z >= 50) return 'orbit-yellow';
  return 'orbit-red';
}

function orbitBarColor(z) {
  if (z >= 80) return 'var(--green)';
  if (z >= 50) return 'var(--yellow)';
  return 'var(--red)';
}

function riskBadge(risk) {
  if (!risk) return '<span class="risk-badge risk-neutral">—</span>';
  const lc = risk.toLowerCase();
  if (lc.includes('zap'))     return `<span class="risk-badge risk-low" title="Talent ORBIT well above draft capital — production not yet priced in. ZAP-type buy signal (per JJ Zachariason's framework).">ZAP Signal</span>`;
  if (lc.includes('capital')) return `<span class="risk-badge risk-high" title="Score driven by draft capital more than college production. For WRs this is a real flag (R²=0.08 in R1); for top-half R1 RBs capital is very reliable (R²=0.53).">Capital Bet</span>`;
  return `<span class="risk-badge risk-neutral" title="Capital and production signals aligned.">Balanced</span>`;
}

function deltaHtml(d) {
  if (d === null || d === undefined) return '<span class="delta-neutral">—</span>';
  const sign = d < -1 ? 'delta-neg' : d > 1 ? 'delta-pos' : 'delta-neutral';
  const prefix = d > 0 ? '+' : '';
  const tip = d > 1 ? 'Capital elevating score above talent-only' : d < -1 ? 'Talent score exceeds draft slot — ZAP signal' : 'Aligned';
  return `<span class="${sign}" title="${tip}">${prefix}${Number(d).toFixed(1)}</span>`;
}

function pctColor(pct) {
  if (pct === null || pct === undefined) return 'pct-gray';
  if (pct >= 80) return 'pct-green';
  if (pct >= 50) return 'pct-blue';
  if (pct >= 20) return 'pct-amber';
  return 'pct-red';
}

function tierLabel(pct) {
  if (pct === null || pct === undefined) return { cls: 'tier-na', text: 'N/A' };
  if (pct >= 85) return { cls: 'tier-elite', text: 'Elite' };
  if (pct >= 65) return { cls: 'tier-good', text: 'Good' };
  if (pct >= 35) return { cls: 'tier-avg', text: 'Average' };
  if (pct >= 15) return { cls: 'tier-below', text: 'Below Avg' };
  return { cls: 'tier-weak', text: 'Weak' };
}

function getPlayers(year, pos) {
  const yr = DASHBOARD_DATA.players[String(year)];
  if (!yr) return [];
  return (yr[pos] || []);
}

function getHistorical(pos) {
  return (DASHBOARD_DATA.historical[pos] || []);
}

// Percentile of score using embedded distributions
function livePercentile(pos, feature, value) {
  if (value === null || value === undefined) return null;
  const dist = (DASHBOARD_DATA.feature_distributions[pos] || {})[feature] || [];
  if (dist.length === 0) return null;
  const sorted = dist.slice().sort((a, b) => a - b);
  let count = 0;
  for (const v of sorted) { if (v <= value) count++; }
  let pct = (count / sorted.length) * 100;
  if (INVERSE.has(feature)) pct = 100 - pct;
  return Math.min(100, Math.max(0, pct));
}

// ═══════════════════════════════════════════════════════════
// TABS
// ═══════════════════════════════════════════════════════════
function switchTab(tabId) {
  state.activeTab = tabId;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
  document.querySelectorAll('.tab-content').forEach(el => el.classList.toggle('active', el.id === `tab-${tabId}`));
  if (tabId === 'scatter') renderScatter();
  if (tabId === 'historical') renderHistorical();
  if (tabId === 'analysis') renderAnalysis();
  if (tabId === 'about') renderAbout();
}

document.querySelectorAll('.tab-btn').forEach(b => {
  b.addEventListener('click', () => switchTab(b.dataset.tab));
});

// ═══════════════════════════════════════════════════════════
// YEAR BUTTONS
// ═══════════════════════════════════════════════════════════
function makeYearBtns(container, stateKey, onchange) {
  const years = DASHBOARD_DATA.meta.years || [];
  container.innerHTML = years.map(y =>
    `<button class="seg-btn ${y === state[stateKey] ? 'active' : ''}" data-year="${y}">${y}</button>`
  ).join('');
  container.querySelectorAll('.seg-btn').forEach(b => {
    b.addEventListener('click', () => {
      state[stateKey] = b.dataset.year;
      container.querySelectorAll('.seg-btn').forEach(x => x.classList.toggle('active', x.dataset.year === b.dataset.year));
      onchange();
    });
  });
}

// Multi-select year buttons for Rankings tab
function makeRankingsYearBtns(container) {
  const years = DASHBOARD_DATA.meta.years || [];
  container.innerHTML = years.map(y =>
    `<button class="seg-btn ${state.activeYears.has(y) ? 'active' : ''}" data-year="${y}">${y}</button>`
  ).join('');
  container.querySelectorAll('.seg-btn').forEach(b => {
    b.addEventListener('click', () => {
      const yr = b.dataset.year;
      if (state.activeYears.has(yr)) {
        // Don't allow deselecting the last active year
        if (state.activeYears.size > 1) state.activeYears.delete(yr);
      } else {
        state.activeYears.add(yr);
      }
      container.querySelectorAll('.seg-btn').forEach(x => x.classList.toggle('active', state.activeYears.has(x.dataset.year)));
      renderRankingsBody();
    });
  });
}

// Multi-select year buttons for Analysis tab
function makeAnalysisYearBtns(container) {
  const years = DASHBOARD_DATA.meta.years || [];
  container.innerHTML = years.map(y =>
    `<button class="seg-btn ${state.analysisYears.has(y) ? 'active' : ''}" data-year="${y}">${y}</button>`
  ).join('');
  container.querySelectorAll('.seg-btn').forEach(b => {
    b.addEventListener('click', () => {
      const yr = b.dataset.year;
      if (state.analysisYears.has(yr)) {
        if (state.analysisYears.size > 1) state.analysisYears.delete(yr);
      } else {
        state.analysisYears.add(yr);
        state.analysisYear = yr; // keep single-year for chart2/chart3 (last clicked)
      }
      container.querySelectorAll('.seg-btn').forEach(x =>
        x.classList.toggle('active', state.analysisYears.has(x.dataset.year))
      );
      if (state.activeTab === 'analysis') renderAnalysis();
    });
  });
}

// ═══════════════════════════════════════════════════════════
// RANKINGS TAB
// ═══════════════════════════════════════════════════════════
const RANK_COLS = [
  { key: 'pos_rank', label: 'Rank', fmt: v => v !== null ? `#${Math.round(v)}` : '—' },
  { key: 'name', label: 'Player', fmt: v => `<span style="font-weight:600">${v}</span>` },
  { key: 'college', label: 'College', fmt: v => v || '—' },
  { key: 'draft_year', label: 'Class', fmt: v => v ? `<span style="color:var(--text-muted)">'${String(v).slice(-2)}</span>` : '—', multiYearOnly: true },
  { key: 'best_age', label: 'Age', fmt: v => v !== null && v !== undefined ? `<span style="color:var(--text-muted)">${Number(v).toFixed(1)}</span>` : '<span class="empty-dash">—</span>' },
  { key: 'best_breakout_score', label: 'BrkOut', fmt: v => v !== null && v !== undefined ? `<span style="color:var(--text-muted)">${Number(v).toFixed(1)}</span>` : '<span class="empty-dash">—</span>' },
  { key: '_htw', label: 'Ht / Wt', fmt: (_, r) => {
      const h = r.height_inches !== null ? fmtHeight(r.height_inches) : '—';
      const w = r.weight_lbs !== null ? `${Math.round(r.weight_lbs)} lbs` : '—';
      return `<span style="color:var(--text-muted)">${h} / ${w}</span>`;
    }, noSort: true },
  { key: 'orbit_score', label: 'ORBIT', fmt: v => {
      if (v === null || v === undefined) return '—';
      const z = Math.round(v);
      const pct = Math.min(100, Math.max(0, v));
      return `<div class="orbit-cell">
        <span class="orbit-num ${orbitClass(v)}">${z}</span>
        <div class="orbit-bar-track">
          <div class="orbit-bar-fill" style="width:${pct}%;background:${orbitBarColor(v)}"></div>
        </div>
      </div>`;
    }},
  { key: 'phase1_orbit', label: 'Talent ORBIT', title: 'Talent-only score (no draft capital) — Phase I model', fmt: v => v !== null && v !== undefined ? `<span style="color:var(--text-muted)">${Math.round(v)}</span>` : '<span class="empty-dash">—</span>' },
  { key: 'capital_delta', label: 'Cap Δ', title: 'Full ORBIT − Talent ORBIT. Positive = capital inflating score (Capital Bet). Negative = talent exceeds draft slot (ZAP Signal).', fmt: v => deltaHtml(v) },
  { key: 'risk', label: 'Signal', fmt: v => riskBadge(v), noSort: true },
  { key: '_ci', label: '80% CI', fmt: (_, r) => {
      if (r.b2s_lo80 === null || r.b2s_hi80 === null) return '—';
      return `<span style="color:var(--text-muted);font-size:12px">${fmt(r.b2s_lo80)}–${fmt(r.b2s_hi80)}</span>`;
    }, noSort: true },
  { key: 'overall_pick', label: 'Pick', fmt: v => {
      if (v === null || v === undefined) return '<span style="color:var(--text-dim)">TBD</span>';
      return `<span style="color:var(--text-muted)">#${Math.round(v)}</span>`;
    }},
];

function buildRankingsHeader() {
  const multiYear = state.activeYears.size > 1;
  const visibleCols = RANK_COLS.filter(c => !c.multiYearOnly || multiYear);
  document.getElementById('rankings-thead').innerHTML =
    `<tr>${visibleCols.map(c => `
      <th data-key="${c.key}" ${c.noSort ? '' : 'class="sortable"'} ${c.title ? `title="${c.title}"` : ''}>
        ${c.label}${c.noSort ? '' : '<span class="sort-icon">↕</span>'}
      </th>`).join('')}</tr>`;

  document.querySelectorAll('#rankings-thead th.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.key;
      if (state.sortCol === key) state.sortDir *= -1;
      else { state.sortCol = key; state.sortDir = 1; }
      renderRankingsBody();
    });
  });
}

function renderRankingsBody() {
  // Gather players from all active years (multi-year support)
  const multiYear = state.activeYears.size > 1;
  let players = [];
  for (const yr of state.activeYears) {
    players = players.concat(getPlayers(yr, state.pos));
  }
  // Rebuild header to show/hide Class column
  buildRankingsHeader();

  // Search filter
  if (state.search) {
    const q = state.search.toLowerCase();
    players = players.filter(p => p.name.toLowerCase().includes(q) || (p.college || '').toLowerCase().includes(q));
  }

  // Sort
  players = players.slice().sort((a, b) => {
    let va = a[state.sortCol];
    let vb = b[state.sortCol];
    if (va === null || va === undefined) va = state.sortDir > 0 ? Infinity : -Infinity;
    if (vb === null || vb === undefined) vb = state.sortDir > 0 ? Infinity : -Infinity;
    if (va < vb) return -state.sortDir;
    if (va > vb) return state.sortDir;
    return 0;
  });

  // Update sort icons
  document.querySelectorAll('#rankings-thead th').forEach(th => {
    const icon = th.querySelector('.sort-icon');
    if (!icon) return;
    const active = th.dataset.key === state.sortCol;
    th.classList.toggle('sorted', active);
    icon.textContent = active ? (state.sortDir > 0 ? '↑' : '↓') : '↕';
  });

  const tbody = document.getElementById('rankings-tbody');
  if (!players.length) {
    tbody.innerHTML = `<tr><td colspan="${RANK_COLS.length}" class="no-data">No players found.</td></tr>`;
    return;
  }

  const visibleCols = RANK_COLS.filter(c => !c.multiYearOnly || multiYear);
  tbody.innerHTML = players.map(p => {
    const isSelected = state.selectedPlayer && state.selectedPlayer.name === p.name && state.selectedPlayer.draft_year === p.draft_year;
    return `<tr class="${isSelected ? 'selected' : ''}" data-name="${p.name}" data-year="${p.draft_year}">
      ${visibleCols.map(c => {
        let html;
        if (c.fmt.length >= 2) html = c.fmt(p[c.key], p);
        else html = c.fmt(p[c.key]);
        return `<td>${html}</td>`;
      }).join('')}
    </tr>`;
  }).join('');

  tbody.querySelectorAll('tr').forEach(tr => {
    tr.addEventListener('click', () => {
      const name = tr.dataset.name;
      const year = tr.dataset.year;
      const player = getPlayers(year, state.pos).find(p => p.name === name && String(p.draft_year) === String(year));
      if (player) openPlayerPanel(player);
    });
  });
}

function initRankings() {
  buildRankingsHeader();

  // Position buttons
  document.querySelectorAll('#pos-btns .seg-btn').forEach(b => {
    b.addEventListener('click', () => {
      state.pos = b.dataset.pos;
      document.querySelectorAll('#pos-btns .seg-btn').forEach(x => x.classList.toggle('active', x.dataset.pos === b.dataset.pos));
      renderRankingsBody();
    });
  });

  // Year buttons (multi-select for rankings)
  makeRankingsYearBtns(document.getElementById('year-btns'));

  // Search
  document.getElementById('search-box').addEventListener('input', e => {
    state.search = e.target.value;
    renderRankingsBody();
  });

  renderRankingsBody();
}

// ═══════════════════════════════════════════════════════════
// PLAYER PANEL
// ═══════════════════════════════════════════════════════════
function openPlayerPanel(player) {
  state.selectedPlayer = player;
  renderPlayerPanel(player);
  document.getElementById('player-panel').classList.add('open');
  renderRankingsBody(); // refresh selection highlight
}

function closePlayerPanel() {
  document.getElementById('player-panel').classList.remove('open');
  state.selectedPlayer = null;
  renderRankingsBody();
}

document.getElementById('panel-close').addEventListener('click', closePlayerPanel);

function renderPlayerPanel(p) {
  const pos = p.position;

  // Determine which features to show
  const modelPerf = (DASHBOARD_DATA.model_performance || {})[pos] || {};
  const capFeatures = modelPerf.selected_features || [];
  const nocapFeatures = modelPerf.nocap_features || [];
  const capCoefs = modelPerf.ridge_coefs || {};
  const nocapCoefs = modelPerf.nocap_ridge_coefs || {};

  // Build HTML
  let html = '';

  // Header
  html += `<div class="panel-header">
    <div class="panel-name">${p.name}</div>
    <div class="panel-sub">${p.college || '—'} · ${pos} · ${p.draft_year}${p.capital_is_projected ? ' <span style="color:var(--text-dim);font-size:11px">(pre-draft)</span>' : ''}</div>
  </div>`;

  // Score cards
  const orbitColor2 = orbitColor(p.orbit_score);
  html += `<div class="score-cards">
    <div class="score-card">
      <div class="score-card-label">ORBIT</div>
      <div class="score-card-val" style="color:${orbitColor2}">${p.orbit_score !== null ? Math.round(p.orbit_score) : '—'}</div>
      <div class="score-card-sub">${p.projected_b2s !== null ? `Proj B2S: ${fmt(p.projected_b2s)}` : ''}</div>
    </div>
    <div class="score-card">
      <div class="score-card-label">Talent ORBIT</div>
      <div class="score-card-val" style="color:var(--text-muted)">${p.phase1_orbit !== null && p.phase1_orbit !== undefined ? Math.round(p.phase1_orbit) : '—'}</div>
      <div class="score-card-sub">No draft capital</div>
    </div>
    <div class="score-card">
      <div class="score-card-label">Capital Δ</div>
      <div class="score-card-val" style="font-size:18px">${deltaHtml(p.capital_delta)}</div>
      <div class="score-card-sub">
        ${p.capital_delta !== null && p.capital_delta > 1 ? `+${Math.round(p.capital_delta)} pts from draft slot` : p.capital_delta !== null && p.capital_delta < -1 ? `${Math.round(p.capital_delta)} pts vs draft slot` : ''}
        ${riskBadge(p.risk)}
      </div>
    </div>
  </div>`;

  // Data quality warnings
  const warnings = [];
  if (p.age_imputed) {
    if (p.age_suspect && p.best_age === 18.5)
      warnings.push('⚠️ Draft age = 18.5 (no DOB/recruit year — likely JUCO or FCS transfer). Phase I score may be inflated.');
    else if (!p.best_age)
      warnings.push('⚠️ Draft age missing — Phase I score uses median imputation.');
  }
  if (p.age_suspect && p.best_age > 26)
    warnings.push(`⚠️ Draft age = ${p.best_age?.toFixed(1)} appears incorrect (DB data issue). Phase I score penalized.`);
  if (p.breakout_imputed)
    warnings.push('⚠️ Breakout score missing (no PFF/CFBD data for best season). Phase I uses median imputation.');
  if (warnings.length) {
    html += `<div style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);border-radius:6px;padding:10px 12px;margin-bottom:12px;font-size:11px;color:#fbbf24;line-height:1.5">
      ${warnings.join('<br>')}
    </div>`;
  }

  // Confidence interval
  if (p.b2s_lo80 !== null && p.b2s_hi80 !== null) {
    const lo80 = p.b2s_lo80, hi80 = p.b2s_hi80;
    const lo90 = p.b2s_lo90, hi90 = p.b2s_hi90;
    const proj = p.projected_b2s || 0;
    const maxVal = Math.max(hi90 || hi80, 25);
    const pctPos = v => Math.min(100, Math.max(0, (v / maxVal) * 100));
    html += `<div class="ci-bar-wrap">
      <div class="ci-title">Predicted B2S Range</div>
      <div class="ci-row">
        <span class="ci-label">80%</span>
        <div class="ci-track">
          <div class="ci-range" style="left:${pctPos(lo80)}%;width:${pctPos(hi80)-pctPos(lo80)}%"></div>
          <div class="ci-point" style="left:${pctPos(proj)}%"></div>
        </div>
        <span class="ci-vals">${fmt(lo80)}–${fmt(hi80)} PPG</span>
      </div>
      ${lo90 !== null ? `<div class="ci-row">
        <span class="ci-label">90%</span>
        <div class="ci-track">
          <div class="ci-range" style="left:${pctPos(lo90)}%;width:${pctPos(hi90)-pctPos(lo90)}%;opacity:0.35"></div>
          <div class="ci-point" style="left:${pctPos(proj)}%"></div>
        </div>
        <span class="ci-vals">${fmt(lo90)}–${fmt(hi90)} PPG</span>
      </div>` : ''}
    </div>`;
  }

  // Physical profile
  html += `<div class="section-title">Physical Profile</div>
  <div class="phys-grid">
    ${physItem('Height', fmtHeight(p.height_inches))}
    ${physItem('Weight', p.weight_lbs !== null ? `${Math.round(p.weight_lbs)} lbs` : '—')}
    ${physItem('40-Yard Dash', p.forty_time !== null ? `${fmt(p.forty_time, 2)}s` : '—')}
    ${physItem('Speed Score', p.speed_score !== null ? fmt(p.speed_score, 1) : '—')}
    ${physItem('Broad Jump', p.broad_jump !== null ? `${Math.round(p.broad_jump)}"` : '—')}
    ${physItem('Vertical', p.vertical_jump !== null ? `${fmt(p.vertical_jump, 1)}"` : '—')}
    ${physItem('3-Cone', p.three_cone !== null ? `${fmt(p.three_cone, 2)}s` : '—')}
    ${physItem('Shuttle', p.shuttle !== null ? `${fmt(p.shuttle, 2)}s` : '—')}
  </div>`;

  // College career season table
  if (p.seasons && p.seasons.length > 0) {
    const isRB = pos === 'RB';
    const showPFF = !isRB;
    html += `<div class="section-title">College Career</div>
    <div class="season-table-wrap">
      <table class="season-table">
        <thead><tr>
          <th>Year</th><th>Team</th><th>G</th>
          <th>Rec</th><th>Tgt</th><th>RecYd</th><th>RecTD</th>
          ${isRB ? '<th>Att</th><th>RuYd</th><th>RuTD</th>' : ''}
          ${showPFF ? '<th>YPRR</th><th>Gr</th>' : ''}
        </tr></thead>
        <tbody>
          ${p.seasons.map(s => `<tr>
            <td>${s.year}</td>
            <td>${s.team || '—'}</td>
            <td>${s.games != null ? s.games : '—'}</td>
            <td>${s.receptions != null ? s.receptions : '—'}</td>
            <td>${s.targets != null ? s.targets : '—'}</td>
            <td>${s.rec_yards != null ? s.rec_yards : '—'}</td>
            <td>${s.rec_tds != null ? s.rec_tds : '—'}</td>
            ${isRB ? `<td>${s.rush_attempts != null ? s.rush_attempts : '—'}</td><td>${s.rush_yards != null ? s.rush_yards : '—'}</td><td>${s.rush_tds != null ? s.rush_tds : '—'}</td>` : ''}
            ${showPFF ? `<td>${s.yprr != null ? Number(s.yprr).toFixed(2) : '—'}</td><td>${s.receiving_grade != null ? Math.round(s.receiving_grade) : '—'}</td>` : ''}
          </tr>`).join('')}
        </tbody>
      </table>
    </div>`;
  }

  // Capital model features
  if (capFeatures.length > 0) {
    html += `<div class="section-title">Capital Model Features <span style="font-weight:400;font-size:11px;color:var(--text-dim)">— signal arrows show coefficient direction</span></div>
    <div class="feat-list">
      ${capFeatures.map(f => featRow(f, p, pos, capCoefs)).join('')}
    </div>`;
  }

  // Phase I features
  if (nocapFeatures.length > 0) {
    // WR-specific multicollinearity note for PFF splits
    const multicollNote = (pos === 'WR') ? `<div style="font-size:10px;color:var(--text-dim);margin-top:6px;margin-bottom:10px;line-height:1.5;padding:8px 10px;background:rgba(148,163,184,0.06);border-radius:6px">
      ℹ️ WR PFF splits (Man YPRR, Zone YPRR, Man–Zone Delta) are correlated (r≈0.8). Ridge regression creates suppressor effects: a negative coefficient doesn't mean the stat is bad — it reflects shared variance. Trust the percentile bar, not the arrow direction, for individual PFF splits.
    </div>` : '';
    html += `<div class="section-title">Phase I Features (talent-only) <span style="font-weight:400;font-size:11px;color:var(--text-dim)">— signal arrows show coefficient direction</span></div>
    ${multicollNote}
    <div class="feat-list">
      ${nocapFeatures.map(f => featRow(f, p, pos, nocapCoefs)).join('')}
    </div>`;
  }

  html += '<div style="height:24px"></div>';
  document.getElementById('panel-scroll').innerHTML = html;
  document.getElementById('panel-scroll').scrollTop = 0;
}

function physItem(label, val) {
  return `<div class="phys-item">
    <div class="phys-label">${label}</div>
    <div class="phys-val">${val}</div>
  </div>`;
}

function featRow(feature, player, pos, coefs) {
  const label = FL[feature] || feature;
  const featData = (player.features || {})[feature];
  const coef = coefs ? (coefs[feature] ?? null) : null;

  // Build signal badge from coefficient
  function signalBadge(c) {
    if (c === null) return '';
    const abs = Math.abs(c);
    // strength: high ≥1.0, medium ≥0.4, low <0.4
    const stars = abs >= 1.0 ? '●●●' : abs >= 0.5 ? '●●○' : '●○○';
    const arrow = c > 0 ? '↑' : '↓';
    const tooltip = `Coefficient: ${c > 0 ? '+' : ''}${c.toFixed(3)} (${c > 0 ? 'higher = better' : 'lower = better'})`;
    const arrowColor = c > 0 ? '#22c55e' : '#f87171';
    return `<span title="${tooltip}" style="font-size:10px;color:${arrowColor};letter-spacing:0">${arrow}</span><span title="${tooltip}" style="font-size:9px;color:var(--text-dim);letter-spacing:-1px">${stars}</span>`;
  }

  // Binary feature
  if (BINARY.has(feature)) {
    const val = typeof featData === 'object' ? featData : featData;
    const display = val === 1 || val === true ? 'Yes' : val === 0 || val === false ? 'No' : '—';
    return `<div class="feat-row">
      <div class="feat-row-top">
        <span class="feat-label">${label} ${signalBadge(coef)}</span>
        <span class="feat-val">${display}</span>
      </div>
    </div>`;
  }

  // Numeric feature — try to get from player.features first, then direct field
  let value = null, pct = null;
  if (featData && typeof featData === 'object') {
    value = featData.value;
    pct = featData.pct;
  } else if (typeof featData === 'number') {
    value = featData;
  }

  // Fallback: compute pct live from distributions
  if (value !== null && pct === null) {
    pct = livePercentile(pos, feature, value);
  }

  // Format value
  let displayVal = '—';
  if (value !== null && value !== undefined) {
    if (feature === 'height_inches') displayVal = fmtHeight(value);
    else if (feature === 'weight_lbs') displayVal = `${Math.round(value)} lbs`;
    else if (feature === 'forty_time') displayVal = `${Number(value).toFixed(2)}s`;
    else if (feature.includes('yprr') || feature.includes('rate') || feature.includes('score') || feature.includes('ppg') || feature.includes('ypc') || feature.includes('usage')) {
      displayVal = Number(value).toFixed(2);
    } else if (feature === 'overall_pick' || feature === 'position_rank' || feature === 'consensus_rank') {
      displayVal = `#${Math.round(value)}`;
    } else if (feature === 'draft_capital_score' || feature === 'capital_x_age' || feature === 'breakout_score_x_capital') {
      displayVal = Number(value).toFixed(1);
    } else {
      displayVal = Number(value).toFixed(2);
    }
  }

  const isImputed = (value === null);
  const tier = tierLabel(pct);
  const pctWidth = pct !== null ? Math.min(100, Math.max(0, pct)) : 0;
  const barColor = pctColor(pct);

  return `<div class="feat-row${isImputed ? ' feat-imputed' : ''}">
    <div class="feat-row-top">
      <span class="feat-label">${label} ${signalBadge(coef)}</span>
      <div style="display:flex;align-items:center;gap:8px">
        ${isImputed ? '<span style="font-size:10px;color:var(--text-dim)" title="Missing — model uses median imputation">imputed</span>' : `<span class="feat-tier ${tier.cls}">${tier.text}</span>`}
        <span class="feat-val">${isImputed ? '—' : displayVal}</span>
      </div>
    </div>
    ${pct !== null && !isImputed ? `<div class="pct-track"><div class="pct-fill ${barColor}" style="width:${pctWidth}%"></div></div>` : ''}
  </div>`;
}

// ═══════════════════════════════════════════════════════════
// SCATTER TAB
// ═══════════════════════════════════════════════════════════
function renderScatter() {
  const players = getPlayers(state.scatterYear, state.scatterPos);
  const canvas = document.getElementById('scatter-canvas');
  const xKey = state.scatterX;
  const yKey = state.scatterY;

  // Filter players with both selected axis values (C1 null safety + C4 configurable)
  const pts = players.filter(p => axisValue(p, xKey) !== null && axisValue(p, yKey) !== null);

  // C1: Guard — need ≥2 points to render
  if (pts.length < 2) {
    if (state.scatterChart) { state.scatterChart.destroy(); state.scatterChart = null; }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = '14px Inter, sans-serif';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.fillText('Not enough data to render scatter (need ≥2 players with both values)', canvas.width / 2, canvas.height / 2);
    return;
  }

  const datasets = [{
    label: 'Prospects',
    data: pts.map(p => ({ x: axisValue(p, xKey), y: axisValue(p, yKey), player: p })),
    backgroundColor: pts.map(p => {
      const r = p.risk || '';
      if (r.includes('Low')) return 'rgba(34,197,94,0.75)';
      if (r.includes('High')) return 'rgba(239,68,68,0.75)';
      return 'rgba(148,163,184,0.65)';
    }),
    pointRadius: 7,
    pointHoverRadius: 10,
    // C3: datalabels — show last name for interesting players only
    datalabels: {
      display: ctx => {
        const p = ctx.dataset.data[ctx.dataIndex].player;
        if (!p) return false;
        const orbit = p.orbit_score;
        const delta = p.capital_delta;
        return (orbit !== null && orbit >= 70) || (delta !== null && Math.abs(delta) >= 25);
      },
      formatter: (val, ctx) => {
        const p = ctx.dataset.data[ctx.dataIndex].player;
        if (!p) return '';
        const parts = p.name.split(' ');
        return parts[parts.length - 1]; // last name
      },
      font: { size: 8, family: 'Inter, sans-serif' },
      color: '#94a3b8',
      anchor: 'end',
      align: 'right',
      offset: 3,
      clamp: true,
    },
  }];

  // Diagonal/reference line (only for same-axis defaults — orbit vs phase1)
  const isDefaultAxes = (xKey === 'phase1_orbit' && yKey === 'orbit_score') ||
                        (xKey === 'orbit_score' && yKey === 'phase1_orbit');
  if (isDefaultAxes) {
    const allVals = pts.flatMap(p => [axisValue(p, xKey), axisValue(p, yKey)]).filter(v => v !== null);
    const minV = Math.min(...allVals, 0);
    const maxV = Math.max(...allVals, 100);
    datasets.push({
      label: 'y=x line',
      data: [{ x: minV, y: minV }, { x: maxV, y: maxV }],
      type: 'line',
      borderColor: 'rgba(255,255,255,0.2)',
      borderDash: [6, 4],
      borderWidth: 1.5,
      pointRadius: 0,
      fill: false,
      datalabels: { display: false },
    });
  }

  if (state.scatterChart) {
    state.scatterChart.destroy();
    state.scatterChart = null;
  }

  state.scatterChart = new Chart(canvas, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const p = ctx.raw.player;
              if (!p) return '';
              const xv = axisValue(p, xKey);
              const yv = axisValue(p, yKey);
              const xLabel = AXIS_LABELS[xKey] || xKey;
              const yLabel = AXIS_LABELS[yKey] || yKey;
              return [
                `${p.name} (${p.college})`,
                `${xLabel}: ${xv !== null ? Number(xv).toFixed(1) : '—'}`,
                `${yLabel}: ${yv !== null ? Number(yv).toFixed(1) : '—'}`,
                `Risk: ${p.risk || '—'}`,
              ];
            },
          },
          backgroundColor: 'rgba(26,29,39,0.97)',
          borderColor: '#2d3148', borderWidth: 1,
          titleColor: '#e2e8f0', bodyColor: '#94a3b8',
          padding: 12,
        },
        datalabels: { display: false }, // default off; per-dataset above overrides
      },
      scales: {
        x: {
          title: { display: true, text: AXIS_LABELS[xKey] || xKey, color: '#94a3b8', font: { size: 12 } },
          grid: { color: 'rgba(45,49,72,0.5)' },
          ticks: { color: '#64748b' },
        },
        y: {
          title: { display: true, text: AXIS_LABELS[yKey] || yKey, color: '#94a3b8', font: { size: 12 } },
          grid: { color: 'rgba(45,49,72,0.5)' },
          ticks: { color: '#64748b' },
        },
      },
      onClick: (evt, items) => {
        if (!items.length) return;
        const item = items[0];
        const ds = state.scatterChart.data.datasets[item.datasetIndex];
        const raw = ds.data[item.index];
        if (raw && raw.player) openPlayerPanel(raw.player);
      },
    },
  });
}

function initScatter() {
  document.querySelectorAll('#scatter-pos-btns .seg-btn').forEach(b => {
    b.addEventListener('click', () => {
      state.scatterPos = b.dataset.pos;
      document.querySelectorAll('#scatter-pos-btns .seg-btn').forEach(x => x.classList.toggle('active', x.dataset.pos === b.dataset.pos));
      renderScatter();
    });
  });
  makeYearBtns(document.getElementById('scatter-year-btns'), 'scatterYear', renderScatter);

  // C4: axis selectors
  const xSel = document.getElementById('scatter-x-select');
  const ySel = document.getElementById('scatter-y-select');
  if (xSel) {
    xSel.value = state.scatterX;
    xSel.addEventListener('change', () => { state.scatterX = xSel.value; renderScatter(); });
  }
  if (ySel) {
    ySel.value = state.scatterY;
    ySel.addEventListener('change', () => { state.scatterY = ySel.value; renderScatter(); });
  }
}

// ═══════════════════════════════════════════════════════════
// HISTORICAL TAB
// ═══════════════════════════════════════════════════════════
function renderHistorical() {
  const pos = state.histPos;
  const players = getHistorical(pos);

  const orbHit = players.filter(p => p.orbit_score !== null && p.orbit_score !== undefined).length;
  const orbNote = orbHit > 0
    ? ` · ORBIT scores shown are in-sample (model trained on these players — slightly optimistic)`
    : '';
  document.getElementById('hist-meta').textContent =
    players.length
      ? `${players.length} ${pos} prospects — 2011–2022 draft classes (known outcomes)${orbNote}`
      : 'No historical data loaded. Re-run export script with --include-historical flag.';

  const thead = document.getElementById('hist-thead');
  const tbody = document.getElementById('hist-tbody');

  const sk = state.histSort.key;
  const sd_arr = state.histSort.dir === 1 ? ' ▲' : ' ▼';
  const shdr = (label, key, title='') =>
    `<th style="cursor:pointer;user-select:none" title="${title}" onclick="sortHistorical('${key}')">${label}${sk===key?sd_arr:''}</th>`;
  thead.innerHTML = `<tr>
    ${shdr('Player','name')}${shdr('College','college')}${shdr('Draft Yr','draft_year')}
    ${shdr('Pick','overall_pick')}${shdr('Capital','draft_capital_score')}
    ${shdr('ORBIT','orbit_score','ORBIT: capital model score (0–100)')}
    ${shdr('Phase I','phase1_orbit','Phase I ORBIT: talent-only score (no draft capital)')}
    ${shdr('Δ Cap','capital_delta','Δ Capital: ORBIT minus Phase I ORBIT')}
    ${shdr('Actual B2S','b2s_score')}
  </tr>`;

  if (!players.length) { tbody.innerHTML = ''; return; }

  const sKey = state.histSort.key;
  const sDir = state.histSort.dir;
  const sorted = players.slice().sort((a, b) => {
    const av = a[sKey], bv = b[sKey];
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    return typeof av === 'string'
      ? sDir * av.localeCompare(bv)
      : sDir * (bv - av);  // dir=-1 = descending (largest first)
  });

  tbody.innerHTML = sorted.map(p => {
    const b2s = p.b2s_score;
    const b2sClass = b2s === null ? '' : b2s >= 12 ? 'outcome-beat' : b2s >= 6 ? 'outcome-near' : 'outcome-miss';
    const orb = p.orbit_score;
    const p1 = p.phase1_orbit;
    const delta = p.capital_delta;
    const orbColor = orb !== null && orb !== undefined ? orbitColor(orb) : 'inherit';
    const p1Color = p1 !== null && p1 !== undefined ? orbitColor(p1) : 'inherit';
    let deltaStr = '—';
    if (delta !== null && delta !== undefined) {
      const sign = delta > 0 ? '+' : '';
      const col = delta >= 15 ? '#ef4444' : delta <= -15 ? '#22c55e' : '#94a3b8';
      deltaStr = `<span style="color:${col}">${sign}${Math.round(delta)}</span>`;
    }
    return `<tr>
      <td style="font-weight:600;cursor:pointer" onclick="openPlayerPanel(${JSON.stringify(p).replace(/"/g,'&quot;')})">${p.name}</td>
      <td class="muted">${p.college || '—'}</td>
      <td class="muted">${p.draft_year || '—'}</td>
      <td class="muted">${p.overall_pick !== null ? `#${Math.round(p.overall_pick)}` : '—'}</td>
      <td class="muted">${p.draft_capital_score !== null ? fmt(p.draft_capital_score, 1) : '—'}</td>
      <td style="color:${orbColor};font-weight:600">${orb !== null && orb !== undefined ? Math.round(orb) : '—'}</td>
      <td style="color:${p1Color}">${p1 !== null && p1 !== undefined ? Math.round(p1) : '—'}</td>
      <td>${deltaStr}</td>
      <td class="${b2sClass}">${b2s !== null ? fmt(b2s, 1) + ' PPG' : '—'}</td>
    </tr>`;
  }).join('');
}

function initHistorical() {
  document.querySelectorAll('#hist-pos-btns .seg-btn').forEach(b => {
    b.addEventListener('click', () => {
      state.histPos = b.dataset.pos;
      document.querySelectorAll('#hist-pos-btns .seg-btn').forEach(x => x.classList.toggle('active', x.dataset.pos === b.dataset.pos));
      renderHistorical();
    });
  });
}

function sortHistorical(key) {
  if (state.histSort.key === key) {
    state.histSort.dir *= -1;
  } else {
    state.histSort.key = key;
    state.histSort.dir = -1;
  }
  renderHistorical();
}

// ═══════════════════════════════════════════════════════════
// ANALYSIS TAB
// ═══════════════════════════════════════════════════════════

function olsLine(pts) {
  // Returns { slope, intercept } for OLS regression through array of {x,y} points
  const n = pts.length;
  if (n < 2) return null;
  const xBar = pts.reduce((s, p) => s + p.x, 0) / n;
  const yBar = pts.reduce((s, p) => s + p.y, 0) / n;
  const ssXX = pts.reduce((s, p) => s + (p.x - xBar) ** 2, 0);
  if (ssXX === 0) return null;
  const slope = pts.reduce((s, p) => s + (p.x - xBar) * (p.y - yBar), 0) / ssXX;
  const intercept = yBar - slope * xBar;
  return { slope, intercept };
}

function destroyAnalysisCharts() {
  for (const key of Object.keys(state.analysisCharts)) {
    try { state.analysisCharts[key].destroy(); } catch(_) {}
  }
  state.analysisCharts = {};
}

function renderAnalysis() {
  destroyAnalysisCharts();
  const pos = state.analysisPos;
  const year = state.analysisYear;
  const hist = getHistorical(pos);
  const currentPlayers = getPlayers(year, pos);
  // Multi-year: all players from all selected years (for chart 2)
  const allCurrentPlayers = [...state.analysisYears].flatMap(yr => getPlayers(yr, pos));

  const darkGrid = 'rgba(45,49,72,0.5)';
  const tickColor = '#64748b';
  const axisTitleColor = '#94a3b8';
  const posColors = { WR: 'rgba(79,156,249,0.7)', RB: 'rgba(34,197,94,0.7)', TE: 'rgba(245,158,11,0.7)' };
  const posColor = posColors[pos] || 'rgba(148,163,184,0.65)';

  // ── Chart 1: Production by Year Out of High School ──────────────────────────
  // Shows metric vs year_out for all historical player-seasons (gray dots),
  // elite R1-R2 average (dashed line), and selected-year prospect best seasons (triangles).
  const c1 = document.getElementById('analysis-age-b2s');
  if (c1) {
    const sd = (DASHBOARD_DATA.season_data || {})[pos] || {};
    const bgPts  = sd.bg        || [];
    const eliteAvg = sd.elite_avg || {};
    const metricLbl = sd.metric_label || 'Production';
    const yMax = sd.y_max || 5;

    // Selected years (multi-year support)
    const yearColors = { '2024': 'rgba(167,139,250,0.9)', '2025': 'rgba(251,146,60,0.9)', '2026': 'rgba(239,68,68,0.9)' };

    const datasets = [
      {
        label: 'All historical',
        data: bgPts,
        backgroundColor: 'rgba(148,163,184,0.18)',
        pointRadius: 3, pointHoverRadius: 5,
        datalabels: { display: false },
      },
    ];

    // Elite average line (dashed)
    const eKeys = Object.keys(eliteAvg).map(Number).sort((a,b)=>a-b);
    if (eKeys.length >= 2) {
      datasets.push({
        label: 'R1-R2 avg',
        data: eKeys.map(yo => ({ x: yo, y: eliteAvg[String(yo)] })),
        type: 'line',
        borderColor: 'rgba(251,191,36,0.7)',
        borderDash: [6, 4],
        borderWidth: 2,
        pointRadius: 4,
        pointBackgroundColor: 'rgba(251,191,36,0.7)',
        fill: false,
        datalabels: { display: false },
      });
    }

    // One dataset per selected year (triangles, named)
    for (const yr of state.analysisYears) {
      const prospPts = ((sd.prospects || {})[yr] || []);
      if (!prospPts.length) continue;
      datasets.push({
        label: `${yr} class`,
        data: prospPts.map(p => ({ x: p.x, y: p.y, name: p.name })),
        backgroundColor: yearColors[yr] || 'rgba(239,68,68,0.85)',
        pointRadius: 7, pointStyle: 'triangle', pointHoverRadius: 10,
        datalabels: {
          display: ctx => {
            const d = ctx.dataset.data[ctx.dataIndex];
            return d && d.y >= yMax * 0.5;
          },
          formatter: (_, ctx) => {
            const d = ctx.dataset.data[ctx.dataIndex];
            if (!d || !d.name) return '';
            const parts = d.name.split(' ');
            return parts[parts.length - 1];
          },
          font: { size: 8 },
          color: yearColors[yr] || '#ef4444',
          anchor: 'end', align: 'top', offset: 3, clamp: true,
        },
      });
    }

    state.analysisCharts.ageb2s = new Chart(c1, {
      type: 'scatter',
      data: { datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#94a3b8', font: { size: 11 } } },
          tooltip: {
            callbacks: {
              label: ctx => {
                const d = ctx.raw;
                if (d.name) return `${d.name} — Year ${d.x}: ${Number(d.y).toFixed(3)}`;
                return `Year ${d.x}: ${Number(d.y).toFixed(3)}`;
              },
            },
            backgroundColor: 'rgba(26,29,39,0.97)', borderColor: '#2d3148', borderWidth: 1,
            titleColor: '#e2e8f0', bodyColor: '#94a3b8', padding: 10,
          },
          datalabels: { display: false },
        },
        scales: {
          x: {
            title: { display: true, text: 'Year Out of High School', color: axisTitleColor, font: { size: 11 } },
            grid: { color: darkGrid }, ticks: { color: tickColor, stepSize: 1 },
            min: 0.5, max: 5.5,
          },
          y: {
            title: { display: true, text: metricLbl, color: axisTitleColor, font: { size: 11 } },
            grid: { color: darkGrid }, ticks: { color: tickColor },
            min: 0, max: yMax,
          },
        },
      },
    });
  }

  // ── Chart 2: ORBIT vs Position Rank (current class) ─────────────────────────
  const c2 = document.getElementById('analysis-orbit-rank');
  if (c2) {
    const yearColors2 = { '2024': 'rgba(167,139,250,0.8)', '2025': 'rgba(251,146,60,0.8)', '2026': 'rgba(239,68,68,0.8)' };
    const allPts2 = allCurrentPlayers
      .filter(p => p.orbit_score !== null && p.position_rank !== null)
      .map(p => ({ x: p.position_rank, y: p.orbit_score, player: p }));
    const ols2 = olsLine(allPts2);
    const ds2 = [...state.analysisYears].map(yr => {
      const pts = getPlayers(yr, pos)
        .filter(p => p.orbit_score !== null && p.position_rank !== null)
        .map(p => ({ x: p.position_rank, y: p.orbit_score, player: p }));
      const baseColor = yearColors2[yr] || posColor;
      return {
        label: `${yr} ${pos}`,
        data: pts,
        backgroundColor: pts.map(p => {
          const d = p.player.capital_delta;
          if (d !== null && d < -15) return 'rgba(34,197,94,0.75)';
          if (d !== null && d > 15) return 'rgba(239,68,68,0.75)';
          return baseColor;
        }),
        pointRadius: 6, pointHoverRadius: 9,
        datalabels: {
          display: ctx => {
            const p = ctx.dataset.data[ctx.dataIndex].player;
            return p && p.orbit_score !== null && p.orbit_score >= 65;
          },
          formatter: (_, ctx) => {
            const p = ctx.dataset.data[ctx.dataIndex].player;
            if (!p) return '';
            const parts = p.name.split(' ');
            return parts[parts.length - 1];
          },
          font: { size: 8 }, color: '#94a3b8', anchor: 'end', align: 'right', offset: 3, clamp: true,
        },
      };
    });
    if (ols2) {
      const xs2 = allPts2.map(p => p.x);
      const xMin2 = Math.min(...xs2), xMax2 = Math.max(...xs2);
      ds2.push({
        label: 'Trend line',
        data: [{ x: xMin2, y: ols2.slope * xMin2 + ols2.intercept }, { x: xMax2, y: ols2.slope * xMax2 + ols2.intercept }],
        type: 'line', borderColor: 'rgba(255,255,255,0.2)', borderDash: [5, 4],
        borderWidth: 1.5, pointRadius: 0, fill: false, datalabels: { display: false },
      });
    }
    state.analysisCharts.orbitRank = new Chart(c2, {
      type: 'scatter',
      data: { datasets: ds2 },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => {
                const p = ctx.raw.player;
                if (!p) return '';
                return [`${p.name} (${p.college})`, `ORBIT: ${Math.round(p.orbit_score)}  Rank: #${Math.round(ctx.raw.x)}`, `Δ Capital: ${p.capital_delta !== null ? Number(p.capital_delta).toFixed(1) : '—'}`];
              },
            },
            backgroundColor: 'rgba(26,29,39,0.97)', borderColor: '#2d3148', borderWidth: 1,
            titleColor: '#e2e8f0', bodyColor: '#94a3b8', padding: 10,
          },
          datalabels: { display: false },
        },
        onClick: (evt, items) => {
          if (!items.length) return;
          const item = items[0];
          const raw = state.analysisCharts.orbitRank.data.datasets[item.datasetIndex].data[item.index];
          if (raw && raw.player) openPlayerPanel(raw.player);
        },
        scales: {
          x: { title: { display: true, text: 'Position Rank (Big Board)', color: axisTitleColor, font: { size: 11 } }, grid: { color: darkGrid }, ticks: { color: tickColor } },
          y: { title: { display: true, text: 'ORBIT Score', color: axisTitleColor, font: { size: 11 } }, grid: { color: darkGrid }, ticks: { color: tickColor } },
        },
      },
    });
  }

  // ── Chart 3: Draft Capital vs B2S (all training classes, all positions) ────────
  const c3 = document.getElementById('analysis-cap-b2s');
  const chart3Title = document.getElementById('chart3-title');
  if (chart3Title) chart3Title.textContent = `Draft Capital vs Actual B2S — Training Set (2014–2022, all positions)`;
  if (c3) {
    const allHist = ['WR', 'RB', 'TE'].flatMap(p =>
      (getHistorical(p) || [])
        .filter(h => h.draft_capital_score !== null && h.b2s_score !== null)
        .map(h => ({ x: h.draft_capital_score, y: h.b2s_score, pos: p, name: h.name }))
    );
    const ds3 = ['WR', 'RB', 'TE'].map(p => ({
      label: p,
      data: allHist.filter(h => h.pos === p),
      // F4d: dim non-selected positions; full opacity for selected
      backgroundColor: p === pos
        ? (posColors[p] || 'rgba(148,163,184,0.5)')
        : 'rgba(100,116,139,0.15)',
      pointRadius: p === pos ? 4 : 3, pointHoverRadius: 7,
      datalabels: { display: false },
    }));
    const olsAll = olsLine(allHist.map(h => ({ x: h.x, y: h.y })));
    if (olsAll) {
      const caps = allHist.map(h => h.x);
      const capMin = Math.min(...caps), capMax = Math.max(...caps);
      ds3.push({
        label: 'Trend line',
        data: [{ x: capMin, y: olsAll.slope * capMin + olsAll.intercept }, { x: capMax, y: olsAll.slope * capMax + olsAll.intercept }],
        type: 'line', borderColor: 'rgba(255,255,255,0.35)', borderDash: [5, 4],
        borderWidth: 2, pointRadius: 0, fill: false, datalabels: { display: false },
      });
    }
    state.analysisCharts.capB2s = new Chart(c3, {
      type: 'scatter',
      data: { datasets: ds3 },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#94a3b8', font: { size: 11 } } },
          tooltip: {
            callbacks: {
              label: ctx => {
                const d = ctx.raw;
                return d.name ? `${d.name} — Cap: ${Number(d.x).toFixed(1)}  B2S: ${Number(d.y).toFixed(1)}` : '';
              },
            },
            backgroundColor: 'rgba(26,29,39,0.97)', borderColor: '#2d3148', borderWidth: 1,
            titleColor: '#e2e8f0', bodyColor: '#94a3b8', padding: 10,
          },
          datalabels: { display: false },
        },
        scales: {
          x: { title: { display: true, text: 'Draft Capital Score', color: axisTitleColor, font: { size: 11 } }, grid: { color: darkGrid }, ticks: { color: tickColor } },
          y: { title: { display: true, text: 'Actual B2S PPG', color: axisTitleColor, font: { size: 11 } }, grid: { color: darkGrid }, ticks: { color: tickColor } },
        },
      },
    });
  }

  if (!hist.length && !currentPlayers.length) {
    document.querySelectorAll('#tab-analysis .chart-card-canvas').forEach(el => {
      el.innerHTML = '<p class="analysis-no-data">No data available. Run with --include-historical to load training set.</p>';
    });
  }
}

function initAnalysis() {
  document.querySelectorAll('#analysis-pos-btns .seg-btn').forEach(b => {
    b.addEventListener('click', () => {
      state.analysisPos = b.dataset.pos;
      document.querySelectorAll('#analysis-pos-btns .seg-btn').forEach(x => x.classList.toggle('active', x.dataset.pos === b.dataset.pos));
      if (state.activeTab === 'analysis') renderAnalysis();
    });
  });
  makeAnalysisYearBtns(document.getElementById('analysis-year-btns'));
}

// ═══════════════════════════════════════════════════════════
// ABOUT TAB
// ═══════════════════════════════════════════════════════════
function renderAbout() {
  const perf = DASHBOARD_DATA.model_performance || {};

  document.getElementById('perf-grid').innerHTML = ['WR', 'RB', 'TE'].map(pos => {
    const m = perf[pos] || {};
    return `<div class="perf-card">
      <h3>${pos}</h3>
      <div class="perf-stat"><span class="perf-stat-label">N (training)</span><span class="perf-stat-val">${m.n_train || '—'}</span></div>
      <div class="perf-stat"><span class="perf-stat-label">LOYO R²</span><span class="perf-stat-val">${m.loyo_r2 !== null && m.loyo_r2 !== undefined ? m.loyo_r2.toFixed(3) : '—'}</span></div>
      <div class="perf-stat"><span class="perf-stat-label">Spearman ρ</span><span class="perf-stat-val">${m.spearman_rho !== null && m.spearman_rho !== undefined ? m.spearman_rho.toFixed(3) : '—'}</span></div>
      <div class="perf-stat"><span class="perf-stat-label">Top-25 Hit Rate</span><span class="perf-stat-val">${m.top25_hit_rate !== null && m.top25_hit_rate !== undefined ? (m.top25_hit_rate * 100).toFixed(1) + '%' : '—'}</span></div>
    </div>`;
  }).join('');

  document.getElementById('feat-section').innerHTML = ['WR', 'RB', 'TE'].map(pos => {
    const m = perf[pos] || {};
    const capFeats = m.selected_features || [];
    const nocapFeats = m.nocap_features || [];
    return `<div style="margin-bottom:20px">
      <div style="font-weight:700;margin-bottom:8px;font-size:15px">${pos}</div>
      <div style="font-size:12px;color:var(--text-muted);margin-bottom:4px">Capital Model (${capFeats.length} features)</div>
      <div class="feat-chips">${capFeats.map(f => `<span class="feat-chip">${FL[f] || f}</span>`).join('')}</div>
      <div style="font-size:12px;color:var(--text-muted);margin-bottom:4px;margin-top:8px">Phase I / Talent-Only (${nocapFeats.length} features)</div>
      <div class="feat-chips">${nocapFeats.map(f => `<span class="feat-chip">${FL[f] || f}</span>`).join('')}</div>
    </div>`;
  }).join('');
}

// ═══════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════
(function init() {
  // Header meta
  const meta = DASHBOARD_DATA.meta;
  document.getElementById('header-meta').textContent =
    `Generated ${meta.generated} · ${meta.years.join(', ')}`;

  initRankings();
  initScatter();
  initHistorical();
  initAnalysis();
  // About / Analysis render on tab switch
})();
</script>
</body>
</html>
"""


def inject_data(html_template: str, data: dict, year: int) -> str:
    """Inject JSON data into the HTML template."""
    import re

    # Use a custom encoder to handle NaN/Inf
    class SafeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
                return None
            return super().default(obj)

        def encode(self, obj):
            # Walk and clean NaN floats before encoding
            return super().encode(obj)

    json_str = json.dumps(data, cls=SafeEncoder, indent=None, ensure_ascii=False)

    html = html_template.replace("{DASHBOARD_DATA_JSON}", json_str)
    html = html.replace("{title}", f"{year} Draft Class")
    return html


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export Dynasty Prospect Model HTML Dashboard")
    parser.add_argument(
        "--year", type=int, action="append", dest="years",
        help="Draft year to include (repeatable: --year 2025 --year 2026).",
    )
    parser.add_argument("--all-years", action="store_true", help="Include all available years (2023-2026)")
    parser.add_argument("--include-historical", action="store_true", help="Include 2011-2022 training players")
    args = parser.parse_args()

    if args.all_years:
        years = [2023, 2024, 2025, 2026]
    elif args.years:
        years = sorted(args.years)
    else:
        years = [2026]

    print(f"Building dashboard for year(s): {years}")
    print(f"  Include historical: {args.include_historical}")

    data = build_dashboard_data(years, include_historical=args.include_historical)

    # Create output directory
    DASH_DIR.mkdir(parents=True, exist_ok=True)

    # Generate HTML
    html = inject_data(HTML_TEMPLATE, data, years[-1])

    out_path = DASH_DIR / f"prospects_{years[-1]}.html"
    out_path.write_text(html, encoding="utf-8")

    size_kb = out_path.stat().st_size / 1024
    print(f"\nDashboard written to: {out_path}")
    print(f"File size: {size_kb:.1f} KB ({out_path.stat().st_size:,} bytes)")

    if size_kb < 50:
        print("WARNING: File is smaller than expected — check for data loading issues.")
    else:
        print("OK: File size looks good.")


if __name__ == "__main__":
    main()
