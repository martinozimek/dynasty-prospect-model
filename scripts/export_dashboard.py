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

    player = {
        "name": str(g("player_name", "")),
        "college": str(g("best_team", "")),
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
        "features": features,
    }
    return player


def build_historical(training_dfs: dict, dists: dict) -> dict:
    """Build historical player records from training CSVs (2011-2022, known outcomes)."""
    hist = {}
    for pos, df in training_dfs.items():
        records = []
        for _, row in df.iterrows():
            def g(col, default=None):
                if col not in row.index:
                    return default
                v = row[col]
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

            # Derive computed features from training df if available
            draft_cap = safe_float(g("draft_capital_score"))
            b2s = safe_float(g("b2s_score"))

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
                "features": features,
            })
        hist[pos] = records
    return hist


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
        historical = build_historical(training_dfs, dists)
        for pos, recs in historical.items():
            print(f"  Historical {pos}: {len(recs)} records")

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

    <!-- ── About Tab ── -->
    <div id="tab-about" class="tab-content">
      <div class="about-wrap">
        <div class="about-section">
          <h2>Dynasty Prospect Model</h2>
          <p>A Ridge regression model trained on 2011–2022 NFL draft classes to predict <strong>Best 2 of 3 Seasons (B2S)</strong> PPR fantasy points per game for WR, RB, and TE prospects. The model uses pre-draft college performance, athleticism, and draft capital signals.</p>
          <p><strong>ORBIT Score</strong> (0–100) is a calibrated composite that incorporates all model signals. <strong>Phase I ORBIT</strong> is the talent-only score (no draft capital). <strong>Capital Delta</strong> measures how much draft capital shifts the score.</p>
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
  scatterPos: 'WR',
  scatterYear: DASHBOARD_DATA.meta.primary_year,
  histPos: 'WR',
  sortCol: 'pos_rank',
  sortDir: 1,
  search: '',
  selectedPlayer: null,
  scatterChart: null,
};

// ═══════════════════════════════════════════════════════════
// UTILS
// ═══════════════════════════════════════════════════════════
const FL = DASHBOARD_DATA.feature_labels || {};
const INVERSE = new Set(DASHBOARD_DATA.inverse_features || []);
const BINARY = new Set(DASHBOARD_DATA.binary_features || []);

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
  if (lc.includes('low')) return `<span class="risk-badge risk-low">Low Risk</span>`;
  if (lc.includes('high')) return `<span class="risk-badge risk-high">High Risk</span>`;
  return `<span class="risk-badge risk-neutral">Neutral</span>`;
}

function deltaHtml(d) {
  if (d === null || d === undefined) return '<span class="delta-neutral">—</span>';
  const sign = d < -1 ? 'delta-neg' : d > 1 ? 'delta-pos' : 'delta-neutral';
  const prefix = d > 0 ? '+' : '';
  return `<span class="${sign}">${prefix}${Number(d).toFixed(1)}</span>`;
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

// ═══════════════════════════════════════════════════════════
// RANKINGS TAB
// ═══════════════════════════════════════════════════════════
const RANK_COLS = [
  { key: 'pos_rank', label: 'Rank', fmt: v => v !== null ? `#${Math.round(v)}` : '—' },
  { key: 'name', label: 'Player', fmt: v => `<span style="font-weight:600">${v}</span>` },
  { key: 'college', label: 'College', fmt: v => v || '—' },
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
  { key: 'phase1_orbit', label: 'Ph1 ORBIT', fmt: v => v !== null && v !== undefined ? `<span style="color:var(--text-muted)">${Math.round(v)}</span>` : '<span class="empty-dash">—</span>' },
  { key: 'capital_delta', label: 'Δ', fmt: v => deltaHtml(v) },
  { key: 'risk', label: 'Risk', fmt: v => riskBadge(v), noSort: true },
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
  document.getElementById('rankings-thead').innerHTML =
    `<tr>${RANK_COLS.map(c => `
      <th data-key="${c.key}" ${c.noSort ? '' : 'class="sortable"'}>
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
  let players = getPlayers(state.year, state.pos);

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

  tbody.innerHTML = players.map(p => {
    const isSelected = state.selectedPlayer && state.selectedPlayer.name === p.name && state.selectedPlayer.draft_year === p.draft_year;
    return `<tr class="${isSelected ? 'selected' : ''}" data-name="${p.name}" data-year="${p.draft_year}">
      ${RANK_COLS.map(c => {
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

  // Year buttons
  makeYearBtns(document.getElementById('year-btns'), 'year', renderRankingsBody);

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
      <div class="score-card-label">Phase I ORBIT</div>
      <div class="score-card-val" style="color:var(--text-muted)">${p.phase1_orbit !== null && p.phase1_orbit !== undefined ? Math.round(p.phase1_orbit) : '—'}</div>
      <div class="score-card-sub">Talent-only</div>
    </div>
    <div class="score-card">
      <div class="score-card-label">Δ Capital</div>
      <div class="score-card-val" style="font-size:18px">${deltaHtml(p.capital_delta)}</div>
      <div class="score-card-sub">${riskBadge(p.risk)}</div>
    </div>
  </div>`;

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

  // Capital model features
  if (capFeatures.length > 0) {
    html += `<div class="section-title">Capital Model Features</div>
    <div class="feat-list">
      ${capFeatures.map(f => featRow(f, p, pos)).join('')}
    </div>`;
  }

  // Phase I features
  if (nocapFeatures.length > 0) {
    html += `<div class="section-title">Phase I Model Features</div>
    <div class="feat-list">
      ${nocapFeatures.map(f => featRow(f, p, pos)).join('')}
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

function featRow(feature, player, pos) {
  const label = FL[feature] || feature;
  const featData = (player.features || {})[feature];

  // Binary feature
  if (BINARY.has(feature)) {
    const val = typeof featData === 'object' ? featData : featData;
    const display = val === 1 || val === true ? 'Yes' : val === 0 || val === false ? 'No' : '—';
    return `<div class="feat-row">
      <div class="feat-row-top">
        <span class="feat-label">${label}</span>
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
    // Special formats
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

  const tier = tierLabel(pct);
  const pctWidth = pct !== null ? Math.min(100, Math.max(0, pct)) : 0;
  const barColor = pctColor(pct);

  return `<div class="feat-row">
    <div class="feat-row-top">
      <span class="feat-label">${label}</span>
      <div style="display:flex;align-items:center;gap:8px">
        <span class="feat-tier ${tier.cls}">${tier.text}</span>
        <span class="feat-val">${displayVal}</span>
      </div>
    </div>
    ${pct !== null ? `<div class="pct-track"><div class="pct-fill ${barColor}" style="width:${pctWidth}%"></div></div>` : ''}
  </div>`;
}

// ═══════════════════════════════════════════════════════════
// SCATTER TAB
// ═══════════════════════════════════════════════════════════
function renderScatter() {
  const players = getPlayers(state.scatterYear, state.scatterPos);
  const canvas = document.getElementById('scatter-canvas');

  // Filter players with both scores
  const pts = players.filter(p => p.orbit_score !== null && p.phase1_orbit !== null);

  const colorMap = { 'Low Risk': '#22c55e', 'Neutral': '#94a3b8', 'High Risk': '#ef4444', null: '#64748b' };

  const datasets = [{
    label: 'Prospects',
    data: pts.map(p => ({ x: p.phase1_orbit, y: p.orbit_score, player: p })),
    backgroundColor: pts.map(p => {
      const r = p.risk || '';
      if (r.includes('Low')) return 'rgba(34,197,94,0.75)';
      if (r.includes('High')) return 'rgba(239,68,68,0.75)';
      return 'rgba(148,163,184,0.65)';
    }),
    pointRadius: 7,
    pointHoverRadius: 10,
  }];

  // Diagonal line
  const minV = Math.min(...pts.map(p => Math.min(p.phase1_orbit, p.orbit_score)), 0);
  const maxV = Math.max(...pts.map(p => Math.max(p.phase1_orbit, p.orbit_score)), 100);
  datasets.push({
    label: 'y=x line',
    data: [{ x: minV, y: minV }, { x: maxV, y: maxV }],
    type: 'line',
    borderColor: 'rgba(255,255,255,0.2)',
    borderDash: [6, 4],
    borderWidth: 1.5,
    pointRadius: 0,
    fill: false,
  });

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
              return [`${p.name} (${p.college})`, `ORBIT: ${Math.round(p.orbit_score)}  Ph1: ${Math.round(p.phase1_orbit)}  Δ: ${p.capital_delta !== null ? Number(p.capital_delta).toFixed(1) : '—'}`, `Risk: ${p.risk || '—'}`];
            },
          },
          backgroundColor: 'rgba(26,29,39,0.97)',
          borderColor: '#2d3148', borderWidth: 1,
          titleColor: '#e2e8f0', bodyColor: '#94a3b8',
          padding: 12,
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Phase I ORBIT (Talent Only)', color: '#94a3b8', font: { size: 12 } },
          grid: { color: 'rgba(45,49,72,0.5)' },
          ticks: { color: '#64748b' },
        },
        y: {
          title: { display: true, text: 'ORBIT Score (Full Model)', color: '#94a3b8', font: { size: 12 } },
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
}

// ═══════════════════════════════════════════════════════════
// HISTORICAL TAB
// ═══════════════════════════════════════════════════════════
function renderHistorical() {
  const pos = state.histPos;
  const players = getHistorical(pos);

  document.getElementById('hist-meta').textContent =
    players.length
      ? `${players.length} ${pos} prospects — 2011–2022 draft classes (known outcomes)`
      : 'No historical data loaded. Re-run export script with --include-historical flag.';

  const thead = document.getElementById('hist-thead');
  const tbody = document.getElementById('hist-tbody');

  thead.innerHTML = `<tr>
    <th>Player</th><th>College</th><th>Draft Yr</th>
    <th>Pick</th><th>Capital</th><th>Actual B2S</th>
  </tr>`;

  if (!players.length) { tbody.innerHTML = ''; return; }

  // Sort by b2s_score descending (best outcomes first)
  const sorted = players.slice().sort((a, b) => (b.b2s_score || 0) - (a.b2s_score || 0));

  tbody.innerHTML = sorted.map(p => {
    const b2s = p.b2s_score;
    const b2sClass = b2s === null ? '' : b2s >= 12 ? 'outcome-beat' : b2s >= 6 ? 'outcome-near' : 'outcome-miss';
    return `<tr>
      <td style="font-weight:600">${p.name}</td>
      <td class="muted">${p.college || '—'}</td>
      <td class="muted">${p.draft_year || '—'}</td>
      <td class="muted">${p.overall_pick !== null ? `#${Math.round(p.overall_pick)}` : '—'}</td>
      <td class="muted">${p.draft_capital_score !== null ? fmt(p.draft_capital_score, 1) : '—'}</td>
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
  // About renders on tab switch
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
    parser.add_argument("--year", type=int, default=2026, help="Primary draft year to export")
    parser.add_argument("--all-years", action="store_true", help="Include all available years (2023-2026)")
    parser.add_argument("--include-historical", action="store_true", help="Include 2011-2022 training players")
    args = parser.parse_args()

    if args.all_years:
        years = [2023, 2024, 2025, 2026]
    else:
        years = [args.year]

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
