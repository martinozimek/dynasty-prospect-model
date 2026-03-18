"""
Build position-specific training CSVs for the dynasty-prospect-model.

Joins:
  cfb-prospect-db:
    - CFBPlayerSeason  (college stats + derived metrics per season)
    - Player           (identity, weight, height)
    - NFLDraftPick     (draft capital score, round, overall pick)
    - NFLCombineResult (speed score, 40 time, weight at combine)
    - Recruiting       (247Sports composite rating)
    - CFBTeamSeason    (team context: SP+ rating, pass attempts)

  nfl-fantasy-db:
    - NFLB2S           (B2S training labels — target variable)
    - CFBLink          (NFL name -> CFB player_id bridge)
    - NFLBigBoard      (consensus pre-draft rank — 2016+ only)

Output:
    data/training_WR.csv
    data/training_RB.csv
    data/training_TE.csv

Feature set (per player — all pre-draft, no NFL data leakage):
  College performance (best qualifying season, min_games configurable):
    best_rec_rate           rec_yards_per_team_pass_att (best season)
    best_dominator          dominator_rating (best season)
    best_reception_share    reception_share (best season)
    best_age                age_at_season_start (best season)
    best_games              games_played (best season)
    best_sp_plus            SP+ rating of team in best season (proxy for competition level)

  Career college totals:
    career_seasons          number of seasons played
    career_rec_yards        total receiving yards
    career_rush_yards       total rushing yards
    career_targets          total targets
    career_rec_per_target   rec_yards / targets (TE YPRR proxy)
    early_declare           1 if career_seasons <= 3 (played ≤ 3 seasons)

  Combine / athleticism:
    weight_lbs              weight at combine (or Player record)
    forty_time              40-yard dash
    speed_score             (weight * 200) / (forty_time^4)

  Draft position:
    draft_capital_score     normalized draft value (0-100, stored in cfb-prospect-db)
    draft_round             round drafted
    overall_pick            overall pick number

  Recruiting:
    recruit_rating          247Sports composite (0.0-1.0)
    recruit_stars           star rating (1-5)

  Pre-draft market expectation:
    consensus_rank          overall consensus big board rank (lower = higher expected)
    position_rank           rank within position group

  Team context:
    teammate_score          sum of draft_capital_score of other WR/RB/TE draftees
                            from same school within ±2 years of this player's draft

Usage:
    python scripts/build_training_set.py
    python scripts/build_training_set.py --min-year 2016 --max-year 2022
    python scripts/build_training_set.py --min-games 6   # qualifying season threshold
    python scripts/build_training_set.py --verbose

Dependencies:
    Both cfb-prospect-db and nfl-fantasy-db must be populated first.
    CFB_DB_PATH and NFL_DB_PATH must be set in .env.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_cfb_db_path, get_data_dir, get_nfl_db_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_POSITIONS = ("WR", "RB", "TE")

# Power-4 conferences for conference tier feature
_POWER4_CONFS = frozenset({"SEC", "Big Ten", "ACC", "Big 12"})


# ---------------------------------------------------------------------------
# DB loader helpers
# ---------------------------------------------------------------------------

def _load_cfb(cfb_db_path: str) -> dict:
    """
    Load all relevant cfb-prospect-db tables into dicts keyed by player_id.
    Returns a dict-of-dicts with keys:
        players, seasons, draft_picks, combine, recruiting, team_seasons
    """
    cfb_root = Path(cfb_db_path).parent
    if str(cfb_root) not in sys.path:
        sys.path.insert(0, str(cfb_root))

    from ffdb.database import (
        CFBPlayerSeason,
        CFBTeamSeason,
        NFLCombineResult,
        NFLDraftPick,
        PFFPlayerSeason,
        Player,
        Recruiting,
        get_session,
    )

    result = {}

    with get_session(cfb_db_path) as s:
        # Players
        result["players"] = {
            p.id: {
                "full_name": p.full_name,
                "position": p.position,
                "weight_lbs": p.weight_lbs,
                "height_inches": p.height_inches,
            }
            for p in s.query(Player).all()
        }

        # College seasons — group by player_id
        seasons: dict[int, list[dict]] = defaultdict(list)
        for row in s.query(CFBPlayerSeason).all():
            seasons[row.player_id].append({
                "season_year": row.season_year,
                "team": row.team,
                "conference": row.conference,
                "games_played": row.games_played,
                "rec_yards": row.rec_yards,
                "targets": row.targets,
                "receptions": row.receptions,
                "rec_tds": row.rec_tds,
                "rush_yards": row.rush_yards,
                "rush_attempts": row.rush_attempts,
                "rush_tds": row.rush_tds,
                "rec_yards_per_team_pass_att": row.rec_yards_per_team_pass_att,
                "dominator_rating": row.dominator_rating,
                "reception_share": row.reception_share,
                "age_at_season_start": row.age_at_season_start,
                "ppa_avg_pass": row.ppa_avg_pass,
                "ppa_avg_overall": row.ppa_avg_overall,
                "ppa_avg_rush": row.ppa_avg_rush,
                "usage_overall": row.usage_overall,
                "usage_pass": row.usage_pass,
                "usage_rush": row.usage_rush,
            })
        # Team seasons — keyed by (team, year) for SP+ lookup
        team_seasons_raw = {
            (row.team, row.season_year): {
                "sp_plus_rating": row.sp_plus_rating,
                "pass_attempts": row.pass_attempts,
                "rush_attempts": row.rush_attempts,
            }
            for row in s.query(CFBTeamSeason).all()
        }
        result["team_seasons"] = team_seasons_raw

        # Compute team total receiving TDs per (team, year) for rec_td_pct feature.
        # Two-pass: aggregate from all loaded player season rows.
        team_rec_tds: dict[tuple, int] = defaultdict(int)
        for slist in seasons.values():
            for sn in slist:
                t, y = sn.get("team"), sn.get("season_year")
                if t and y:
                    team_rec_tds[(t, y)] += sn.get("rec_tds") or 0

        # Enrich each player season dict with team context so helpers
        # (_breakout_score, _total_yards_rate) can access SP+ and pass_att
        # without requiring a separate team_seasons lookup argument.
        for slist in seasons.values():
            for sn in slist:
                key = (sn.get("team"), sn.get("season_year"))
                ts = team_seasons_raw.get(key, {})
                sn["sp_plus_rating"]  = ts.get("sp_plus_rating")
                sn["team_pass_att"]   = ts.get("pass_attempts")
                sn["team_rush_att"]   = ts.get("rush_attempts")
                total_tds = team_rec_tds.get(key, 0)
                player_tds = sn.get("rec_tds") or 0
                sn["rec_td_pct"] = player_tds / total_tds if total_tds > 0 else None

        # Draft picks
        result["draft_picks"] = {
            row.player_id: {
                "draft_year": row.draft_year,
                "draft_round": row.draft_round,
                "overall_pick": row.overall_pick,
                "position_drafted": row.position_drafted,
                "nfl_team": row.nfl_team,
                "draft_capital_score": row.draft_capital_score,
            }
            for row in s.query(NFLDraftPick).all()
        }

        # Combine
        result["combine"] = {
            row.player_id: {
                "combine_weight_lbs": row.weight_lbs,
                "forty_time": row.forty_time,
                "speed_score": row.speed_score,
                "height_inches_combine": row.height_inches,
                "vertical_jump": row.vertical_jump,
                "broad_jump": row.broad_jump,
                "three_cone": row.three_cone,
                "shuttle": row.shuttle,
                "bench_press": row.bench_press,
            }
            for row in s.query(NFLCombineResult).all()
        }

        # Recruiting — take the best record per player (highest rating)
        recruiting: dict[int, dict] = {}
        for row in s.query(Recruiting).order_by(Recruiting.rating.desc()).all():
            if row.player_id not in recruiting:
                recruiting[row.player_id] = {
                    "recruit_rating": row.rating,
                    "recruit_stars": row.stars,
                    "recruit_year": row.recruit_year,
                    "recruit_rank_national": row.ranking_national,
                    "classification": row.classification,
                }
        result["recruiting"] = recruiting

        # PFF data — all seasons keyed by (player_id, season_year).
        # Merged into season dicts below as source-of-truth for counting stats.
        pff_by_season: dict[tuple, dict] = {}
        pff_row_count = 0
        for row in s.query(PFFPlayerSeason).all():
            # Parse PFF split JSON columns into summary scalars.
            # All JSON values are stored as strings — cast to float before arithmetic.
            def _f(v):
                """Convert JSON string value to float, or None if absent/blank."""
                try:
                    return float(v) if v is not None and v != "" else None
                except (ValueError, TypeError):
                    return None

            _splits: dict = {}
            _base_tgt = None
            if row.depth_data:
                try:
                    _d = json.loads(row.depth_data)
                    _base_tgt = _f(_d.get("base_targets"))
                    _splits["deep_yprr"] = _f(_d.get("deep_yprr"))
                    if _base_tgt:
                        _deep_tgt = _f(_d.get("deep_targets"))
                        _splits["deep_target_rate"] = (
                            round(_deep_tgt / _base_tgt, 4) if _deep_tgt is not None else None
                        )
                        _blos_tgt = _f(_d.get("behind_los_targets"))
                        _splits["behind_los_rate"] = (
                            round(_blos_tgt / _base_tgt, 4) if _blos_tgt is not None else None
                        )
                except Exception:
                    pass
            if row.concept_data:
                try:
                    _c = json.loads(row.concept_data)
                    _bt = _base_tgt or _f(_c.get("base_targets"))
                    _splits["slot_yprr"] = _f(_c.get("slot_yprr"))
                    if _bt:
                        _slot_tgt = _f(_c.get("slot_targets"))
                        _splits["slot_target_rate"] = (
                            round(_slot_tgt / _bt, 4) if _slot_tgt is not None else None
                        )
                        _scr_tgt = _f(_c.get("screen_targets"))
                        _splits["screen_rate"] = (
                            round(_scr_tgt / _bt, 4) if _scr_tgt is not None else None
                        )
                except Exception:
                    pass
            if row.scheme_data:
                try:
                    _sc = json.loads(row.scheme_data)
                    _man = _f(_sc.get("man_yprr"))
                    _zone = _f(_sc.get("zone_yprr"))
                    _splits["man_yprr"] = _man
                    _splits["zone_yprr"] = _zone
                    if _man is not None and _zone is not None:
                        _splits["man_zone_delta"] = round(_man - _zone, 4)
                except Exception:
                    pass
            pff_by_season[(row.player_id, row.season_year)] = {
                # Counting overrides (replace CFBD values where non-null)
                "targets":      row.targets,
                "receptions":   row.receptions,
                "rec_yards":    row.rec_yards,
                "rec_tds":      row.rec_tds,
                "games_played": row.games_played,
                "rush_yards":   row.rush_yards,
                "rush_attempts": row.rush_attempts,
                "rush_tds":     row.rush_tds,
                # PFF-specific efficiency metrics
                "yprr":                 row.yprr,
                "routes_run":           row.routes_run,
                "routes_per_game":      row.routes_per_game,
                "receiving_grade":      row.receiving_grade,
                "catch_rate":           row.catch_rate,
                "td_rate":              row.td_rate,
                "yards_per_reception":  row.yards_per_reception,
                "drop_rate":            row.drop_rate,
                "contested_catch_rate": row.contested_catch_rate,
                "target_separation":    row.target_separation,
                "offense_grade":        row.offense_grade,
                "rush_grade":           row.rush_grade,
                # PFF split scalars (depth / concept / scheme JSON columns)
                **_splits,
            }
            pff_row_count += 1

    # Merge PFF into season dicts (outside the session — data already extracted)
    pff_merged = 0
    for pid, slist in seasons.items():
        for sn in slist:
            pff_sn = pff_by_season.get((pid, sn.get("season_year")))
            if pff_sn is None:
                continue
            # Override counting stats with PFF where PFF is non-null
            for field in ("targets", "receptions", "rec_yards", "rec_tds", "games_played",
                          "rush_yards", "rush_attempts", "rush_tds"):
                pff_val = pff_sn.get(field)
                if pff_val is not None:
                    sn[field] = pff_val
            # Embed PFF efficiency metrics directly on season dict
            for field in ("yprr", "routes_run", "routes_per_game", "receiving_grade",
                          "catch_rate", "td_rate", "yards_per_reception", "drop_rate",
                          "contested_catch_rate", "target_separation", "offense_grade",
                          "rush_grade",
                          # PFF split scalars (depth / concept / scheme)
                          "deep_yprr", "deep_target_rate", "behind_los_rate",
                          "slot_yprr", "slot_target_rate", "screen_rate",
                          "man_yprr", "zone_yprr", "man_zone_delta"):
                sn[field] = pff_sn.get(field)
            # Recompute rec_yards_per_team_pass_att using PFF rec_yards
            pff_rec_yards = pff_sn.get("rec_yards")
            team_pass_att = sn.get("team_pass_att")
            if pff_rec_yards is not None and team_pass_att and team_pass_att > 0:
                sn["rec_yards_per_team_pass_att"] = round(pff_rec_yards / team_pass_att, 6)
            pff_merged += 1

    result["seasons"] = dict(seasons)

    logger.info(
        "CFB loaded: %d players, %d season groups, %d draft picks, %d combine rows; "
        "PFF merged into %d/%d season rows",
        len(result["players"]),
        len(result["seasons"]),
        len(result["draft_picks"]),
        len(result["combine"]),
        pff_merged, sum(len(v) for v in seasons.values()),
    )
    return result


def _load_nfl(nfl_db_path: str) -> dict:
    """
    Load nfl-fantasy-db tables.
    Returns dict with keys: b2s, links, bigboard.
    """
    nfl_root = Path(nfl_db_path).parent
    if str(nfl_root) not in sys.path:
        sys.path.insert(0, str(nfl_root))

    from ffnfl.database import CFBLink, NFLB2S, NFLBigBoard, get_session

    result = {}

    with get_session(nfl_db_path) as s:
        # B2S labels — keyed by (player_name, position, draft_year)
        result["b2s"] = {
            (row.player_name, row.position, row.draft_year): {
                "b2s_score": row.b2s_score,
                "year1_ppg": row.year1_ppg,
                "year2_ppg": row.year2_ppg,
                "year3_ppg": row.year3_ppg,
                "qualifying_seasons": row.qualifying_seasons,
            }
            for row in s.query(NFLB2S).all()
        }

        # CFB links — keyed by (player_name, position)
        result["links"] = {
            (row.nfl_player_name, row.position): {
                "cfb_player_id": row.cfb_player_id,
                "cfb_full_name": row.cfb_full_name,
                "draft_year": row.draft_year,
                "match_score": row.match_score,
            }
            for row in s.query(CFBLink).all()
        }

        # Big board — keyed by (player_name, draft_year)
        result["bigboard"] = {
            (row.player_name, row.draft_year): {
                "consensus_rank": row.consensus_rank,
                "position_rank": row.position_rank,
            }
            for row in s.query(NFLBigBoard).all()
        }

    logger.info(
        "NFL loaded: %d B2S labels, %d CFB links, %d big board rows",
        len(result["b2s"]),
        len(result["links"]),
        len(result["bigboard"]),
    )
    return result


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _best_season(
    seasons: list[dict],
    metric: str = "rec_yards_per_team_pass_att",
    min_games: int = 6,
) -> dict | None:
    """Return the season dict with the highest value of *metric* among qualifying seasons."""
    qualifying = [s for s in seasons if (s.get("games_played") or 0) >= min_games]
    if not qualifying:
        return None
    return max(qualifying, key=lambda s: s.get(metric) or 0.0)


def _best_pff_metric(
    seasons: list[dict],
    metric_key: str,
    min_games: int = 6,
    higher_is_better: bool = True,
    min_routes: int | None = None,
) -> float | None:
    """Best value of a PFF metric across all qualifying seasons (independent of rec_rate season).
    JJ principle: each metric is evaluated at its peak, not locked to the best volume season."""
    best = None
    for s in seasons:
        if (s.get("games_played") or 0) < min_games:
            continue
        if min_routes is not None and (s.get("routes_run") or 0) < min_routes:
            continue
        val = s.get(metric_key)
        if val is None:
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue
        if best is None:
            best = val
        elif higher_is_better and val > best:
            best = val
        elif not higher_is_better and val < best:
            best = val
    return round(best, 4) if best is not None else None


# CFBD dominator_rating = (rec_yards_pct + rec_td_pct) / 2 — receiving production only.
# For WR/TE, 20% is JJ Zachariason's standard breakout threshold.
# For RB, CFBD dominator is receiving-only (does not include rushing), so the proper
# JJ threshold (15%) is too high — only ~6% of RBs clear it. Use 0.10 (top-quartile
# receiving RBs) as the breakout signal: first season where a RB posted 10%+ receiving
# dominator = early "pass-catching back" signal in college.
_BREAKOUT_THRESHOLD = {"WR": 0.20, "TE": 0.20, "RB": 0.10}


def _breakout_age(
    seasons: list[dict],
    position: str,
    min_games: int = 6,
) -> float | None:
    """
    Return the age at which the player first posted a dominator_rating at or above
    the position threshold in a qualifying season (>= min_games).

    JJ Zachariason's definition: breakout before age 20 is elite signal.
    Distinct from best_age (age at highest production season).
    Returns None if the player never crossed the threshold.
    """
    threshold = _BREAKOUT_THRESHOLD.get(position, 0.20)
    qualifying = sorted(
        [s for s in seasons if (s.get("games_played") or 0) >= min_games],
        key=lambda s: s.get("season_year") or 9999,
    )
    for s in qualifying:
        dom = s.get("dominator_rating") or 0.0
        if dom >= threshold:
            return s.get("age_at_season_start")
    return None


# Position-specific age threshold for breakout_score (set from Phase A analysis).
# Phase A univariate R² sweep (analyze_breakout.py, 2026-03-18):
#   WR: rec_rate×age(24) R²=0.133 > bs_pff_24 R²=0.125 > current(age26) R²=0.122
#   RB: bs_pff_28 R²=0.086 >> current R²=0.057  (high T rewards late-breaking RBs)
#   TE: current CFBD formula R²=0.167 > bs_pff_27 R²=0.160  (PFF costs 20 obs, hurts)
_BREAKOUT_AGE_THRESHOLD = {
    "WR": 24,   # Phase A result: T=24 optimal (younger WRs score highest)
    "RB": 28,   # Phase A result: T=28 optimal (late-breaking RBs most valuable)
    "TE": 26,   # Phase A result: CFBD formula wins; T=26 = current best
}

# Positions that use CFBD rec_rate exclusively (no PFF yprr base).
# Phase A: TE CFBD formula (R²=0.167) > PFF formula (R²=0.160) AND mixed
# PFF/CFBD scale distorts breakout_score_x_capital interaction for TEs.
_BREAKOUT_CFBD_ONLY = {"TE"}


def _breakout_score(seasons: list[dict], position: str,
                    min_games: int = 6) -> float | None:
    """
    Position-specific Breakout Score (Ep 1083 + Phase B redesign).

    Base metric:
      WR/RB: yprr (PFF, season-locked to qualifying seasons) → CFBD fallback
      TE:    rec_yards_per_team_pass_att (CFBD always — Phase A: CFBD beats PFF
             R²=0.167 vs 0.160; mixed scale distorts breakout_score_x_capital)

    Formula:
      efficiency  = yprr [PFF/WR/RB] OR rec_rate [CFBD/all]
      × SOS mult  = max(0.70, min(1.30, 1 + (sp_plus − 5) / 100))
      × age mult  = max(0, T(pos) − age_at_season_start)
        where T = _BREAKOUT_AGE_THRESHOLD[position]

    Taken as MAXIMUM across all qualifying seasons (≥ min_games).
    Season dicts must be pre-enriched with sp_plus_rating, team_pass_att, and yprr
    (done in _load_cfb() for training data).
    """
    threshold = _BREAKOUT_AGE_THRESHOLD.get(position, 26)
    use_cfbd_only = position in _BREAKOUT_CFBD_ONLY
    best = None
    for s in seasons:
        if (s.get("games_played") or 0) < min_games:
            continue

        if use_cfbd_only:
            # TE: always use CFBD rec_rate (consistent scale across training set)
            team_pass_att = s.get("team_pass_att") or 0
            if team_pass_att < 200:
                continue
            efficiency = s.get("rec_yards_per_team_pass_att") or 0.0
        else:
            # WR/RB: PFF-first, fall back to CFBD when yprr unavailable
            yprr_raw = s.get("yprr")
            if yprr_raw is not None:
                try:
                    efficiency = float(yprr_raw)
                except (ValueError, TypeError):
                    efficiency = None
            else:
                efficiency = None

            if efficiency is None:
                # CFBD fallback: rec_yards / team_pass_att
                team_pass_att = s.get("team_pass_att") or 0
                if team_pass_att < 200:   # skip option-offense seasons (Navy ~100, FBS avg ~380)
                    continue
                efficiency = s.get("rec_yards_per_team_pass_att") or 0.0

        age = s.get("age_at_season_start")
        if age is None:
            continue
        sp_plus = s.get("sp_plus_rating")
        sos_mult = (
            max(0.70, min(1.30, 1.0 + (sp_plus - 5.0) / 100.0))
            if sp_plus is not None else 1.0
        )
        score = efficiency * sos_mult * max(0.0, float(threshold) - age)
        if best is None or score > best:
            best = score
    return round(best, 4) if best is not None else None


def _total_yards_rate(seasons: list[dict], min_games: int = 6) -> float | None:
    """
    JJ's 'adjusted yards per team play' for RBs (Ep 1081, Feb 2026).

    Formula:
      base  = (rec_yards + rush_yards) / team_pass_attempts
              [team rush_att not stored; pass_att is closest available proxy]
      × SOS mult  = max(0.70, min(1.30, 1 + (sp_plus − 5) / 100))
      × age mult  = max(0, 26 − age_at_season_start)

    Taken as MAXIMUM across qualifying seasons (≥ min_games).
    Captures total RB contribution relative to team passing volume,
    age- and schedule-adjusted.
    min_team_pass_att (200) guards against option-offense inflation (Navy etc.).
    """
    best = None
    for s in seasons:
        if (s.get("games_played") or 0) < min_games:
            continue
        team_pass_att = s.get("team_pass_att")
        if not team_pass_att or team_pass_att < 200:
            continue
        total_yards = (s.get("rec_yards") or 0) + (s.get("rush_yards") or 0)
        age = s.get("age_at_season_start")
        if age is None:
            continue
        sp_plus = s.get("sp_plus_rating")
        sos_mult = (
            max(0.70, min(1.30, 1.0 + (sp_plus - 5.0) / 100.0))
            if sp_plus is not None else 1.0
        )
        rate = (total_yards / team_pass_att) * sos_mult * max(0.0, 26.0 - age)
        if best is None or rate > best:
            best = rate
    return round(best, 4) if best is not None else None


def _total_yards_rate_v2(seasons: list[dict], min_games: int = 6) -> float | None:
    """
    Phase D variant: denominator = team_pass_att + team_rush_att (total team plays).
    Otherwise identical to _total_yards_rate().
    If team_rush_att is unavailable, falls back to pass_att-only denominator.
    """
    best = None
    for s in seasons:
        if (s.get("games_played") or 0) < min_games:
            continue
        team_pass_att = s.get("team_pass_att")
        if not team_pass_att or team_pass_att < 200:
            continue
        team_rush_att = s.get("team_rush_att") or 0
        total_plays = team_pass_att + team_rush_att if team_rush_att else team_pass_att
        total_yards = (s.get("rec_yards") or 0) + (s.get("rush_yards") or 0)
        age = s.get("age_at_season_start")
        if age is None:
            continue
        sp_plus = s.get("sp_plus_rating")
        sos_mult = (
            max(0.70, min(1.30, 1.0 + (sp_plus - 5.0) / 100.0))
            if sp_plus is not None else 1.0
        )
        rate = (total_yards / total_plays) * sos_mult * max(0.0, 26.0 - age)
        if best is None or rate > best:
            best = rate
    return round(best, 4) if best is not None else None


def _compute_teammate_score(
    player_id: int,
    draft_year: int,
    best_team: str | None,
    cfb: dict,
    look_back: int = 2,   # Part 5: count teammates drafted ≤ look_back years BEFORE
    look_ahead: int = 0,  # Part 5: count teammates drafted ≤ look_ahead years AFTER
) -> float:
    """
    Teammate score: sum of draft_capital_score for other WR/RB/TE draftees
    from the same school within the prospect's productive window.

    Part 5 fix: JJ considers only the last 2 years of a player's career. The old
    symmetric ±2 window included teammates drafted AFTER the prospect (who may never
    have overlapped at school). Asymmetric look_back=2, look_ahead=0 fixes this:
    only count teammates who were drafted BEFORE or in the same year as the prospect.

    Signals offensive context quality — a player from a school that produced
    multiple drafted skill players was in a richer passing environment.
    """
    if not best_team:
        return 0.0

    score = 0.0
    for pid, pick in cfb["draft_picks"].items():
        if pid == player_id:
            continue
        if not pick.get("position_drafted") or pick["position_drafted"] not in _POSITIONS:
            continue
        teammate_draft_year = pick.get("draft_year") or 0
        # Asymmetric window: teammate was drafted at most look_back years before
        # the prospect (overlapped at school) and at most look_ahead years after.
        diff = teammate_draft_year - draft_year
        if not (-look_back <= diff <= look_ahead):
            continue
        # Check if this player had a season at the same school
        for season in cfb["seasons"].get(pid, []):
            if season.get("team") == best_team:
                score += (pick.get("draft_capital_score") or 0.0)
                break

    return round(score, 2)


def _build_row(
    nfl_name: str,
    position: str,
    draft_year: int,
    b2s: dict,
    link: dict,
    cfb: dict,
    nfl: dict,
    min_games: int,
) -> dict | None:
    """
    Assemble one training row for a player. Returns None if critical data is missing.
    """
    cfb_player_id = link["cfb_player_id"]

    player = cfb["players"].get(cfb_player_id)
    if player is None:
        return None

    seasons_raw = cfb["seasons"].get(cfb_player_id, [])
    if not seasons_raw:
        return None

    best = _best_season(seasons_raw, min_games=min_games)
    if best is None:
        return None

    best_team = best.get("team")
    best_team_season = cfb["team_seasons"].get((best_team, best.get("season_year")), {})

    draft_pick = cfb["draft_picks"].get(cfb_player_id, {})
    combine = cfb["combine"].get(cfb_player_id, {})
    recruiting = cfb["recruiting"].get(cfb_player_id, {})

    # Big board — try exact name, then cfb_full_name
    board_key = (nfl_name, draft_year)
    board = nfl["bigboard"].get(board_key) or nfl["bigboard"].get((link.get("cfb_full_name", ""), draft_year), {})

    # Career totals
    career_seasons = len(seasons_raw)
    career_rec_yards = sum(s.get("rec_yards") or 0 for s in seasons_raw)
    career_rush_yards = sum(s.get("rush_yards") or 0 for s in seasons_raw)
    career_targets = sum(s.get("targets") or 0 for s in seasons_raw)
    career_receptions = sum(s.get("receptions") or 0 for s in seasons_raw)
    career_rush_attempts = sum(s.get("rush_attempts") or 0 for s in seasons_raw)
    career_rec_tds = sum(s.get("rec_tds") or 0 for s in seasons_raw)
    career_rush_tds = sum(s.get("rush_tds") or 0 for s in seasons_raw)
    career_total_tds = career_rec_tds + career_rush_tds
    career_rec_per_target = (
        career_rec_yards / career_targets if career_targets > 0 else None
    )

    # Breakout age (kept for diagnostic reference)
    breakout_age = _breakout_age(seasons_raw, position, min_games=min_games)
    # Breakout score: PFF-first (yprr × SOS × age(T_pos)), CFBD fallback
    best_breakout_score = _breakout_score(seasons_raw, position=position, min_games=min_games)
    # Total yards rate: RB primary input ((rec+rush yds)/team_pass_att × SOS mult × age mult)
    best_total_yards_rate = _total_yards_rate(seasons_raw, min_games=min_games)
    # Phase D variant: denominator = total team plays (pass_att + rush_att)
    best_total_yards_rate_v2 = _total_yards_rate_v2(seasons_raw, min_games=min_games)

    # Best season per-play rates
    best_targets_val = best.get("targets") or 0
    best_receptions_val = best.get("receptions") or 0
    best_rec_yards_val = best.get("rec_yards") or 0
    best_rec_tds_val = best.get("rec_tds") or 0
    best_rush_yards_val = best.get("rush_yards") or 0
    best_rush_attempts_val = best.get("rush_attempts") or 0
    best_rush_tds_val = best.get("rush_tds") or 0
    best_games_val = best.get("games_played") or 0

    best_catch_rate = (
        best_receptions_val / best_targets_val if best_targets_val > 0 else None
    )
    best_td_rate = (
        best_rec_tds_val / best_targets_val if best_targets_val > 0 else None
    )
    best_ypr = (
        best_rec_yards_val / best_receptions_val if best_receptions_val > 0 else None
    )
    best_ypt = (
        best_rec_yards_val / best_targets_val if best_targets_val > 0 else None
    )
    best_rush_ypc = (
        best_rush_yards_val / best_rush_attempts_val if best_rush_attempts_val > 0 else None
    )
    best_yards_per_touch = (
        (best_rec_yards_val + best_rush_yards_val) / (best_targets_val + best_rush_attempts_val)
        if (best_targets_val + best_rush_attempts_val) > 0
        else None
    )
    college_fantasy_ppg = (
        (
            best_receptions_val * 1.0
            + best_rec_yards_val / 10.0
            + best_rec_tds_val * 6.0
            + best_rush_yards_val / 10.0
            + best_rush_tds_val * 6.0
        ) / best_games_val
        if best_games_val > 0
        else None
    )

    # Teammate score
    teammate_score = _compute_teammate_score(
        cfb_player_id, draft_year, best_team, cfb
    )

    # Weight: prefer combine, fall back to Player record
    weight = combine.get("combine_weight_lbs") or player.get("weight_lbs")

    # B2S label
    b2s_key = (nfl_name, position, draft_year)
    b2s_data = b2s.get(b2s_key, {})

    return {
        # Identity
        "nfl_name": nfl_name,
        "cfb_name": link.get("cfb_full_name"),
        "cfb_player_id": cfb_player_id,
        "position": position,
        "draft_year": draft_year,
        "match_score": link.get("match_score"),
        # Target
        "b2s_score": b2s_data.get("b2s_score"),
        "year1_ppg": b2s_data.get("year1_ppg"),
        "year2_ppg": b2s_data.get("year2_ppg"),
        "year3_ppg": b2s_data.get("year3_ppg"),
        "qualifying_nfl_seasons": b2s_data.get("qualifying_seasons"),
        # Best college season — core ORBIT metrics
        "best_rec_rate": best.get("rec_yards_per_team_pass_att"),
        "best_dominator": best.get("dominator_rating"),
        "best_reception_share": best.get("reception_share"),
        "best_age": best.get("age_at_season_start"),
        "best_games": best.get("games_played"),
        "best_season_year": best.get("season_year"),
        "best_team": best_team,
        "best_sp_plus": best_team_season.get("sp_plus_rating"),
        "best_ppa_pass": best.get("ppa_avg_pass"),
        "best_ppa_overall": best.get("ppa_avg_overall"),
        "best_ppa_rush": best.get("ppa_avg_rush"),
        "best_usage": best.get("usage_overall"),
        "best_usage_pass": best.get("usage_pass"),
        "best_usage_rush": best.get("usage_rush"),
        # Best season counting stats
        "best_rec_yards": best_rec_yards_val if best_rec_yards_val > 0 else None,
        "best_receptions": best_receptions_val if best_receptions_val > 0 else None,
        "best_targets": best_targets_val if best_targets_val > 0 else None,
        "best_rec_tds": best_rec_tds_val,
        "best_rush_tds": best_rush_tds_val,
        # Best season per-play rates
        "best_catch_rate": best_catch_rate,
        "best_td_rate": best_td_rate,
        "best_ypr": best_ypr,
        "best_ypt": best_ypt,
        "best_rush_ypc": best_rush_ypc,
        "best_yards_per_touch": best_yards_per_touch,
        "college_fantasy_ppg": college_fantasy_ppg,
        # Breakout age (diagnostic reference — not used in models)
        "breakout_age": breakout_age,
        # Breakout score: WR primary input (rec_rate × SOS mult × age mult)
        "best_breakout_score": best_breakout_score,
        # Total yards rate: RB primary input ((rec+rush)/team_pass_att × SOS mult × age mult)
        "best_total_yards_rate": best_total_yards_rate,
        # Phase D variant: total team plays denominator (pass_att + rush_att)
        "best_total_yards_rate_v2": best_total_yards_rate_v2,
        # Career totals
        "career_seasons": career_seasons,
        "career_rec_yards": career_rec_yards,
        "career_rush_yards": career_rush_yards,
        "career_targets": career_targets,
        "career_receptions": career_receptions,
        "career_rush_attempts": career_rush_attempts,
        "career_rec_tds": career_rec_tds,
        "career_rush_tds": career_rush_tds,
        "career_total_tds": career_total_tds,
        "career_rec_per_target": career_rec_per_target,
        # Prefer calendar-year calculation to avoid bias from missing injury seasons.
        # JUCO players: use FBS season count, not all-college years.
        "early_declare": (
            int(draft_year - recruiting.get("recruit_year") <= 3)
            if recruiting.get("recruit_year") is not None
               and recruiting.get("classification") != "JUCO"
            else int(career_seasons <= 3)
        ),
        # Age at draft day — JJ's "under-22 on draft day" threshold.
        # Formula: 18.5 + (draft_year - recruit_year). Distinct from best_age
        # (age at best college season). High collinearity with best_age (~0.85);
        # Lasso will select at most one. JUCO: None (different recruitment timeline).
        "age_at_draft": (
            round(18.5 + (draft_year - recruiting.get("recruit_year")), 1)
            if recruiting.get("recruit_year") is not None
               and recruiting.get("classification") != "JUCO"
            else None
        ),
        # Combine
        "weight_lbs": weight,
        "forty_time": combine.get("forty_time"),
        "speed_score": combine.get("speed_score"),
        "height_inches": combine.get("height_inches_combine") or player.get("height_inches"),
        "vertical_jump": combine.get("vertical_jump"),
        "broad_jump": combine.get("broad_jump"),
        "three_cone": combine.get("three_cone"),
        "shuttle": combine.get("shuttle"),
        "bench_press": combine.get("bench_press"),
        # Draft
        "draft_capital_score": draft_pick.get("draft_capital_score"),
        "draft_round": draft_pick.get("draft_round"),
        "overall_pick": draft_pick.get("overall_pick"),
        # Recruiting
        "recruit_rating": recruiting.get("recruit_rating"),
        "recruit_stars": recruiting.get("recruit_stars"),
        "recruit_year": recruiting.get("recruit_year"),
        "recruit_rank_national": recruiting.get("recruit_rank_national"),
        # PFF base metrics — season-locked to the best rec_rate season (same context as
        # breakout_score denominator). Independent-peak approach was noisy: a player with
        # 25 routes in a freshman year showed an inflated "best_yprr" (Part 0 fix).
        "best_yprr":                 best.get("yprr"),
        "best_routes_per_game":      best.get("routes_per_game"),
        "best_receiving_grade":      best.get("receiving_grade"),
        "best_contested_catch_rate": best.get("contested_catch_rate"),
        "best_drop_rate":            best.get("drop_rate"),
        "best_target_sep":           best.get("target_separation"),
        # PFF split scalars — independent maxima; min_routes=50 guards noisy small samples
        "best_deep_yprr":            _best_pff_metric(seasons_raw, "deep_yprr",         min_routes=50),
        "best_deep_target_rate":     _best_pff_metric(seasons_raw, "deep_target_rate",  min_routes=50),
        "best_behind_los_rate":      _best_pff_metric(seasons_raw, "behind_los_rate"),
        "best_slot_yprr":            _best_pff_metric(seasons_raw, "slot_yprr",         min_routes=50),
        "best_slot_target_rate":     _best_pff_metric(seasons_raw, "slot_target_rate",  min_routes=50),
        "best_screen_rate":          _best_pff_metric(seasons_raw, "screen_rate"),
        "best_man_yprr":             _best_pff_metric(seasons_raw, "man_yprr",          min_routes=50),
        "best_zone_yprr":            _best_pff_metric(seasons_raw, "zone_yprr",         min_routes=50),
        "best_man_zone_delta":       _best_pff_metric(seasons_raw, "man_zone_delta",    min_routes=50),
        # Pre-draft market expectation
        "consensus_rank": board.get("consensus_rank"),
        "position_rank": board.get("position_rank"),
        # Team context
        "teammate_score": teammate_score,
        # Conference tier (JJ conference factor)
        "power4_conf": int(best.get("conference") in _POWER4_CONFS) if best.get("conference") else None,
        # TD share in best season (JJ's sub-20% flag for WRs)
        "best_rec_td_pct": best.get("rec_td_pct"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build training CSVs from cfb-prospect-db + nfl-fantasy-db."
    )
    parser.add_argument("--cfb-db", type=str, default=None)
    parser.add_argument("--nfl-db", type=str, default=None)
    parser.add_argument(
        "--min-year", type=int, default=2014,
        help="Minimum draft year to include (default: 2014 — first year with full "
             "PFF season coverage; pre-2014 rows use median-imputed PFF values "
             "that create era bias in the model).",
    )
    parser.add_argument(
        "--max-year", type=int, default=2022,
        help="Maximum draft year to include (default: 2022, requires >=3 NFL seasons).",
    )
    parser.add_argument(
        "--min-games", type=int, default=6,
        help="Minimum college games to qualify a season for 'best season' metrics (default: 6).",
    )
    parser.add_argument(
        "--min-link-score", type=float, default=80.0,
        help="Drop training rows whose CFB link match_score is below this threshold "
             "(default: 80). Rows below this are likely wrong links.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-player details.",
    )
    args = parser.parse_args()

    cfb_db_path = args.cfb_db or get_cfb_db_path()
    nfl_db_path = args.nfl_db or get_nfl_db_path()
    data_dir = get_data_dir()

    cfb = _load_cfb(cfb_db_path)
    nfl = _load_nfl(nfl_db_path)

    # Build rows
    rows_by_pos: dict[str, list[dict]] = {pos: [] for pos in _POSITIONS}
    skipped_no_link = skipped_no_seasons = skipped_no_b2s = 0

    for (nfl_name, pos, draft_year), b2s_data in nfl["b2s"].items():
        if pos not in _POSITIONS:
            continue
        if draft_year is None or not (args.min_year <= draft_year <= args.max_year):
            continue

        link = nfl["links"].get((nfl_name, pos))
        if link is None:
            skipped_no_link += 1
            if args.verbose:
                logger.debug("  No CFB link: %s %s %s", nfl_name, pos, draft_year)
            continue

        row = _build_row(
            nfl_name=nfl_name,
            position=pos,
            draft_year=draft_year,
            b2s=nfl["b2s"],
            link=link,
            cfb=cfb,
            nfl=nfl,
            min_games=args.min_games,
        )
        if row is None:
            skipped_no_seasons += 1
            if args.verbose:
                logger.debug("  No qualifying CFB season: %s %s %s", nfl_name, pos, draft_year)
            continue

        rows_by_pos[pos].append(row)

    logger.info(
        "Skipped: %d no CFB link, %d no qualifying college season",
        skipped_no_link, skipped_no_seasons,
    )

    # Deduplicate by cfb_player_id within each position.
    # When multiple NFL player names share the same CFB player ID — either because
    # nflverse records the same player under two name spellings (e.g. "Laviska Shenault"
    # and "Laviska Shenault Jr."), or because a fuzzy match wrongly links a different
    # NFL player to the same CFB record — keep only the highest-match-score row.
    # Tie-break: higher b2s_score (the name variant with more complete NFL data).
    dedup_dropped = 0
    for pos in _POSITIONS:
        seen: dict[int, dict] = {}
        for r in rows_by_pos[pos]:
            pid = r["cfb_player_id"]
            if pid not in seen:
                seen[pid] = r
            else:
                prev = seen[pid]
                this_better = (r["match_score"] or 0) > (prev["match_score"] or 0) or (
                    (r["match_score"] or 0) == (prev["match_score"] or 0)
                    and (r["b2s_score"] or 0) > (prev["b2s_score"] or 0)
                )
                loser, winner = (prev, r) if this_better else (r, prev)
                logger.warning(
                    "  [dedup %s] dropping %r (score=%.0f, b2s=%.2f) — "
                    "same cfb_player_id as %r (score=%.0f, b2s=%.2f)",
                    pos,
                    loser["nfl_name"], loser["match_score"] or 0, loser["b2s_score"] or 0,
                    winner["nfl_name"], winner["match_score"] or 0, winner["b2s_score"] or 0,
                )
                seen[pid] = winner
                dedup_dropped += 1
        rows_by_pos[pos] = list(seen.values())
    if dedup_dropped:
        logger.info("Deduplication: removed %d rows with duplicate cfb_player_id.", dedup_dropped)

    # Drop rows with low link confidence. Links below min_link_score are likely
    # wrong matches (different player with similar name) — their CFB features do
    # not represent the actual NFL player, poisoning the training signal.
    low_score_dropped = 0
    for pos in _POSITIONS:
        before = len(rows_by_pos[pos])
        rows_by_pos[pos] = [
            r for r in rows_by_pos[pos]
            if (r.get("match_score") or 0) >= args.min_link_score
        ]
        dropped = before - len(rows_by_pos[pos])
        if dropped:
            logger.warning(
                "  [link-score filter %s] dropped %d rows with match_score < %.0f",
                pos, dropped, args.min_link_score,
            )
            low_score_dropped += dropped
    if low_score_dropped:
        logger.info(
            "Link-score filter: removed %d rows with match_score < %.0f.",
            low_score_dropped, args.min_link_score,
        )

    # Write CSVs
    for pos in _POSITIONS:
        rows = rows_by_pos[pos]
        if not rows:
            logger.info("  %s: 0 rows — skipping.", pos)
            continue

        df = pd.DataFrame(rows).sort_values(["draft_year", "b2s_score"], ascending=[True, False])

        # Coverage report
        n_total = len(df)
        n_with_b2s = df["b2s_score"].notna().sum()
        n_with_combine = df["speed_score"].notna().sum()
        n_with_bigboard = df["consensus_rank"].notna().sum()
        n_with_recruiting = df["recruit_rating"].notna().sum()

        logger.info(
            "  %s: %d rows | b2s=%d (%.0f%%) | combine=%d (%.0f%%) | "
            "bigboard=%d (%.0f%%) | recruiting=%d (%.0f%%)",
            pos, n_total,
            n_with_b2s, 100 * n_with_b2s / n_total,
            n_with_combine, 100 * n_with_combine / n_total,
            n_with_bigboard, 100 * n_with_bigboard / n_total,
            n_with_recruiting, 100 * n_with_recruiting / n_total,
        )

        out_path = data_dir / f"training_{pos}.csv"
        df.to_csv(out_path, index=False)
        logger.info("  %s: wrote %s", pos, out_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
