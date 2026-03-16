"""
Apply fitted dynasty prospect models to an upcoming draft class.

Produces a ranked prospect sheet with:
  - Projected B2S score (PPR PPG, natural model units)
  - ORBIT Score (0-100 scale, bounded at 100 as theoretical ceiling)
    ORBIT = percentile of prospect's projected B2S vs training set predictions.
    Interpretation: ORBIT 70 → scores better than 70% of all 2011-2022 training players.
    100 is the theoretical ceiling; real prospects with elite profiles score in the 70-90s.

Pre-draft capital projection:
  When --post-draft is NOT set, draft capital is projected from big board consensus rank:
    - consensus_rank used directly as the projected overall pick number
    - Converted to draft_capital_score via the same exponential decay formula as the DB
    - All capital interaction terms (capital_x_age etc.) computed from projected capital
    - Players not on the board default to pick ~220 (late-round/UDFA floor)
  This dramatically improves pre-draft accuracy vs. imputing unknown capital to median.
  Re-run with --post-draft after the draft to use actual pick data.

Requires:
  - scripts/fit_model.py --all must have been run first (produces models/ artifacts)
  - cfb-prospect-db players must be marked via:
      python cfb-prospect-db/scripts/mark_declarations.py --draft-year 2026
  - nfl-fantasy-db NFLBigBoard must be populated:
      python nfl-fantasy-db/scripts/populate_bigboard.py --year 2026

Usage:
    python scripts/score_class.py --year 2026
    python scripts/score_class.py --year 2026 --position WR
    python scripts/score_class.py --year 2026 --model lgbm
    python scripts/score_class.py --year 2026 --output output/scores_2026.csv
    python scripts/score_class.py --year 2026 --post-draft   # use actual draft pick data
    python scripts/score_class.py --year 2026 --phase1       # show Phase I no-capital delta
"""

import argparse
import json
import logging
import math
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # for analyze.py imports

from analyze import engineer_features

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_POSITIONS = ("WR", "RB", "TE")
_BREAKOUT_THRESHOLD = {"WR": 0.20, "TE": 0.20, "RB": 0.10}
_POWER4_CONFS = frozenset({"SEC", "Big Ten", "ACC", "Big 12"})

# Floor pick for unranked prospects (no big board entry).
# Late 6th / early 7th round / UDFA fringe → draft_capital ≈ 0.6
_UNRANKED_FLOOR_PICK = 220


def _pick_to_draft_capital(overall_pick: int) -> float:
    """
    Same formula as cfb-prospect-db/ffdb/collectors/pfr_collector.py.
    draft_capital_score = 100 * exp(-0.023 * (pick - 1))
    Pick 1 → 100.0 | Pick 32 → 49.0 | Pick 100 → 10.3 | Pick 220 → 0.6
    """
    return round(100.0 * math.exp(-0.023 * (max(1, overall_pick) - 1)), 1)


def _projected_round(overall_pick: int) -> int:
    """Map projected pick to draft round (32 picks per round)."""
    return min(7, (overall_pick - 1) // 32 + 1)


# ---------------------------------------------------------------------------
# Data loading - mirrors build_training_set.py but for declared prospects
# ---------------------------------------------------------------------------

def _load_cfb_prospects(cfb_db_path: str, draft_year: int) -> dict:
    """
    Load all skill-position players declared for draft_year from cfb-prospect-db.
    Returns dict with same structure as build_training_set._load_cfb().
    """
    cfb_root = Path(cfb_db_path).parent
    if str(cfb_root) not in sys.path:
        sys.path.insert(0, str(cfb_root))

    from sqlalchemy import func as sa_func

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

    result: dict = {}

    with get_session(cfb_db_path) as s:
        # All players declared for this draft year at skill positions
        declared_players = (
            s.query(Player)
            .filter(
                Player.declared_draft_year == draft_year,
                Player.position.in_(["WR", "RB", "TE"]),
            )
            .all()
        )
        declared_ids = {p.id for p in declared_players}

        result["players"] = {
            p.id: {
                "full_name": p.full_name,
                "position":  p.position,
                "weight_lbs": p.weight_lbs,
                "height_inches": p.height_inches,
            }
            for p in declared_players
        }

        if not declared_ids:
            log.warning(
                "No players with declared_draft_year=%d found in cfb-prospect-db.\n"
                "Run: python cfb-prospect-db/scripts/mark_declarations.py --draft-year %d",
                draft_year, draft_year,
            )
            result["seasons"] = {}
            result["team_seasons"] = {}
            result["draft_picks"] = {}
            result["combine"] = {}
            result["recruiting"] = {}
            return result

        # College seasons
        seasons: dict[int, list[dict]] = defaultdict(list)
        for row in (
            s.query(CFBPlayerSeason)
            .filter(CFBPlayerSeason.player_id.in_(list(declared_ids)))
            .all()
        ):
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
        # Team seasons (denominators)
        team_seasons_raw = {
            (row.team, row.season_year): {
                "sp_plus_rating": row.sp_plus_rating,
                "pass_attempts": row.pass_attempts,
                "rush_attempts": row.rush_attempts,
            }
            for row in s.query(CFBTeamSeason).all()
        }
        result["team_seasons"] = team_seasons_raw

        # Team total receiving TDs per (team, year) - from ALL players, not just declared.
        # Required to compute each declared prospect's TD share in their best season.
        team_rec_tds_rows = (
            s.query(
                CFBPlayerSeason.team,
                CFBPlayerSeason.season_year,
                sa_func.sum(CFBPlayerSeason.rec_tds).label("total_rec_tds"),
            )
            .group_by(CFBPlayerSeason.team, CFBPlayerSeason.season_year)
            .all()
        )
        team_rec_tds = {
            (row.team, row.season_year): (row.total_rec_tds or 0)
            for row in team_rec_tds_rows
        }

        # Enrich player season dicts with team context so helpers
        # (_breakout_score, _total_yards_rate) can access SP+ without extra lookup args.
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

        # Draft picks - for post-draft scoring only; None pre-draft
        result["draft_picks"] = {
            row.player_id: {
                "draft_year": row.draft_year,
                "draft_round": row.draft_round,
                "overall_pick": row.overall_pick,
                "draft_capital_score": row.draft_capital_score,
            }
            for row in (
                s.query(NFLDraftPick)
                .filter(NFLDraftPick.player_id.in_(list(declared_ids)))
                .all()
            )
        }

        # Combine - may be available if combine has already run
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
            for row in (
                s.query(NFLCombineResult)
                .filter(NFLCombineResult.player_id.in_(list(declared_ids)))
                .all()
            )
        }

        # Recruiting
        recruiting: dict[int, dict] = {}
        for row in (
            s.query(Recruiting)
            .filter(Recruiting.player_id.in_(list(declared_ids)))
            .order_by(Recruiting.rating.desc())
            .all()
        ):
            if row.player_id not in recruiting:
                recruiting[row.player_id] = {
                    "recruit_rating": row.rating,
                    "recruit_stars": row.stars,
                    "recruit_year": row.recruit_year,
                    "recruit_rank_national": row.ranking_national,
                }
        result["recruiting"] = recruiting

        # PFF data - all seasons keyed by (player_id, season_year).
        # Merged into season dicts below as source-of-truth for counting stats.
        pff_by_season: dict[tuple, dict] = {}
        for row in (
            s.query(PFFPlayerSeason)
            .filter(PFFPlayerSeason.player_id.in_(list(declared_ids)))
            .all()
        ):
            # Parse PFF split JSON columns into summary scalars.
            # All JSON values are stored as strings - cast to float before arithmetic.
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
                "targets":      row.targets,
                "receptions":   row.receptions,
                "rec_yards":    row.rec_yards,
                "rec_tds":      row.rec_tds,
                "games_played": row.games_played,
                "rush_yards":   row.rush_yards,
                "rush_attempts": row.rush_attempts,
                "rush_tds":     row.rush_tds,
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

    # Merge PFF into season dicts (outside session - data already extracted)
    pff_merged = 0
    for pid, slist in seasons.items():
        for sn in slist:
            pff_sn = pff_by_season.get((pid, sn.get("season_year")))
            if pff_sn is None:
                continue
            for field in ("targets", "receptions", "rec_yards", "rec_tds", "games_played",
                          "rush_yards", "rush_attempts", "rush_tds"):
                pff_val = pff_sn.get(field)
                if pff_val is not None:
                    sn[field] = pff_val
            for field in ("yprr", "routes_run", "routes_per_game", "receiving_grade",
                          "catch_rate", "td_rate", "yards_per_reception", "drop_rate",
                          "contested_catch_rate", "target_separation", "offense_grade",
                          "rush_grade",
                          # PFF split scalars (depth / concept / scheme)
                          "deep_yprr", "deep_target_rate", "behind_los_rate",
                          "slot_yprr", "slot_target_rate", "screen_rate",
                          "man_yprr", "zone_yprr", "man_zone_delta"):
                sn[field] = pff_sn.get(field)
            pff_rec_yards = pff_sn.get("rec_yards")
            team_pass_att = sn.get("team_pass_att")
            if pff_rec_yards is not None and team_pass_att and team_pass_att > 0:
                sn["rec_yards_per_team_pass_att"] = round(pff_rec_yards / team_pass_att, 6)
            pff_merged += 1

    result["seasons"] = dict(seasons)
    log.info("PFF merged into %d/%d prospect season rows",
             pff_merged, sum(len(v) for v in seasons.values()))

    log.info(
        "Loaded %d declared %d prospects (%s) with %d season groups, "
        "%d combine rows, %d draft picks",
        len(result["players"]),
        draft_year,
        "/".join(_POSITIONS),
        len(result["seasons"]),
        len(result["combine"]),
        len(result["draft_picks"]),
    )
    return result


def _load_bigboard(nfl_db_path: str, draft_year: int) -> dict[str, dict]:
    """
    Load big board ranks for the target draft year from nfl-fantasy-db.
    Returns dict keyed by player_name (lower) → {consensus_rank, position_rank}.
    """
    nfl_root = Path(nfl_db_path).parent
    if str(nfl_root) not in sys.path:
        sys.path.insert(0, str(nfl_root))

    try:
        from ffnfl.database import NFLBigBoard, get_session as nfl_get_session

        board: dict[str, dict] = {}
        with nfl_get_session(nfl_db_path) as s:
            for row in (
                s.query(NFLBigBoard)
                .filter(NFLBigBoard.draft_year == draft_year)
                .all()
            ):
                key = _normalize_name(row.player_name)
                board[key] = {
                    "consensus_rank": row.consensus_rank,
                    "position_rank": row.position_rank,
                }
        log.info("Loaded %d big board entries for %d", len(board), draft_year)
        return board

    except Exception as e:
        log.warning("Could not load big board: %s - consensus_rank will be None", e)
        return {}


def _compute_teammate_score(
    player_id: int,
    best_team: str | None,
    cfb: dict,
    draft_year: int,
    year_window: int = 2,
) -> float:
    """Same logic as build_training_set._compute_teammate_score but uses training draft picks."""
    if not best_team:
        return 0.0
    score = 0.0
    for pid, pick in cfb["draft_picks"].items():
        if pid == player_id:
            continue
        teammate_draft_year = pick.get("draft_year") or 0
        if abs(teammate_draft_year - draft_year) > year_window:
            continue
        for season in cfb["seasons"].get(pid, []):
            if season.get("team") == best_team:
                score += (pick.get("draft_capital_score") or 0.0)
                break
    return round(score, 2)


_SUFFIX_RE = re.compile(r"\s+(jr\.?|sr\.?|ii|iii|iv|v)\s*$", re.IGNORECASE)


def _normalize_name(name: str) -> str:
    """Lowercase + strip generational suffixes for board-to-DB name matching.

    Handles mismatches like 'Chris Brazzell' (board) vs 'Chris Brazzell II' (DB),
    'Emmanuel Henderson' vs 'Emmanuel Henderson Jr.', etc.
    """
    return _SUFFIX_RE.sub("", name.lower().strip())


def _best_season(seasons: list[dict], min_games: int = 6) -> dict | None:
    qualifying = [s for s in seasons if (s.get("games_played") or 0) >= min_games]
    if not qualifying:
        return None
    return max(qualifying, key=lambda s: s.get("rec_yards_per_team_pass_att") or 0.0)


def _resolve_age(s: dict, recruit_year: int | None) -> float | None:
    """Return age_at_season_start, falling back to recruit_year estimate when DOB is absent.

    CFBD often lacks DOB for recent prospects. Estimate: recruit class of year Y implies
    the player was ~18.5 at the start of their freshman season (season_year == Y).
    age_at_season_start ≈ 18.5 + (season_year − recruit_year)
    """
    age = s.get("age_at_season_start")
    if age is not None:
        return age
    if recruit_year is not None:
        season_year = s.get("season_year")
        if season_year is not None:
            return 18.5 + (season_year - recruit_year)
    return None


def _breakout_score(seasons: list[dict], min_games: int = 6,
                    recruit_year: int | None = None) -> float | None:
    """
    JJ Zachariason's Breakout Score - primary WR production input (Ep 1083, Feb 2026).

    Formula (transcript-confirmed):
      base  = rec_yards_per_team_pass_att
      × SOS mult  = max(0.70, min(1.30, 1 + (sp_plus − 5) / 100))
      × age mult  = max(0, 26 − age_at_season_start)

    Taken as MAXIMUM across qualifying seasons (>= min_games).
    Season dicts must be pre-enriched with sp_plus_rating and team_pass_att.
    min_team_pass_att guard skips option-offense seasons (Navy ~100 vs FBS avg ~380).
    When age_at_season_start is absent, falls back to recruit_year estimate.
    """
    best = None
    for s in seasons:
        if (s.get("games_played") or 0) < min_games:
            continue
        team_pass_att = s.get("team_pass_att") or 0
        if team_pass_att < 200:
            continue
        rec_rate = s.get("rec_yards_per_team_pass_att") or 0.0
        age = _resolve_age(s, recruit_year)
        if age is None:
            continue
        sp_plus = s.get("sp_plus_rating")
        sos_mult = (
            max(0.70, min(1.30, 1.0 + (sp_plus - 5.0) / 100.0))
            if sp_plus is not None else 1.0
        )
        score = rec_rate * sos_mult * max(0.0, 26.0 - age)
        if best is None or score > best:
            best = score
    return round(best, 4) if best is not None else None


def _total_yards_rate(seasons: list[dict], min_games: int = 6,
                      recruit_year: int | None = None) -> float | None:
    """
    JJ's 'adjusted yards per team play' for RBs (Ep 1081, Feb 2026).

    Formula:
      base  = (rec_yards + rush_yards) / team_pass_attempts
      × SOS mult  = max(0.70, min(1.30, 1 + (sp_plus − 5) / 100))
      × age mult  = max(0, 26 − age_at_season_start)

    Taken as MAXIMUM across qualifying seasons (>= min_games).
    When age_at_season_start is absent, falls back to recruit_year estimate.
    """
    best = None
    for s in seasons:
        if (s.get("games_played") or 0) < min_games:
            continue
        team_pass_att = s.get("team_pass_att") or 0
        if team_pass_att < 200:
            continue
        total_yards = (s.get("rec_yards") or 0) + (s.get("rush_yards") or 0)
        age = _resolve_age(s, recruit_year)
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


def _total_yards_rate_v2(seasons: list[dict], min_games: int = 6,
                         recruit_year: int | None = None) -> float | None:
    """
    Phase D variant: denominator = team_pass_att + team_rush_att (total team plays).
    Otherwise identical to _total_yards_rate(). Falls back to pass_att-only if
    team_rush_att is unavailable.
    """
    best = None
    for s in seasons:
        if (s.get("games_played") or 0) < min_games:
            continue
        team_pass_att = s.get("team_pass_att") or 0
        if team_pass_att < 200:
            continue
        team_rush_att = s.get("team_rush_att") or 0
        total_plays = team_pass_att + team_rush_att if team_rush_att else team_pass_att
        total_yards = (s.get("rec_yards") or 0) + (s.get("rush_yards") or 0)
        age = _resolve_age(s, recruit_year)
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


def _breakout_age(seasons: list[dict], position: str, min_games: int = 6,
                  recruit_year: int | None = None) -> float | None:
    threshold = _BREAKOUT_THRESHOLD.get(position, 0.20)
    qualifying = sorted(
        [s for s in seasons if (s.get("games_played") or 0) >= min_games],
        key=lambda s: s.get("season_year") or 9999,
    )
    for s in qualifying:
        dom = s.get("dominator_rating") or 0.0
        if dom >= threshold:
            return _resolve_age(s, recruit_year)
    return None


def _build_prospect_row(
    player_id: int,
    cfb: dict,
    board: dict[str, dict],
    draft_year: int,
    post_draft: bool,
    cfb_training: dict | None,
    min_games: int = 6,
) -> dict | None:
    """Build one feature row for a prospect. Returns None if no qualifying college season."""
    player = cfb["players"].get(player_id)
    if player is None:
        return None

    position = player["position"]
    seasons_raw = cfb["seasons"].get(player_id, [])

    best = _best_season(seasons_raw, min_games=min_games)
    if best is None:
        # Try with lower bar (min_games=3) for players with limited season data
        best = _best_season(seasons_raw, min_games=3)
    if best is None:
        return None

    best_team = best.get("team")
    best_team_season = cfb["team_seasons"].get((best_team, best.get("season_year")), {})

    combine = cfb["combine"].get(player_id, {})
    recruiting = cfb["recruiting"].get(player_id, {})
    recruit_year: int | None = recruiting.get("recruit_year")

    # Big board lookup - suffix-normalized to handle Jr./II/III mismatches
    name_key = _normalize_name(player["full_name"])
    board_data = board.get(name_key, {})

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

    ba = _breakout_age(seasons_raw, position, min_games=min_games, recruit_year=recruit_year)
    best_breakout_score = _breakout_score(seasons_raw, min_games=min_games, recruit_year=recruit_year)
    best_total_yards_rate = _total_yards_rate(seasons_raw, min_games=min_games, recruit_year=recruit_year)
    best_total_yards_rate_v2 = _total_yards_rate_v2(seasons_raw, min_games=min_games, recruit_year=recruit_year)

    best_targets_val = best.get("targets") or 0
    best_receptions_val = best.get("receptions") or 0
    best_rec_yards_val = best.get("rec_yards") or 0
    best_rec_tds_val = best.get("rec_tds") or 0
    best_rush_yards_val = best.get("rush_yards") or 0
    best_rush_attempts_val = best.get("rush_attempts") or 0
    best_rush_tds_val = best.get("rush_tds") or 0
    best_games_val = best.get("games_played") or 0

    college_fantasy_ppg = (
        (
            best_receptions_val * 1.0 + best_rec_yards_val / 10.0
            + best_rec_tds_val * 6.0 + best_rush_yards_val / 10.0
            + best_rush_tds_val * 6.0
        ) / best_games_val
        if best_games_val > 0 else None
    )

    best_ypr = best_rec_yards_val / best_receptions_val if best_receptions_val > 0 else None
    best_ypt = best_rec_yards_val / best_targets_val if best_targets_val > 0 else None
    best_rush_ypc = best_rush_yards_val / best_rush_attempts_val if best_rush_attempts_val > 0 else None
    best_yards_per_touch = (
        (best_rec_yards_val + best_rush_yards_val) / (best_targets_val + best_rush_attempts_val)
        if (best_targets_val + best_rush_attempts_val) > 0 else None
    )

    weight = combine.get("combine_weight_lbs") or player.get("weight_lbs")

    # Draft capital
    draft_pick = cfb["draft_picks"].get(player_id, {})
    if post_draft and draft_pick:
        # Post-draft: use actual pick data
        draft_capital_score = draft_pick.get("draft_capital_score")
        draft_round         = draft_pick.get("draft_round")
        overall_pick        = draft_pick.get("overall_pick")
        is_projected        = False
    else:
        # Pre-draft: project from big board consensus rank.
        # consensus_rank on nflmockdraftdatabase is the overall board position,
        # which closely tracks actual draft slot for top prospects.
        board_rank = board_data.get("consensus_rank")
        proj_pick  = int(board_rank) if board_rank is not None else _UNRANKED_FLOOR_PICK
        draft_capital_score = _pick_to_draft_capital(proj_pick)
        draft_round         = _projected_round(proj_pick)
        overall_pick        = proj_pick
        is_projected        = True

    # Teammate score (uses training draft picks if available; empty pre-draft)
    team_picks = cfb_training["draft_picks"] if cfb_training else cfb["draft_picks"]
    teammate_score = _compute_teammate_score(
        player_id, best_team,
        {**cfb, "draft_picks": team_picks},
        draft_year,
    )

    return {
        # Identity
        "player_name": player["full_name"],
        "position": position,
        "best_team": best_team,
        "best_season_year": best.get("season_year"),
        "declared_draft_year": draft_year,
        # College
        "best_rec_rate": best.get("rec_yards_per_team_pass_att"),
        "best_dominator": best.get("dominator_rating"),
        "best_reception_share": best.get("reception_share"),
        "best_age": _resolve_age(best, recruit_year),
        "best_games": best_games_val,
        "best_sp_plus": best_team_season.get("sp_plus_rating"),
        "best_ppa_pass": best.get("ppa_avg_pass"),
        "best_ppa_overall": best.get("ppa_avg_overall"),
        "best_ppa_rush": best.get("ppa_avg_rush"),
        "best_usage": best.get("usage_overall"),
        "best_usage_pass": best.get("usage_pass"),
        "best_usage_rush": best.get("usage_rush"),
        "best_rec_yards": best_rec_yards_val or None,
        "best_receptions": best_receptions_val or None,
        "best_targets": best_targets_val or None,
        "best_rec_tds": best_rec_tds_val,
        "best_rush_tds": best_rush_tds_val,
        "best_ypr": best_ypr,
        "best_ypt": best_ypt,
        "best_rush_ypc": best_rush_ypc,
        "best_yards_per_touch": best_yards_per_touch,
        "college_fantasy_ppg": college_fantasy_ppg,
        "breakout_age": ba,
        "best_breakout_score": best_breakout_score,
        "best_total_yards_rate": best_total_yards_rate,
        "best_total_yards_rate_v2": best_total_yards_rate_v2,
        # Career
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
        "early_declare": int(career_seasons <= 3),
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
        # Draft capital (actual post-draft; projected from board rank pre-draft)
        "draft_capital_score": draft_capital_score,
        "draft_round": draft_round,
        "overall_pick": overall_pick,
        "capital_is_projected": is_projected,
        # Recruiting
        "recruit_rating": recruiting.get("recruit_rating"),
        "recruit_stars": recruiting.get("recruit_stars"),
        "recruit_rank_national": recruiting.get("recruit_rank_national"),
        # Pre-draft market
        "consensus_rank": board_data.get("consensus_rank"),
        "position_rank": board_data.get("position_rank"),
        # Context
        "teammate_score": teammate_score,
        # Conference tier (Power-4 = 1, all others = 0)
        "power4_conf": int(best.get("conference") in _POWER4_CONFS) if best.get("conference") else None,
        # TD share in best season (JJ's sub-20% ceiling flag for WRs)
        "best_rec_td_pct": best.get("rec_td_pct"),
        # PFF metrics - read from best season dict (PFF merged as source of truth)
        "best_yprr": best.get("yprr"),
        "best_routes_per_game": best.get("routes_per_game"),
        "best_receiving_grade": best.get("receiving_grade"),
        "best_contested_catch_rate": best.get("contested_catch_rate"),
        "best_drop_rate": best.get("drop_rate"),
        "best_target_sep": best.get("target_separation"),
        # PFF split scalars - depth / concept / scheme zones
        "best_deep_yprr": best.get("deep_yprr"),
        "best_deep_target_rate": best.get("deep_target_rate"),
        "best_behind_los_rate": best.get("behind_los_rate"),
        "best_slot_yprr": best.get("slot_yprr"),
        "best_slot_target_rate": best.get("slot_target_rate"),
        "best_screen_rate": best.get("screen_rate"),
        "best_man_yprr": best.get("man_yprr"),
        "best_zone_yprr": best.get("zone_yprr"),
        "best_man_zone_delta": best.get("man_zone_delta"),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_position(
    position: str,
    prospects_df: pd.DataFrame,
    models_dir: Path,
    model_type: str = "ridge",
    post_draft: bool = False,
    model_suffix: str = "",   # "" for capital model; "_nocap" for Phase I
) -> pd.DataFrame:
    """
    Load fitted pipeline and score prospects for one position.

    ORBIT Score (0-100):
      = percentileofscore(training_ridge_predictions, prospect_ridge_prediction)
      The reference distribution is the Ridge model's in-sample predictions on
      training players - stored in models/metadata.json under {pos}_train_preds.
      This ensures 100 is a true ceiling: a prospect must match or exceed the
      best training player's model projection to reach 100.
    """
    model_path = models_dir / f"{position}_{model_type}{model_suffix}.pkl"
    features_path = models_dir / f"{position}_features{model_suffix}.json"
    metadata_path = models_dir / f"metadata{model_suffix}.json"

    if not model_path.exists():
        log.warning("Model not found: %s - run fit_model.py --all first", model_path)
        return pd.DataFrame()

    pipeline  = joblib.load(model_path)
    features  = json.loads(features_path.read_text())
    metadata  = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    train_preds_ref = np.array(metadata.get(position, {}).get("train_preds", []))

    df = prospects_df[prospects_df["position"] == position].copy()
    if df.empty:
        log.info("  %s: no declared prospects found.", position)
        return pd.DataFrame()

    # Apply feature engineering (same as analyze.py + fit_model.py)
    df["draft_year"] = df["declared_draft_year"]

    # Cast all non-identity columns to numeric (prospect rows may have Python None
    # which creates object-dtype Series, causing np.log / clip to fail)
    _id_cols = {"player_name", "position", "best_team", "model", "post_draft",
                "declared_draft_year", "draft_year", "best_season_year",
                "capital_is_projected"}
    for col in df.columns:
        if col not in _id_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df_eng = engineer_features(df)

    # Ensure all model features are present (fill missing with NaN → imputed)
    for f in features:
        if f not in df_eng.columns:
            df_eng[f] = np.nan

    X     = df_eng[features].values
    preds = np.clip(pipeline.predict(X), 0, None)

    # ORBIT Score - percentile vs Ridge training prediction distribution.
    # Bounded [0, 100]: any prospect at or above the training max → ORBIT = 100.
    if len(train_preds_ref) > 0:
        orbit_scores = np.array([
            min(100.0, float(percentileofscore(train_preds_ref, p, kind="weak")))
            for p in preds
        ])
    else:
        log.warning("  No training predictions in metadata - ORBIT scores unavailable.")
        orbit_scores = np.full(len(preds), np.nan)

    # D1: Conformal prediction intervals from LOYO residual distribution
    loyo_residuals = np.array(metadata.get(position, {}).get("loyo_abs_residuals", []))
    if len(loyo_residuals) > 0:
        q80 = float(np.quantile(loyo_residuals, 0.80))
        q90 = float(np.quantile(loyo_residuals, 0.90))
    else:
        q80 = q90 = float("nan")

    keep_cols = [c for c in [
        "player_name", "position", "best_team", "best_season_year",
        # Production
        "best_dominator", "best_rec_rate", "college_fantasy_ppg",
        "breakout_age", "best_age", "best_breakout_score",
        "best_total_yards_rate", "best_usage_pass", "best_usage_rush",
        "best_rush_ypc", "early_declare", "power4_conf",
        # PFF efficiency
        "best_yprr", "best_receiving_grade", "best_routes_per_game",
        "best_drop_rate", "best_target_sep", "best_contested_catch_rate",
        "best_deep_yprr", "best_deep_target_rate", "best_behind_los_rate",
        "best_slot_yprr", "best_slot_target_rate", "best_screen_rate",
        "best_man_yprr", "best_zone_yprr", "best_man_zone_delta",
        # Athleticism
        "weight_lbs", "height_inches", "speed_score",
        "forty_time", "broad_jump", "vertical_jump",
        "three_cone", "shuttle", "bench_press",
        # Recruiting
        "recruit_rating", "recruit_stars", "recruit_rank_national",
        # Draft / board
        "consensus_rank", "position_rank",
        "draft_capital_score", "overall_pick", "capital_is_projected",
    ] if c in df.columns]

    result = df[keep_cols].copy()
    result["projected_b2s"] = preds.round(2)
    result["orbit_score"]   = orbit_scores.round(1)
    if not np.isnan(q80):
        result["b2s_lo80"] = (preds - q80).clip(min=0).round(2)
        result["b2s_hi80"] = (preds + q80).round(2)
        result["b2s_lo90"] = (preds - q90).clip(min=0).round(2)
        result["b2s_hi90"] = (preds + q90).round(2)
    result["model"]         = model_type
    result["post_draft"]    = post_draft
    result = result.sort_values("orbit_score", ascending=False).reset_index(drop=True)
    result.insert(0, "pos_rank", result.index + 1)

    log.info(
        "  %s: %d prospects scored | top 3: %s",
        position,
        len(result),
        ", ".join(
            f"{r['player_name']} (ORBIT={r['orbit_score']:.0f}, B2S={r['projected_b2s']:.1f})"
            for _, r in result.head(3).iterrows()
        ),
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a draft class with fitted dynasty prospect models."
    )
    parser.add_argument("--year", type=int, required=True, help="Draft year to score.")
    parser.add_argument("--position", choices=["WR", "RB", "TE"],
                        help="Score a single position (default: all).")
    parser.add_argument("--model", choices=["ridge", "lgbm"], default="ridge",
                        help="Which model to use for scoring (default: ridge).")
    parser.add_argument("--post-draft", action="store_true",
                        help="Incorporate actual draft pick data (run after the draft).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: output/scores_{year}.csv).")
    parser.add_argument("--min-games", type=int, default=6,
                        help="Minimum games for a college season to qualify (default: 6).")
    parser.add_argument(
        "--phase1", action="store_true",
        help="Also score with Phase I (no-capital) model and show capital delta. "
             "Requires fit_model.py --all --no-capital to have been run first.",
    )
    args = parser.parse_args()

    from config import get_cfb_db_path, get_data_dir, get_nfl_db_path

    cfb_db_path = get_cfb_db_path()
    nfl_db_path = get_nfl_db_path()
    data_dir    = get_data_dir()
    models_dir  = _ROOT / "models"
    output_dir  = _ROOT / "output" / "scores"
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = args.output or str(output_dir / f"scores_{args.year}_{args.model}.csv")
    positions = _POSITIONS if args.position is None else (args.position,)

    # Load all prospect data at once
    cfb = _load_cfb_prospects(cfb_db_path, args.year)
    board = _load_bigboard(nfl_db_path, args.year)

    # Load training draft picks separately for teammate_score (historical context)
    # This uses all historical draft picks, not just declared prospects
    try:
        cfb_root = Path(cfb_db_path).parent
        if str(cfb_root) not in sys.path:
            sys.path.insert(0, str(cfb_root))
        from ffdb.database import NFLDraftPick, get_session as cfb_session

        with cfb_session(cfb_db_path) as s:
            training_picks = {
                row.player_id: {
                    "draft_year": row.draft_year,
                    "draft_capital_score": row.draft_capital_score,
                }
                for row in s.query(NFLDraftPick).all()
            }
        cfb_training = {"draft_picks": training_picks, "seasons": cfb["seasons"]}
    except Exception:
        cfb_training = None

    # Build feature rows for all declared prospects
    prospect_rows = []
    for player_id in cfb["players"]:
        row = _build_prospect_row(
            player_id=player_id,
            cfb=cfb,
            board=board,
            draft_year=args.year,
            post_draft=args.post_draft,
            cfb_training=cfb_training,
            min_games=args.min_games,
        )
        if row is not None:
            prospect_rows.append(row)

    if not prospect_rows:
        log.error("No prospect rows built - check declarations and season data.")
        return

    prospects_df = pd.DataFrame(prospect_rows)
    log.info("Built %d prospect rows across %s positions", len(prospects_df),
             "/".join(str(p) for p in prospects_df["position"].unique()))

    # Score each position
    all_scored = []
    for pos in positions:
        # Step 0 / 7d: check TE LightGBM status from metadata.
        # If lgbm_status != "active", fall back to Ridge with a warning.
        effective_model = args.model
        if pos == "TE" and args.model == "lgbm":
            try:
                _meta = json.loads((models_dir / "metadata.json").read_text())
                _lgbm_status = _meta.get("TE", {}).get("lgbm_status", "disabled")
            except Exception:
                _lgbm_status = "disabled"
            if _lgbm_status != "active":
                print(
                    f"WARNING: TE LightGBM is not active (status='{_lgbm_status}', "
                    f"see overfitting remediation plan Step 0/7d). Scoring TE with Ridge."
                )
                effective_model = "ridge"
            else:
                _te_loyo = _meta.get("TE", {}).get("lgbm_loyo_r2", float("nan"))
                print(
                    f"NOTE: TE LightGBM active after Step 7d retune "
                    f"(LOYO R2={_te_loyo:.3f}). Using lgbm as requested."
                )

        scored = score_position(
            position=pos,
            prospects_df=prospects_df,
            models_dir=models_dir,
            model_type=effective_model,
            post_draft=args.post_draft,
        )
        if not scored.empty:
            all_scored.append(scored)

    if not all_scored:
        log.error("No prospects scored - check model files and declared player list.")
        return

    combined = pd.concat(all_scored, ignore_index=True)

    # --- Phase I (no-capital) scoring ---
    if args.phase1:
        _nocap_meta_path = models_dir / "metadata_nocap.json"
        if not _nocap_meta_path.exists():
            print(
                "WARNING: Phase I models not found (metadata_nocap.json missing). "
                "Run: python scripts/fit_model.py --all --no-capital"
            )
        else:
            ph1_scored = []
            for pos in positions:
                ph1 = score_position(
                    position=pos,
                    prospects_df=prospects_df,
                    models_dir=models_dir,
                    model_type="ridge",
                    post_draft=args.post_draft,
                    model_suffix="_nocap",
                )
                if not ph1.empty:
                    ph1_scored.append(ph1[["player_name", "position", "orbit_score"]]
                                      .rename(columns={"orbit_score": "phase1_orbit"}))

            if ph1_scored:
                ph1_df = pd.concat(ph1_scored, ignore_index=True)
                combined = combined.merge(
                    ph1_df, on=["player_name", "position"], how="left"
                )
                combined["phase1_orbit"] = combined["phase1_orbit"].round(1)
                combined["capital_delta"] = (
                    combined["orbit_score"] - combined["phase1_orbit"]
                ).round(1)

                def _risk(delta):
                    if pd.isna(delta):
                        return "N/A"
                    if delta <= -15:
                        return "Low Risk"
                    if delta >= 20:
                        return "High Risk"
                    return "Neutral"

                combined["risk"] = combined["capital_delta"].apply(_risk)

    combined.to_csv(out_path, index=False)
    log.info("Scores written to: %s", out_path)

    # Print summary table
    show_phase1 = args.phase1 and "phase1_orbit" in combined.columns
    print(f"\n{'='*95 if show_phase1 else 85}")
    print(f"  Dynasty Prospect Model - {args.year} Draft Class Scores ({args.model.upper()})")
    if not args.post_draft:
        print("  NOTE: Pre-draft - draft capital projected from big board consensus rank")
    if show_phase1:
        print("  Phase I columns: Ph1=no-capital ORBIT | Delta=ORBIT-Ph1 | Risk profile")
    print(f"{'='*95 if show_phase1 else 85}")
    for pos in positions:
        sub = combined[combined["position"] == pos]
        if sub.empty:
            continue
        print(f"\n  {pos} ({len(sub)} prospects)")
        # Check if interval columns exist for this position subset
        _has_interval = "b2s_lo80" in sub.columns and sub["b2s_lo80"].notna().any()
        _ivl_hdr = f"{'80% Interval':>14}" if _has_interval else ""
        _ivl_w   = 15 if _has_interval else 0
        if show_phase1:
            print(f"  {'Rank':<4} {'Player':<28} {'Proj B2S':>8} {'ORBIT':>6} "
                  f"{'Ph1':>5} {'Delta':>6} {'Risk':<10} "
                  f"{_ivl_hdr}{'Dom':>6} {'RecRate':>7} {'Board':>5}")
            print("  " + "-" * (92 + _ivl_w))
        else:
            print(f"  {'Rank':<4} {'Player':<28} {'Proj B2S':>8} {'ORBIT':>6} "
                  f"{_ivl_hdr}{'Dom':>6} {'RecRate':>7} {'CfpPPG':>6} {'Board':>5}")
            print("  " + "-" * (80 + _ivl_w))
        for _, r in sub.head(20).iterrows():
            if _has_interval and pd.notna(r.get("b2s_lo80")):
                _ivl_str = f"[{r['b2s_lo80']:.1f}–{r['b2s_hi80']:.1f}]"
                _ivl_col = f"{_ivl_str:>14} "
            else:
                _ivl_col = f"{'':>{_ivl_w}}" if _ivl_w else ""
            if show_phase1:
                ph1_val   = r.get("phase1_orbit")
                delta_val = r.get("capital_delta")
                risk_val  = r.get("risk", "N/A")
                ph1_str   = f"{ph1_val:>5.1f}" if pd.notna(ph1_val) else f"{'--':>5}"
                delta_str = (f"{delta_val:>+6.1f}" if pd.notna(delta_val) else f"{'--':>6}")
                print(
                    f"  {r['pos_rank']:<4} {r['player_name']:<28} "
                    f"{r['projected_b2s']:>8.2f} {r['orbit_score']:>6.1f} "
                    f"{ph1_str} {delta_str} {risk_val:<10} "
                    f"{_ivl_col}"
                    f"{(r.get('best_dominator') or 0):>6.3f} "
                    f"{(r.get('best_rec_rate') or 0):>7.4f} "
                    f"{int(r['consensus_rank']) if pd.notna(r.get('consensus_rank')) else '-':>5}"
                )
            else:
                print(
                    f"  {r['pos_rank']:<4} {r['player_name']:<28} "
                    f"{r['projected_b2s']:>8.2f} {r['orbit_score']:>6.1f} "
                    f"{_ivl_col}"
                    f"{(r.get('best_dominator') or 0):>6.3f} "
                    f"{(r.get('best_rec_rate') or 0):>7.4f} "
                    f"{(r.get('college_fantasy_ppg') or 0):>6.1f} "
                    f"{int(r['consensus_rank']) if pd.notna(r.get('consensus_rank')) else '-':>5}"
                )
    # Step 6: Per-year R2 transparency table (model stability calibration)
    metadata_path = _ROOT / "models" / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        print(f"\n  Model stability (LOYO R2 by holdout year):")
        for pos in positions:
            pos_meta = metadata.get(pos, {})
            year_results = pos_meta.get("ridge_loyo_years", [])
            if not year_results:
                continue
            loyo_r2 = pos_meta.get("ridge_loyo_r2", float("nan"))
            r2_vals = [y["r2"] for y in year_results if not (y["r2"] != y["r2"])]
            r2_min  = min(r2_vals) if r2_vals else float("nan")
            r2_max  = max(r2_vals) if r2_vals else float("nan")
            r2_std  = float(np.std(r2_vals)) if r2_vals else float("nan")
            ridge_alpha = pos_meta.get("ridge_alpha", None)
            alpha_str   = f"  alpha={ridge_alpha:.1f}" if ridge_alpha is not None else ""
            lgbm_status = pos_meta.get("lgbm_status", "")
            lgbm_note   = f"  LGBM={lgbm_status}" if lgbm_status else ""
            print(
                f"  {pos} Ridge LOYO R2={loyo_r2:.3f} "
                f"(range: {r2_min:.2f}-{r2_max:.2f}, std={r2_std:.3f})"
                f"{alpha_str}{lgbm_note}"
            )
            row_parts = []
            for y in year_results:
                flag = " *" if y["r2"] < 0.20 else ""
                row_parts.append(f"    {y['year']}: n={y['n_test']:>2}  R2={y['r2']:>6.3f}{flag}")
            print("\n".join(row_parts))
            low_conf = [y["year"] for y in year_results if y["r2"] < 0.20]
            if low_conf:
                print(f"    * Low-confidence years (R2<0.20): {low_conf} - treat rankings as approximate.")
            # D3: rank-order calibration
            rho   = pos_meta.get("loyo_spearman_rho")
            top25 = pos_meta.get("loyo_top25_hit_rate")
            if rho is not None and rho == rho:
                top25_str = f"  Top-25% hit rate={top25:.1%}" if (top25 is not None and top25 == top25) else ""
                print(f"    Rank-order calibration: Spearman rho={rho:.3f}{top25_str} (vs 25% base rate)")

        # Also print rolling/kfold for TE
        te_meta = metadata.get("TE", {})
        rc = te_meta.get("rolling_cv", {})
        kc = te_meta.get("kfold_cv", {})
        if rc.get("r2") is not None and rc["r2"] == rc["r2"]:
            print(
                f"  TE Rolling-window R2={rc['r2']:.3f} "
                f"(train<=2018 n={rc.get('n_train',0)}, test>=2019 n={rc.get('n_test',0)})"
            )
            print(f"  TE K-fold (k=5) R2={kc.get('r2', float('nan')):.3f}")

    print(f"\n{'='*85}\n")


if __name__ == "__main__":
    main()
