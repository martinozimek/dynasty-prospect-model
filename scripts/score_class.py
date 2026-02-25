"""
Apply fitted dynasty prospect models to an upcoming draft class.

Produces a ranked prospect sheet with projected B2S scores and percentile ranks
relative to the historical training distribution.

Pre-draft limitations:
  - draft_capital_score, overall_pick, draft_round: unknown → imputed to training median
  - All capital interaction terms (capital_x_age etc.) also set to NaN → imputed
  - Model accuracy is lower pre-draft; projections improve once actual draft capital is known
  - Re-run after the draft with --post-draft to incorporate actual pick data

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
"""

import argparse
import json
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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


# ---------------------------------------------------------------------------
# Data loading — mirrors build_training_set.py but for declared prospects
# ---------------------------------------------------------------------------

def _load_cfb_prospects(cfb_db_path: str, draft_year: int) -> dict:
    """
    Load all skill-position players declared for draft_year from cfb-prospect-db.
    Returns dict with same structure as build_training_set._load_cfb().
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
            result["pff"] = {}
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
        result["seasons"] = dict(seasons)

        # Team seasons (denominators)
        result["team_seasons"] = {
            (row.team, row.season_year): {
                "sp_plus_rating": row.sp_plus_rating,
                "pass_attempts": row.pass_attempts,
            }
            for row in s.query(CFBTeamSeason).all()
        }

        # Draft picks — for post-draft scoring only; None pre-draft
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

        # Combine — may be available if combine has already run
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

        # PFF data
        pff: dict[int, dict] = {}
        for row in (
            s.query(PFFPlayerSeason)
            .filter(PFFPlayerSeason.player_id.in_(list(declared_ids)))
            .order_by(PFFPlayerSeason.yprr.desc().nullslast())
            .all()
        ):
            if row.player_id not in pff:
                pff[row.player_id] = {
                    "best_yprr": row.yprr,
                    "best_routes_per_game": row.routes_per_game,
                    "best_receiving_grade": row.receiving_grade,
                    "best_contested_catch_rate": row.contested_catch_rate,
                    "best_drop_rate": row.drop_rate,
                    "best_target_sep": row.target_separation,
                }
        result["pff"] = pff

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
                key = row.player_name.lower().strip()
                board[key] = {
                    "consensus_rank": row.consensus_rank,
                    "position_rank": row.position_rank,
                }
        log.info("Loaded %d big board entries for %d", len(board), draft_year)
        return board

    except Exception as e:
        log.warning("Could not load big board: %s — consensus_rank will be None", e)
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


def _best_season(seasons: list[dict], min_games: int = 6) -> dict | None:
    qualifying = [s for s in seasons if (s.get("games_played") or 0) >= min_games]
    if not qualifying:
        return None
    return max(qualifying, key=lambda s: s.get("rec_yards_per_team_pass_att") or 0.0)


def _breakout_age(seasons: list[dict], position: str, min_games: int = 6) -> float | None:
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
    pff = cfb["pff"].get(player_id, {})

    # Big board lookup by player name (case-insensitive)
    name_key = player["full_name"].lower().strip()
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

    ba = _breakout_age(seasons_raw, position, min_games=min_games)

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

    # Draft capital — available post-draft or not yet (pre-draft → None)
    draft_pick = cfb["draft_picks"].get(player_id, {})
    if post_draft and draft_pick:
        draft_capital_score = draft_pick.get("draft_capital_score")
        draft_round = draft_pick.get("draft_round")
        overall_pick = draft_pick.get("overall_pick")
    else:
        draft_capital_score = None
        draft_round = None
        overall_pick = None

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
        "best_age": best.get("age_at_season_start"),
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
        # Draft (None pre-draft)
        "draft_capital_score": draft_capital_score,
        "draft_round": draft_round,
        "overall_pick": overall_pick,
        # Recruiting
        "recruit_rating": recruiting.get("recruit_rating"),
        "recruit_stars": recruiting.get("recruit_stars"),
        "recruit_rank_national": recruiting.get("recruit_rank_national"),
        # Pre-draft market
        "consensus_rank": board_data.get("consensus_rank"),
        "position_rank": board_data.get("position_rank"),
        # Context
        "teammate_score": teammate_score,
        # PFF stubs
        "best_yprr": pff.get("best_yprr"),
        "best_routes_per_game": pff.get("best_routes_per_game"),
        "best_receiving_grade": pff.get("best_receiving_grade"),
        "best_contested_catch_rate": pff.get("best_contested_catch_rate"),
        "best_drop_rate": pff.get("best_drop_rate"),
        "best_target_sep": pff.get("best_target_sep"),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _load_training_b2s(data_dir: Path, position: str) -> pd.Series:
    """Load B2S scores from training CSV to compute percentile ranks."""
    path = data_dir / f"training_{position}.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path)
    return df["b2s_score"].dropna()


def score_position(
    position: str,
    prospects_df: pd.DataFrame,
    models_dir: Path,
    data_dir: Path,
    model_type: str = "ridge",
    post_draft: bool = False,
) -> pd.DataFrame:
    """Load fitted pipeline and score prospects for one position."""
    model_path = models_dir / f"{position}_{model_type}.pkl"
    features_path = models_dir / f"{position}_features.json"

    if not model_path.exists():
        log.warning("Model not found: %s — run fit_model.py --all first", model_path)
        return pd.DataFrame()

    pipeline = joblib.load(model_path)
    features = json.loads(features_path.read_text())

    df = prospects_df[prospects_df["position"] == position].copy()
    if df.empty:
        log.info("  %s: no declared prospects found.", position)
        return pd.DataFrame()

    # Apply feature engineering (same as analyze.py + fit_model.py)
    # Need draft_year column for engineer_features (combined_ath uses overall_pick)
    df["draft_year"] = df["declared_draft_year"]

    # Cast all non-identity columns to numeric (prospect rows may have Python None
    # which creates object-dtype Series, causing np.log / clip to fail)
    _id_cols = {"player_name", "position", "best_team", "model", "post_draft",
                "declared_draft_year", "draft_year", "best_season_year"}
    for col in df.columns:
        if col not in _id_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df_eng = engineer_features(df)

    # Predict — use only features the model knows
    available_features = [f for f in features if f in df_eng.columns]
    missing_features = [f for f in features if f not in df_eng.columns]
    if missing_features:
        log.debug("  %s: features missing from prospect data (will be imputed): %s",
                  position, missing_features)
        for f in missing_features:
            df_eng[f] = np.nan

    X = df_eng[features].values
    preds = np.clip(pipeline.predict(X), 0, None)

    # Percentile rank vs training distribution
    train_b2s = _load_training_b2s(data_dir, position)
    if len(train_b2s) > 0:
        percentiles = np.array([
            float(np.mean(train_b2s <= p)) * 100 for p in preds
        ])
    else:
        percentiles = np.full(len(preds), np.nan)

    result = df[["player_name", "position", "best_team", "best_season_year",
                 "best_dominator", "best_rec_rate", "college_fantasy_ppg",
                 "breakout_age", "best_age", "weight_lbs", "speed_score",
                 "consensus_rank", "position_rank",
                 "draft_capital_score", "overall_pick"]].copy()
    result["projected_b2s"] = preds.round(2)
    result["percentile"] = percentiles.round(1)
    result["model"] = model_type
    result["post_draft"] = post_draft
    result = result.sort_values("projected_b2s", ascending=False).reset_index(drop=True)
    result.insert(0, "pos_rank", result.index + 1)

    log.info(
        "  %s: %d prospects scored | top 3: %s",
        position,
        len(result),
        ", ".join(
            f"{r['player_name']} ({r['projected_b2s']:.1f})"
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
        log.error("No prospect rows built — check declarations and season data.")
        return

    prospects_df = pd.DataFrame(prospect_rows)
    log.info("Built %d prospect rows across %s positions", len(prospects_df),
             "/".join(str(p) for p in prospects_df["position"].unique()))

    # Score each position
    all_scored = []
    for pos in positions:
        scored = score_position(
            position=pos,
            prospects_df=prospects_df,
            models_dir=models_dir,
            data_dir=data_dir,
            model_type=args.model,
            post_draft=args.post_draft,
        )
        if not scored.empty:
            all_scored.append(scored)

    if not all_scored:
        log.error("No prospects scored — check model files and declared player list.")
        return

    combined = pd.concat(all_scored, ignore_index=True)
    combined.to_csv(out_path, index=False)
    log.info("Scores written to: %s", out_path)

    # Print summary table
    print(f"\n{'='*85}")
    print(f"  Dynasty Prospect Model — {args.year} Draft Class Scores ({args.model.upper()})")
    if not args.post_draft:
        print("  NOTE: Pre-draft — draft capital features unknown, scores are estimates")
    print(f"{'='*85}")
    for pos in positions:
        sub = combined[combined["position"] == pos]
        if sub.empty:
            continue
        print(f"\n  {pos} ({len(sub)} prospects)")
        print(f"  {'Rank':<4} {'Player':<28} {'Proj B2S':>8} {'Pct%':>5} "
              f"{'Dom':>6} {'RecRate':>7} {'CfpPPG':>6} {'Board':>5}")
        print("  " + "-" * 80)
        for _, r in sub.head(20).iterrows():
            print(
                f"  {r['pos_rank']:<4} {r['player_name']:<28} "
                f"{r['projected_b2s']:>8.2f} {r['percentile']:>5.1f} "
                f"{(r.get('best_dominator') or 0):>6.3f} "
                f"{(r.get('best_rec_rate') or 0):>7.4f} "
                f"{(r.get('college_fantasy_ppg') or 0):>6.1f} "
                f"{int(r['consensus_rank']) if pd.notna(r.get('consensus_rank')) else '—':>5}"
            )
    print(f"\n{'='*85}\n")


if __name__ == "__main__":
    main()
