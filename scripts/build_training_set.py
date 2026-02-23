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
                "games_played": row.games_played,
                "rec_yards": row.rec_yards,
                "targets": row.targets,
                "receptions": row.receptions,
                "rec_tds": row.rec_tds,
                "rush_yards": row.rush_yards,
                "rush_attempts": row.rush_attempts,
                "rec_yards_per_team_pass_att": row.rec_yards_per_team_pass_att,
                "dominator_rating": row.dominator_rating,
                "reception_share": row.reception_share,
                "age_at_season_start": row.age_at_season_start,
                "ppa_avg_pass": row.ppa_avg_pass,
                "usage_overall": row.usage_overall,
            })
        result["seasons"] = dict(seasons)

        # Team seasons — keyed by (team, year) for SP+ lookup
        result["team_seasons"] = {
            (row.team, row.season_year): {
                "sp_plus_rating": row.sp_plus_rating,
                "pass_attempts": row.pass_attempts,
            }
            for row in s.query(CFBTeamSeason).all()
        }

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
                }
        result["recruiting"] = recruiting

    logger.info(
        "CFB loaded: %d players, %d season groups, %d draft picks, %d combine rows",
        len(result["players"]),
        len(result["seasons"]),
        len(result["draft_picks"]),
        len(result["combine"]),
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
                "top_season_ppg": row.top_season_ppg,
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


def _compute_teammate_score(
    player_id: int,
    draft_year: int,
    best_team: str | None,
    cfb: dict,
    year_window: int = 2,
) -> float:
    """
    Teammate score: sum of draft_capital_score for other WR/RB/TE draftees
    from the same school within ±year_window years of this player's draft year.

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
        if abs(teammate_draft_year - draft_year) > year_window:
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
    career_rec_per_target = (
        career_rec_yards / career_targets if career_targets > 0 else None
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
        "top_season_ppg": b2s_data.get("top_season_ppg"),
        "year1_ppg": b2s_data.get("year1_ppg"),
        "year2_ppg": b2s_data.get("year2_ppg"),
        "year3_ppg": b2s_data.get("year3_ppg"),
        "qualifying_nfl_seasons": b2s_data.get("qualifying_seasons"),
        # Best college season
        "best_rec_rate": best.get("rec_yards_per_team_pass_att"),
        "best_dominator": best.get("dominator_rating"),
        "best_reception_share": best.get("reception_share"),
        "best_age": best.get("age_at_season_start"),
        "best_games": best.get("games_played"),
        "best_season_year": best.get("season_year"),
        "best_team": best_team,
        "best_sp_plus": best_team_season.get("sp_plus_rating"),
        "best_ppa_pass": best.get("ppa_avg_pass"),
        "best_usage": best.get("usage_overall"),
        # Career totals
        "career_seasons": career_seasons,
        "career_rec_yards": career_rec_yards,
        "career_rush_yards": career_rush_yards,
        "career_targets": career_targets,
        "career_rec_per_target": career_rec_per_target,
        "early_declare": int(career_seasons <= 3),
        # Combine
        "weight_lbs": weight,
        "forty_time": combine.get("forty_time"),
        "speed_score": combine.get("speed_score"),
        "height_inches": combine.get("height_inches_combine") or player.get("height_inches"),
        "vertical_jump": combine.get("vertical_jump"),
        "broad_jump": combine.get("broad_jump"),
        # Draft
        "draft_capital_score": draft_pick.get("draft_capital_score"),
        "draft_round": draft_pick.get("draft_round"),
        "overall_pick": draft_pick.get("overall_pick"),
        # Recruiting
        "recruit_rating": recruiting.get("recruit_rating"),
        "recruit_stars": recruiting.get("recruit_stars"),
        "recruit_year": recruiting.get("recruit_year"),
        # Pre-draft market expectation
        "consensus_rank": board.get("consensus_rank"),
        "position_rank": board.get("position_rank"),
        # Team context
        "teammate_score": teammate_score,
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
        "--min-year", type=int, default=2011,
        help="Minimum draft year to include (default: 2011).",
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
