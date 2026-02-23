"""
Data integrity audit for the dynasty-prospect-model training pipeline.

Mandate: verify and validate all data before any model work. A model trained
on bad data produces bad predictions regardless of algorithm quality.

Sections:
  1. cfb-prospect-db   -  raw college stats, derived metrics, draft/combine ranges
  2. nfl-fantasy-db    -  B2S labels, season scores, CFB link quality
  3. Cross-DB          -  link correctness (position match, draft year, duplicates)
  4. Training set      -  feature coverage, target distribution, class balance

Output:
  PASS / WARN / FAIL summary printed to stdout.
  FAIL = definitive data integrity problem  -  must fix before model work.
  WARN = suspicious value  -  review manually, may be legitimate.
  Flagged rows written to data/audit/*.csv for manual inspection.

Usage:
    python scripts/audit_data.py
    python scripts/audit_data.py --section cfb      # one section only
    python scripts/audit_data.py --section nfl
    python scripts/audit_data.py --section crossdb
    python scripts/audit_data.py --section training
    python scripts/audit_data.py --no-csv           # print only, no file output
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_cfb_db_path, get_data_dir, get_nfl_db_path

logging.basicConfig(
    level=logging.WARNING,    # suppress ORM noise
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_POSITIONS = ("WR", "RB", "TE")

# B2S plausibility ceilings by position (PPG)
_B2S_MAX = {"WR": 40.0, "RB": 40.0, "TE": 35.0}

# Combine weight plausibility bounds (lbs)
_WEIGHT_MIN, _WEIGHT_MAX = 150, 380
# 40-yard dash bounds (seconds)
_FORTY_MIN, _FORTY_MAX = 4.0, 6.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    status: str                          # PASS | WARN | FAIL
    message: str
    flagged: pd.DataFrame | None = None  # rows to save as CSV
    csv_name: str | None = None          # filename stem for the flagged CSV


_RESULTS: list[CheckResult] = []

ICON = {"PASS": "OK", "WARN": "!!", "FAIL": "XX"}


def _record(result: CheckResult, audit_dir: Path | None, write_csv: bool) -> None:
    _RESULTS.append(result)
    icon = ICON.get(result.status, "?")
    print(f"  [{result.status}] {icon}  {result.name}: {result.message}")
    if (
        result.flagged is not None
        and not result.flagged.empty
        and write_csv
        and audit_dir
        and result.csv_name
    ):
        path = audit_dir / f"{result.csv_name}.csv"
        result.flagged.to_csv(path, index=False)
        print(f"          -> {path.name} ({len(result.flagged)} rows)")


# ---------------------------------------------------------------------------
# DB loading helpers (same pattern as build_training_set.py)
# ---------------------------------------------------------------------------

def _load_cfb_db(cfb_db_path: str) -> dict:
    cfb_root = Path(cfb_db_path).parent
    if str(cfb_root) not in sys.path:
        sys.path.insert(0, str(cfb_root))
    from ffdb.database import (
        CFBPlayerSeason, CFBTeamSeason, NFLCombineResult,
        NFLDraftPick, Player, get_session,
    )
    data = {}
    with get_session(cfb_db_path) as s:
        data["players"]      = pd.read_sql(s.query(Player).statement, s.bind)
        data["seasons"]      = pd.read_sql(s.query(CFBPlayerSeason).statement, s.bind)
        data["team_seasons"] = pd.read_sql(s.query(CFBTeamSeason).statement, s.bind)
        data["draft_picks"]  = pd.read_sql(s.query(NFLDraftPick).statement, s.bind)
        data["combine"]      = pd.read_sql(s.query(NFLCombineResult).statement, s.bind)
    return data


def _load_nfl_db(nfl_db_path: str) -> dict:
    nfl_root = Path(nfl_db_path).parent
    if str(nfl_root) not in sys.path:
        sys.path.insert(0, str(nfl_root))
    from ffnfl.database import (
        NFLB2S, CFBLink, NFLPlayerSeasonScore, NFLPlayerGameLog, get_session,
    )
    data = {}
    with get_session(nfl_db_path) as s:
        data["b2s"]           = pd.read_sql(s.query(NFLB2S).statement, s.bind)
        data["links"]         = pd.read_sql(s.query(CFBLink).statement, s.bind)
        data["season_scores"] = pd.read_sql(s.query(NFLPlayerSeasonScore).statement, s.bind)
        data["game_logs"]     = pd.read_sql(s.query(NFLPlayerGameLog).statement, s.bind)
    return data


# ---------------------------------------------------------------------------
# Section 1: cfb-prospect-db integrity
# ---------------------------------------------------------------------------

def audit_cfb(cfb: dict, audit_dir: Path | None, write_csv: bool) -> None:
    print("\n=== Section 1: cfb-prospect-db ===")
    seasons     = cfb["seasons"]
    players     = cfb["players"]
    team_seasons = cfb["team_seasons"]
    draft_picks = cfb["draft_picks"]
    combine     = cfb["combine"]

    n_seasons = len(seasons)
    n_players = len(players)

    # --- 1.1 games_played null rate ---
    null_gp = seasons["games_played"].isna().sum()
    pct = 100 * null_gp / n_seasons
    status = "PASS" if null_gp == 0 else ("WARN" if pct < 5 else "FAIL")
    _record(CheckResult(
        name="games_played null rate",
        status=status,
        message=f"{null_gp:,} / {n_seasons:,} null ({pct:.1f}%)",
        flagged=seasons[seasons["games_played"].isna()][
            ["player_id", "season_year", "team", "games_played"]
        ] if null_gp else None,
        csv_name="cfb_games_played_nulls",
    ), audit_dir, write_csv)

    # --- 1.2 games_played range plausibility ---
    # College regular season: 1-14 games; bowl game adds 1; max realistic ~15
    gp_high = seasons[seasons["games_played"].fillna(0) > 15]
    gp_low  = seasons[(seasons["games_played"].fillna(99) < 1) & seasons["games_played"].notna()]
    flag_gp = pd.concat([gp_high, gp_low])[["player_id", "season_year", "team", "games_played"]]
    status = "PASS" if flag_gp.empty else "WARN"
    _record(CheckResult(
        name="games_played range (1-15)",
        status=status,
        message=f"{len(flag_gp)} rows out of range (>15 or <1)",
        flagged=flag_gp if not flag_gp.empty else None,
        csv_name="cfb_games_played_range",
    ), audit_dir, write_csv)

    # --- 1.3 Duplicate (player_id, season_year, team) ---
    dups = seasons[seasons.duplicated(subset=["player_id", "season_year", "team"], keep=False)]
    status = "PASS" if dups.empty else "FAIL"
    _record(CheckResult(
        name="Duplicate (player_id, season_year, team)",
        status=status,
        message=f"{len(dups)} duplicate rows found",
        flagged=dups[["player_id", "season_year", "team"]].drop_duplicates() if not dups.empty else None,
        csv_name="cfb_duplicate_seasons",
    ), audit_dir, write_csv)

    # --- 1.4 Season stat plausibility ---
    # Coerce to numeric first  -  CFBD returns object dtype with None values
    rec_yards_n  = pd.to_numeric(seasons["rec_yards"],   errors="coerce")
    targets_n    = pd.to_numeric(seasons["targets"],     errors="coerce")
    receptions_n = pd.to_numeric(seasons["receptions"],  errors="coerce")

    # rec_yards > 2,000 in a season is historically possible (record ~1,900) but worth flagging
    hi_rec = seasons[rec_yards_n.fillna(0) > 2000][
        ["player_id", "season_year", "team", "games_played", "rec_yards", "targets"]
    ]
    # targets > 200 in a season (record is ~180) = suspect
    hi_tgt = seasons[targets_n.fillna(0) > 200][
        ["player_id", "season_year", "team", "games_played", "targets", "rec_yards"]
    ]
    # Receptions > targets (impossible)
    bad_rec = seasons[
        targets_n.notna() & receptions_n.notna() &
        (receptions_n > targets_n)
    ][["player_id", "season_year", "team", "targets", "receptions"]]

    all_flags = pd.concat([
        hi_rec.assign(flag="rec_yards>2000"),
        hi_tgt.assign(flag="targets>200"),
        bad_rec.assign(flag="receptions>targets"),
    ], ignore_index=True)
    status = "FAIL" if not bad_rec.empty else ("WARN" if not all_flags.empty else "PASS")
    _record(CheckResult(
        name="Season stat plausibility",
        status=status,
        message=(
            f"rec_yards>2000: {len(hi_rec)} | targets>200: {len(hi_tgt)} | "
            f"receptions>targets: {len(bad_rec)}"
        ),
        flagged=all_flags if not all_flags.empty else None,
        csv_name="cfb_stat_plausibility",
    ), audit_dir, write_csv)

    # --- 1.5 rec_yards_per_team_pass_att (rec_rate) ---
    # NOTE: rec_yards_per_team_pass_att is a RATE (yards per team pass attempt), NOT a
    # proportion. Values > 1.0 are normal for elite WRs (e.g. 1200 rec_yards / 500 att = 2.4).
    # Historical range: 0.0 to ~4.0 for WRs. Flag only extreme outliers (> 6.0).
    rec_rate_col = "rec_yards_per_team_pass_att"
    null_rr = seasons[rec_rate_col].isna().sum()
    pct_null = 100 * null_rr / n_seasons
    # > 6.0 would require ~3000 rec yards on a team with 500 pass attempts -- extreme outlier
    extreme_rr = seasons[seasons[rec_rate_col].fillna(0) > 6.0][
        ["player_id", "season_year", "team", rec_rate_col, "rec_yards"]
    ]
    # Nulls are expected for non-WR/RB/TE positions in the DB; flag if > 50% null
    status = "FAIL" if not extreme_rr.empty else (
        "WARN" if pct_null > 50 else "PASS"
    )
    _record(CheckResult(
        name="rec_rate (yd/att) null rate & extreme outliers",
        status=status,
        message=(
            f"null: {null_rr:,} ({pct_null:.1f}%) -- expected for non-skill positions | "
            f">6.0 (extreme): {len(extreme_rr)} | "
            f"note: values 1.0-4.0 are normal for elite WRs (rate, not proportion)"
        ),
        flagged=extreme_rr if not extreme_rr.empty else None,
        csv_name="cfb_rec_rate_extreme",
    ), audit_dir, write_csv)

    # --- 1.6 dominator_rating range ---
    # dominator_rating = player_rec_yards / team_total_rec_yards -- a TRUE proportion (0-1).
    # Values > 1.0 or < 0 indicate a calculation error (usually team denominator too small
    # or floating-point rounding on rows where rec_yards is null/zero).
    # Nulls are expected for non-skill position players.
    dom_col = "dominator_rating"
    null_dom = seasons[dom_col].isna().sum()
    # Only check rows where rec_yards is not null (skill-position players)
    skill_dom = seasons[seasons["rec_yards"].notna()]
    dom_negative = skill_dom[
        skill_dom[dom_col].notna() & (skill_dom[dom_col] < 0)
    ][["player_id", "season_year", "team", dom_col, "rec_yards"]]
    dom_over1 = skill_dom[
        skill_dom[dom_col].notna() & (skill_dom[dom_col] > 1.0)
    ][["player_id", "season_year", "team", dom_col, "rec_yards"]]
    dom_hi = skill_dom[
        skill_dom[dom_col].fillna(0) > 0.65
    ][["player_id", "season_year", "team", dom_col, "rec_yards"]]
    status = "FAIL" if (not dom_negative.empty or not dom_over1.empty) else (
        "WARN" if not dom_hi.empty else "PASS"
    )
    _record(CheckResult(
        name="dominator_rating range (skill players only)",
        status=status,
        message=(
            f"null: {null_dom:,} (expected for non-skill positions) | "
            f"<0 (error): {len(dom_negative)} | >1.0 (impossible): {len(dom_over1)} | "
            f">0.65 (elite, review): {len(dom_hi)}"
        ),
        flagged=pd.concat([
            dom_negative.assign(flag="<0_error"),
            dom_over1.assign(flag=">1.0_impossible"),
            dom_hi.assign(flag=">0.65_review"),
        ], ignore_index=True) if (not dom_negative.empty or not dom_over1.empty or not dom_hi.empty) else None,
        csv_name="cfb_dominator_flags",
    ), audit_dir, write_csv)

    # --- 1.7 Team pass attempts sanity ---
    ts = team_seasons
    low_pa  = ts[ts["pass_attempts"].fillna(999) < 100][["team", "season_year", "pass_attempts"]]
    hi_pa   = ts[ts["pass_attempts"].fillna(0) > 900][["team", "season_year", "pass_attempts"]]
    null_pa = ts["pass_attempts"].isna().sum()
    flag_pa = pd.concat([low_pa.assign(flag="<100"), hi_pa.assign(flag=">900")], ignore_index=True)
    status  = "PASS" if flag_pa.empty else "WARN"
    _record(CheckResult(
        name="Team pass_attempts range",
        status=status,
        message=(
            f"null: {null_pa} | <100 (suspect low): {len(low_pa)} | "
            f">900 (suspect high): {len(hi_pa)}"
        ),
        flagged=flag_pa if not flag_pa.empty else None,
        csv_name="cfb_team_pass_att_flags",
    ), audit_dir, write_csv)

    # --- 1.8 Draft picks ---
    dp = draft_picks
    bad_round  = dp[dp["draft_round"].notna() & ~dp["draft_round"].between(1, 7)]
    bad_pick   = dp[dp["overall_pick"].notna() & ~dp["overall_pick"].between(1, 262)]
    null_cap   = dp["draft_capital_score"].isna().sum()
    null_year  = dp["draft_year"].isna().sum()
    flag_dp    = pd.concat([
        bad_round.assign(flag="round_out_of_range"),
        bad_pick.assign(flag="pick_out_of_range"),
    ], ignore_index=True)[["player_id", "draft_year", "draft_round", "overall_pick", "flag"]] \
        if (not bad_round.empty or not bad_pick.empty) else pd.DataFrame()
    status = "FAIL" if not flag_dp.empty else (
        "WARN" if (null_cap > 0 or null_year > 0) else "PASS"
    )
    _record(CheckResult(
        name="Draft pick round/pick range",
        status=status,
        message=(
            f"bad round: {len(bad_round)} | bad pick: {len(bad_pick)} | "
            f"null capital_score: {null_cap} | null draft_year: {null_year}"
        ),
        flagged=flag_dp if not flag_dp.empty else None,
        csv_name="cfb_draft_flags",
    ), audit_dir, write_csv)

    # --- 1.9 Combine measurables ---
    c = combine
    null_weight = c["weight_lbs"].isna().sum()
    null_forty  = c["forty_time"].isna().sum()
    null_speed  = c["speed_score"].isna().sum()
    n_combine   = len(c)
    # Bad weight range
    bad_weight = c[
        c["weight_lbs"].notna() & ~c["weight_lbs"].between(_WEIGHT_MIN, _WEIGHT_MAX)
    ][["player_id", "combine_year", "position", "weight_lbs"]]
    # Bad 40 range
    bad_forty = c[
        c["forty_time"].notna() & ~c["forty_time"].between(_FORTY_MIN, _FORTY_MAX)
    ][["player_id", "combine_year", "position", "forty_time"]]
    # speed_score null when both weight and forty are present (should be computed)
    speed_should_exist = c[c["weight_lbs"].notna() & c["forty_time"].notna() & c["speed_score"].isna()]
    flag_c = pd.concat([
        bad_weight.assign(flag="weight_out_of_range"),
        bad_forty.assign(flag="forty_out_of_range"),
        speed_should_exist[["player_id", "combine_year", "position"]].assign(flag="speed_score_missing"),
    ], ignore_index=True)
    status = "FAIL" if not speed_should_exist.empty else (
        "WARN" if not flag_c.empty else "PASS"
    )
    _record(CheckResult(
        name="Combine measurables range",
        status=status,
        message=(
            f"{n_combine} rows | null weight: {null_weight} | null forty: {null_forty} | "
            f"null speed_score: {null_speed} | "
            f"speed_score missing when computable: {len(speed_should_exist)}"
        ),
        flagged=flag_c if not flag_c.empty else None,
        csv_name="cfb_combine_flags",
    ), audit_dir, write_csv)


# ---------------------------------------------------------------------------
# Section 2: nfl-fantasy-db integrity
# ---------------------------------------------------------------------------

def audit_nfl(nfl: dict, audit_dir: Path | None, write_csv: bool) -> None:
    print("\n=== Section 2: nfl-fantasy-db ===")
    b2s           = nfl["b2s"]
    links         = nfl["links"]
    season_scores = nfl["season_scores"]
    game_logs     = nfl["game_logs"]

    # --- 2.1 B2S score range by position ---
    for pos in _POSITIONS:
        pos_b2s = b2s[b2s["position"] == pos]
        n = len(pos_b2s)
        if n == 0:
            continue
        null_b2s = pos_b2s["b2s_score"].isna().sum()
        neg_b2s  = pos_b2s[pos_b2s["b2s_score"].fillna(0) < 0]
        hi_b2s   = pos_b2s[pos_b2s["b2s_score"].fillna(0) > _B2S_MAX[pos]]
        mean_b2s = pos_b2s["b2s_score"].mean()
        status = "FAIL" if (not neg_b2s.empty or null_b2s > 0) else (
            "WARN" if not hi_b2s.empty else "PASS"
        )
        flagged = pd.concat([
            neg_b2s.assign(flag="negative"),
            hi_b2s.assign(flag=f">{_B2S_MAX[pos]}_review"),
        ], ignore_index=True)[["player_name", "position", "draft_year", "b2s_score", "flag"]] \
            if (not neg_b2s.empty or not hi_b2s.empty) else None
        _record(CheckResult(
            name=f"B2S range [{pos}]",
            status=status,
            message=(
                f"{n} rows | null: {null_b2s} | mean: {mean_b2s:.2f} ppg | "
                f"negative: {len(neg_b2s)} | >{_B2S_MAX[pos]}: {len(hi_b2s)}"
            ),
            flagged=flagged,
            csv_name=f"nfl_b2s_flags_{pos.lower()}",
        ), audit_dir, write_csv)

    # --- 2.2 B2S zero-score rate (qualifying_seasons = 0) ---
    zero_b2s = b2s[b2s["qualifying_seasons"].fillna(99) == 0]
    pct_zero = 100 * len(zero_b2s) / len(b2s) if len(b2s) > 0 else 0
    # ~10-20% zero rate is expected (busts, short careers); > 35% is suspicious
    status = "PASS" if pct_zero <= 35 else "WARN"
    _record(CheckResult(
        name="B2S zero-score rate (qualifying_seasons=0)",
        status=status,
        message=f"{len(zero_b2s)} / {len(b2s)} rows ({pct_zero:.1f}%) have 0 qualifying NFL seasons",
        flagged=zero_b2s[["player_name", "position", "draft_year", "b2s_score", "qualifying_seasons"]],
        csv_name="nfl_b2s_zero_score",
    ), audit_dir, write_csv)

    # --- 2.3 Season score PPG plausibility ---
    # Use PPR format as reference
    ppr_scores = season_scores[season_scores["format_name"] == "ppr"]
    hi_ppg = ppr_scores[ppr_scores["fantasy_ppg"].fillna(0) > 50][
        ["player_name", "position", "season_year", "fantasy_ppg", "games_played"]
    ]
    # Small negative PPG (down to ~-2.0) is legitimate: net rushing losses produce
    # negative points. Only flag truly impossible negatives (< -5.0).
    neg_ppg_extreme = ppr_scores[ppr_scores["fantasy_ppg"].fillna(0) < -5.0][
        ["player_name", "position", "season_year", "fantasy_ppg", "games_played"]
    ]
    neg_ppg_minor = ppr_scores[
        (ppr_scores["fantasy_ppg"].fillna(0) < 0) &
        (ppr_scores["fantasy_ppg"].fillna(0) >= -5.0)
    ][["player_name", "position", "season_year", "fantasy_ppg", "games_played"]]
    status = "FAIL" if not neg_ppg_extreme.empty else ("WARN" if not hi_ppg.empty else "PASS")
    _record(CheckResult(
        name="Season score PPG range (PPR)",
        status=status,
        message=(
            f"PPG > 50: {len(hi_ppg)} | PPG < -5.0 (FAIL): {len(neg_ppg_extreme)} | "
            f"PPG -5.0 to 0 (legitimate -- net rushing loss): {len(neg_ppg_minor)} | "
            f"max: {ppr_scores['fantasy_ppg'].max():.1f}"
        ),
        flagged=pd.concat([hi_ppg.assign(flag=">50_review"), neg_ppg_extreme.assign(flag="<-5_error")], ignore_index=True)
        if (not hi_ppg.empty or not neg_ppg_extreme.empty) else None,
        csv_name="nfl_season_score_flags",
    ), audit_dir, write_csv)

    # --- 2.4 Season score format completeness ---
    # Each player-season should have a row for every format
    ps_counts = season_scores.groupby(["nflverse_id", "season_year"])["format_id"].count()
    n_formats = season_scores["format_id"].nunique()
    incomplete = ps_counts[ps_counts < n_formats]
    status = "PASS" if incomplete.empty else "WARN"
    _record(CheckResult(
        name="Season score format completeness",
        status=status,
        message=(
            f"{n_formats} formats x {len(ps_counts)} player-seasons | "
            f"incomplete (< {n_formats} formats): {len(incomplete)}"
        ),
    ), audit_dir, write_csv)

    # --- 2.5 Game log week range ---
    bad_weeks = game_logs[
        ~game_logs["week"].between(1, 18)
    ][["player_name", "nflverse_id", "season_year", "week"]]
    status = "FAIL" if not bad_weeks.empty else "PASS"
    _record(CheckResult(
        name="Game log week range (1-18)",
        status=status,
        message=f"{len(bad_weeks)} rows with week out of range",
        flagged=bad_weeks if not bad_weeks.empty else None,
        csv_name="nfl_gamelog_week_flags",
    ), audit_dir, write_csv)

    # --- 2.6 CFB link match score distribution ---
    null_links  = links["match_score"].isna().sum()
    low_links   = links[links["match_score"].fillna(100) < 80][
        ["nfl_player_name", "position", "draft_year", "cfb_full_name", "match_score", "match_method"]
    ]
    med_links   = links[
        links["match_score"].fillna(100).between(80, 89.9)
    ][["nfl_player_name", "position", "draft_year", "cfb_full_name", "match_score", "match_method"]]
    pct_low = 100 * len(low_links) / len(links) if len(links) > 0 else 0
    status = "FAIL" if len(low_links) > 0 else ("WARN" if len(med_links) > 20 else "PASS")
    _record(CheckResult(
        name="CFB link match scores",
        status=status,
        message=(
            f"{len(links)} links | null: {null_links} | "
            f"<80 (questionable): {len(low_links)} ({pct_low:.1f}%) | "
            f"80-89 (review): {len(med_links)}"
        ),
        flagged=pd.concat([
            low_links.assign(flag="<80_questionable"),
            med_links.assign(flag="80-89_review"),
        ], ignore_index=True) if (not low_links.empty or not med_links.empty) else None,
        csv_name="nfl_link_score_flags",
    ), audit_dir, write_csv)

    # --- 2.7 Duplicate cfb_player_id in CFBLink ---
    dup_cfb_ids = links[
        links["cfb_player_id"].notna() &
        links.duplicated(subset=["cfb_player_id"], keep=False)
    ][["nfl_player_name", "position", "draft_year", "cfb_player_id", "cfb_full_name", "match_score"]]
    status = "FAIL" if not dup_cfb_ids.empty else "PASS"
    _record(CheckResult(
        name="Duplicate cfb_player_id in links",
        status=status,
        message=f"{len(dup_cfb_ids)} rows share a cfb_player_id (one CFB player -> multiple NFL names)",
        flagged=dup_cfb_ids if not dup_cfb_ids.empty else None,
        csv_name="nfl_link_dup_cfb_ids",
    ), audit_dir, write_csv)

    # --- 2.8 Fumble/INT count sanity ---
    # Fumbles lost > 10 in a season for WR/RB/TE is implausible
    season_ppr = season_scores[season_scores["format_name"] == "ppr"]
    hi_fl = season_ppr[season_ppr["fumbles_lost"].fillna(0) > 10][
        ["player_name", "position", "season_year", "games_played", "fumbles_lost"]
    ]
    neg_fl = season_ppr[season_ppr["fumbles_lost"].fillna(0) < 0][
        ["player_name", "position", "season_year", "fumbles_lost"]
    ]
    status = "FAIL" if not neg_fl.empty else ("WARN" if not hi_fl.empty else "PASS")
    _record(CheckResult(
        name="Fumbles lost range",
        status=status,
        message=f"negative: {len(neg_fl)} | >10/season: {len(hi_fl)}",
        flagged=pd.concat([neg_fl.assign(flag="negative"), hi_fl.assign(flag=">10_review")], ignore_index=True)
        if (not neg_fl.empty or not hi_fl.empty) else None,
        csv_name="nfl_fumbles_flags",
    ), audit_dir, write_csv)


# ---------------------------------------------------------------------------
# Section 3: Cross-DB join integrity
# ---------------------------------------------------------------------------

def audit_crossdb(cfb: dict, nfl: dict, audit_dir: Path | None, write_csv: bool) -> None:
    print("\n=== Section 3: Cross-DB join integrity ===")
    links   = nfl["links"]
    players = cfb["players"]
    seasons = cfb["seasons"]

    valid_cfb_ids = set(players["id"].dropna().astype(int))

    # --- 3.1 Every cfb_player_id in CFBLink exists in cfb-prospect-db ---
    linked = links[links["cfb_player_id"].notna()].copy()
    linked["cfb_player_id_int"] = linked["cfb_player_id"].astype(int)
    missing_ids = linked[~linked["cfb_player_id_int"].isin(valid_cfb_ids)][
        ["nfl_player_name", "position", "draft_year", "cfb_player_id", "cfb_full_name"]
    ]
    status = "FAIL" if not missing_ids.empty else "PASS"
    _record(CheckResult(
        name="CFBLink -> Player ID exists",
        status=status,
        message=f"{len(missing_ids)} linked cfb_player_id values not found in cfb-prospect-db",
        flagged=missing_ids if not missing_ids.empty else None,
        csv_name="crossdb_missing_cfb_ids",
    ), audit_dir, write_csv)

    # --- 3.2 Position match: CFBLink.position vs Player.position ---
    merged = linked.merge(
        players[["id", "position"]].rename(columns={"id": "cfb_player_id_int", "position": "cfb_position"}),
        on="cfb_player_id_int", how="left",
    )
    # Flag cross-position matches (WR->TE, WR->RB, etc.)  -  not same-group variants
    pos_mismatches = merged[
        merged["cfb_position"].notna() &
        (merged["position"] != merged["cfb_position"])
    ][["nfl_player_name", "position", "cfb_full_name", "cfb_position", "match_score", "draft_year"]]
    # Distinguish hard mismatches (WR->TE) from soft (RB->FB)
    soft_variants = {("RB", "FB"), ("FB", "RB"), ("WR", "WR"), ("TE", "TE"), ("RB", "RB")}
    hard_mismatch = pos_mismatches[
        ~pos_mismatches.apply(
            lambda r: (r["position"], r["cfb_position"]) in soft_variants, axis=1
        )
    ]
    status = "FAIL" if not hard_mismatch.empty else (
        "WARN" if not pos_mismatches.empty else "PASS"
    )
    _record(CheckResult(
        name="CFBLink position match",
        status=status,
        message=(
            f"total mismatches: {len(pos_mismatches)} | "
            f"hard (cross-group): {len(hard_mismatch)} | "
            f"soft (RB/FB variants): {len(pos_mismatches) - len(hard_mismatch)}"
        ),
        flagged=pos_mismatches.assign(
            mismatch_type=pos_mismatches.apply(
                lambda r: "soft" if (r["position"], r["cfb_position"]) in soft_variants else "HARD",
                axis=1,
            )
        ) if not pos_mismatches.empty else None,
        csv_name="crossdb_position_mismatches",
    ), audit_dir, write_csv)

    # --- 3.3 Draft year alignment ---
    # Each linked player's last CFB season should be draft_year - 1 or draft_year - 2.
    # Flag if last CFB season > draft_year - 1 (impossible) or < draft_year - 4 (long gap).
    player_last_season = (
        seasons[seasons["player_id"].isin(valid_cfb_ids)]
        .groupby("player_id")["season_year"]
        .max()
        .reset_index()
        .rename(columns={"season_year": "last_cfb_season"})
    )
    merged2 = linked[linked["draft_year"].notna()].merge(
        player_last_season.rename(columns={"player_id": "cfb_player_id_int"}),
        on="cfb_player_id_int", how="left",
    )
    merged2["gap"] = merged2["draft_year"] - merged2["last_cfb_season"]
    # gap should be 1 or 2 (sat out a year, or mid-season draft)
    bad_future = merged2[merged2["gap"].fillna(99) < 1]   # last CFB season AFTER draft year
    bad_old    = merged2[merged2["gap"].fillna(0) > 4]    # last CFB season >4 years before draft
    flag_align = pd.concat([
        bad_future[["nfl_player_name", "position", "draft_year", "last_cfb_season", "gap"]].assign(flag="cfb_after_draft"),
        bad_old[["nfl_player_name", "position", "draft_year", "last_cfb_season", "gap"]].assign(flag="gap>4_years"),
    ], ignore_index=True)
    # gap=3 (two-year sitters) is WARN; gap<1 or gap>4 is FAIL
    status = "FAIL" if not bad_future.empty else (
        "WARN" if not bad_old.empty else "PASS"
    )
    _record(CheckResult(
        name="Draft year alignment (last CFB season vs draft_year)",
        status=status,
        message=(
            f"gap<1 (CFB after draft, impossible): {len(bad_future)} | "
            f"gap>4 (stale link): {len(bad_old)}"
        ),
        flagged=flag_align if not flag_align.empty else None,
        csv_name="crossdb_draft_year_alignment",
    ), audit_dir, write_csv)

    # --- 3.4 Duplicate cfb_player_id (confirmed here vs 2.7 which was within links only) ---
    dup_in_merged = merged[
        merged["cfb_player_id_int"].notna() &
        merged.duplicated(subset=["cfb_player_id_int"], keep=False)
    ][["nfl_player_name", "position", "cfb_full_name", "cfb_position", "match_score", "draft_year"]]
    status = "FAIL" if not dup_in_merged.empty else "PASS"
    _record(CheckResult(
        name="Duplicate cfb_player_id across links (cross-check)",
        status=status,
        message=f"{len(dup_in_merged)} rows with shared cfb_player_id",
        flagged=dup_in_merged if not dup_in_merged.empty else None,
        csv_name="crossdb_dup_cfb_ids_confirmed",
    ), audit_dir, write_csv)

    # --- 3.5 B2S players with no CFB link ---
    b2s = nfl["b2s"]
    b2s_in_window = b2s[b2s["draft_year"].between(2011, 2022)]
    link_keys = set(zip(links["nfl_player_name"], links["position"]))
    unlinked = b2s_in_window[
        ~b2s_in_window.apply(lambda r: (r["player_name"], r["position"]) in link_keys, axis=1)
    ][["player_name", "position", "draft_year", "b2s_score"]]
    pct_unlinked = 100 * len(unlinked) / len(b2s_in_window) if len(b2s_in_window) > 0 else 0
    status = "WARN" if pct_unlinked > 15 else "PASS"
    _record(CheckResult(
        name="B2S players with no CFB link (2011-2022 window)",
        status=status,
        message=f"{len(unlinked)} / {len(b2s_in_window)} ({pct_unlinked:.1f}%) have no CFB link",
        flagged=unlinked.sort_values("b2s_score", ascending=False) if not unlinked.empty else None,
        csv_name="crossdb_b2s_no_link",
    ), audit_dir, write_csv)


# ---------------------------------------------------------------------------
# Section 4: Training set validation
# ---------------------------------------------------------------------------

def audit_training(data_dir: Path, audit_dir: Path | None, write_csv: bool) -> None:
    print("\n=== Section 4: Training set ===")

    for pos in _POSITIONS:
        csv_path = data_dir / f"training_{pos}.csv"
        if not csv_path.exists():
            print(f"  [WARN] !!  training_{pos}.csv not found -- run build_training_set.py first")
            continue

        df = pd.read_csv(csv_path)
        n  = len(df)
        print(f"\n  --- {pos} ({n} rows) ---")

        # --- 4.1 Feature coverage matrix ---
        key_features = [
            "best_rec_rate", "best_dominator", "best_reception_share",
            "best_age", "best_games", "best_sp_plus",
            "career_rec_yards", "career_rush_yards", "career_targets",
            "weight_lbs", "forty_time", "speed_score",
            "draft_capital_score", "draft_round", "overall_pick",
            "recruit_rating", "consensus_rank",
            "teammate_score", "b2s_score",
        ]
        coverage = []
        for col in key_features:
            if col not in df.columns:
                coverage.append({"feature": col, "n_non_null": 0, "pct_coverage": 0.0, "note": "MISSING_COLUMN"})
                continue
            n_nn  = df[col].notna().sum()
            pct   = 100 * n_nn / n if n > 0 else 0
            note  = "OK" if pct >= 90 else ("WARN" if pct >= 60 else "LOW")
            coverage.append({"feature": col, "n_non_null": n_nn, "pct_coverage": round(pct, 1), "note": note})
        cov_df = pd.DataFrame(coverage)

        # Print coverage table
        print(f"  {'Feature':<30} {'Coverage':>10}  {'%':>6}  Note")
        print(f"  {'-'*58}")
        for _, r in cov_df.iterrows():
            icon = "OK" if r["note"] == "OK" else ("!!" if r["note"] == "WARN" else "XX")
            print(f"  {r['feature']:<30} {r['n_non_null']:>7}/{n:<4} {r['pct_coverage']:>5.1f}%  {icon} {r['note']}")

        low_cov = cov_df[cov_df["pct_coverage"] < 60]
        status = "WARN" if not low_cov.empty else "PASS"
        _record(CheckResult(
            name=f"Feature coverage [{pos}]",
            status=status,
            message=f"{len(low_cov)} features below 60% coverage",
            flagged=cov_df if write_csv else None,
            csv_name=f"training_coverage_{pos.lower()}",
        ), audit_dir, write_csv)

        # --- 4.2 B2S target distribution ---
        b2s = df["b2s_score"].dropna()
        null_b2s = df["b2s_score"].isna().sum()
        if len(b2s) > 0:
            print(
                f"\n  B2S target [{pos}]: n={len(b2s)} (null={null_b2s}) | "
                f"mean={b2s.mean():.2f} | median={b2s.median():.2f} | "
                f"p25={b2s.quantile(0.25):.2f} | p75={b2s.quantile(0.75):.2f} | "
                f"max={b2s.max():.2f}"
            )
        status = "FAIL" if null_b2s > 0 else "PASS"
        _record(CheckResult(
            name=f"B2S null count [{pos}]",
            status=status,
            message=f"{null_b2s} null b2s_score values in training set",
        ), audit_dir, write_csv)

        # --- 4.3 Draft class coverage ---
        class_counts = df.groupby("draft_year").size().reset_index(name="n_players")
        print(f"\n  Draft class coverage [{pos}]:")
        low_classes = []
        for _, row in class_counts.iterrows():
            flag = ""
            if row["n_players"] < 10:
                flag = "<-- LOW"
                low_classes.append(row["draft_year"])
            print(f"    {int(row['draft_year'])}: {row['n_players']:>3} players  {flag}")
        status = "WARN" if low_classes else "PASS"
        _record(CheckResult(
            name=f"Draft class coverage [{pos}]",
            status=status,
            message=f"{len(low_classes)} classes with <10 players: {low_classes}",
        ), audit_dir, write_csv)

        # --- 4.4 Low match score rows in training set ---
        if "match_score" in df.columns:
            low_ms = df[df["match_score"].fillna(100) < 85][
                ["nfl_name", "cfb_name", "position", "draft_year", "match_score", "b2s_score"]
            ]
            status = "WARN" if not low_ms.empty else "PASS"
            _record(CheckResult(
                name=f"Low match score in training set [{pos}]",
                status=status,
                message=f"{len(low_ms)} rows with match_score < 85 (review link quality)",
                flagged=low_ms if not low_ms.empty else None,
                csv_name=f"training_low_match_{pos.lower()}",
            ), audit_dir, write_csv)

        # --- 4.5 Feature plausibility spot-checks ---
        # NOTE: best_rec_rate is rec_yards / team_pass_att (a RATE, not a proportion).
        # Values 1.0-4.0 are normal for elite WRs. Flag only extreme values (>6.0).
        flags = []
        if "best_rec_rate" in df.columns:
            impossible = df[df["best_rec_rate"].fillna(0) > 6.0]
            if not impossible.empty:
                flags.append(f"best_rec_rate>6.0 (extreme): {len(impossible)}")
        if "best_age" in df.columns:
            bad_age = df[df["best_age"].notna() & ~df["best_age"].between(17, 26)]
            if not bad_age.empty:
                flags.append(f"best_age out of [17,26]: {len(bad_age)}")
        if "draft_capital_score" in df.columns:
            bad_cap = df[df["draft_capital_score"].notna() & ~df["draft_capital_score"].between(0, 100)]
            if not bad_cap.empty:
                flags.append(f"draft_capital_score out of [0,100]: {len(bad_cap)}")

        status = "FAIL" if flags else "PASS"
        _record(CheckResult(
            name=f"Feature plausibility [{pos}]",
            status=status,
            message="; ".join(flags) if flags else "all checked features in valid range",
        ), audit_dir, write_csv)

        # --- 4.6 Wrong-link detection: match_score AND best_season_year vs draft_year ---
        # A training row is suspect if best_season_year >= draft_year
        # (player's best CFB season should be BEFORE they were drafted)
        if "best_season_year" in df.columns and "draft_year" in df.columns:
            wrong_link = df[
                df["best_season_year"].notna() &
                df["draft_year"].notna() &
                (df["best_season_year"] >= df["draft_year"])
            ][["nfl_name", "cfb_name", "position", "draft_year", "best_season_year", "match_score", "b2s_score"]]
            status = "FAIL" if not wrong_link.empty else "PASS"
            _record(CheckResult(
                name=f"Wrong-link detection (best_season_year >= draft_year) [{pos}]",
                status=status,
                message=(
                    f"{len(wrong_link)} rows where best CFB season is in draft year or later "
                    f"-- almost certainly wrong CFB player linked"
                ),
                flagged=wrong_link if not wrong_link.empty else None,
                csv_name=f"training_wrong_links_{pos.lower()}",
            ), audit_dir, write_csv)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary() -> None:
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)

    fails = [r for r in _RESULTS if r.status == "FAIL"]
    warns = [r for r in _RESULTS if r.status == "WARN"]
    passes = [r for r in _RESULTS if r.status == "PASS"]

    print(f"  PASS: {len(passes):>3}")
    print(f"  WARN: {len(warns):>3}")
    print(f"  FAIL: {len(fails):>3}")

    if fails:
        print("\nFAILURES (must fix before model work):")
        for r in fails:
            print(f"  XX  {r.name}: {r.message}")

    if warns:
        print("\nWARNINGS (review manually):")
        for r in warns:
            print(f"  !!  {r.name}: {r.message}")

    overall = "FAIL" if fails else ("WARN" if warns else "PASS")
    print(f"\nOverall: {overall}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data integrity audit for the dynasty-prospect-model pipeline."
    )
    parser.add_argument("--cfb-db", type=str, default=None)
    parser.add_argument("--nfl-db", type=str, default=None)
    parser.add_argument(
        "--section", type=str,
        choices=["cfb", "nfl", "crossdb", "training", "all"],
        default="all",
        help="Run only one audit section (default: all).",
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Print report only; do not write flagged-row CSV files.",
    )
    args = parser.parse_args()

    cfb_db_path = args.cfb_db or get_cfb_db_path()
    nfl_db_path = args.nfl_db or get_nfl_db_path()
    data_dir    = get_data_dir()
    write_csv   = not args.no_csv

    audit_dir: Path | None = None
    if write_csv:
        audit_dir = data_dir / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)

    sections = [args.section] if args.section != "all" else ["cfb", "nfl", "crossdb", "training"]
    needs_cfb = any(s in sections for s in ("cfb", "crossdb"))
    needs_nfl = any(s in sections for s in ("nfl", "crossdb"))

    cfb = _load_cfb_db(cfb_db_path) if needs_cfb else {}
    nfl = _load_nfl_db(nfl_db_path) if needs_nfl else {}

    if "cfb" in sections:
        audit_cfb(cfb, audit_dir, write_csv)
    if "nfl" in sections:
        audit_nfl(nfl, audit_dir, write_csv)
    if "crossdb" in sections:
        audit_crossdb(cfb, nfl, audit_dir, write_csv)
    if "training" in sections:
        audit_training(data_dir, audit_dir, write_csv)

    _print_summary()


if __name__ == "__main__":
    main()
