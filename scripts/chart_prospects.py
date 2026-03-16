"""
Campus2Canton-style scatter charts: model metrics by year out of high school.

For each chart:
  - Gray dots: all historical player-seasons (min 6 games played)
  - Colored + labeled dots: top 2026 prospects
  - Dashed line: average of Round 1-2 draft picks at each year_out

Output: output/charts/
"""

import sys
import sqlite3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
CFB_DB = Path(r"C:\Users\Ozimek\Documents\Claude\FF\cfb-prospect-db\ff.db")
SCORES_CSV = REPO_ROOT / "output" / "scores" / "scores_2026_ridge.csv"
OUT_DIR = REPO_ROOT / "output" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Chart configuration
# Correct y_lim from actual data distributions (p99 caps)
# ---------------------------------------------------------------------------

CHARTS = [
    dict(
        metric="rec_yards_per_team_pass_att",
        position="WR",
        y_label="Rec Yds Per Team Pass Att",
        title="Experience Adjusted Rec Yds Per Team Pass Att — WR",
        stub="WR_YPTA",
        y_lim=(-0.1, 4.0),
        top_n=12,
    ),
    dict(
        metric="ppa_avg_pass",
        position="WR",
        y_label="PPA Pass (Avg)",
        title="Experience Adjusted PPA Pass Average — WR",
        stub="WR_PPA",
        y_lim=(-0.1, 2.2),
        top_n=12,
    ),
    dict(
        metric="dominator_rating",
        position="RB",
        y_label="Dominator Rating",
        title="Experience Adjusted Dominator Rating — RB",
        stub="RB_DOM",
        y_lim=(-0.01, 0.25),
        top_n=10,
    ),
    dict(
        metric="usage_pass",
        position="RB",
        y_label="Pass Usage Rate",
        title="Experience Adjusted Pass Usage Rate — RB",
        stub="RB_USAGE_PASS",
        y_lim=(-0.01, 0.20),
        top_n=10,
    ),
    dict(
        metric="rec_yards_per_team_pass_att",
        position="TE",
        y_label="Rec Yds Per Team Pass Att",
        title="Experience Adjusted Rec Yds Per Team Pass Att — TE",
        stub="TE_YPTA",
        y_lim=(-0.1, 2.5),
        top_n=10,
    ),
    dict(
        metric="ppa_avg_pass",
        position="TE",
        y_label="PPA Pass (Avg)",
        title="Experience Adjusted PPA Pass Average — TE",
        stub="TE_PPA",
        y_lim=(-0.1, 2.0),
        top_n=10,
    ),
]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

BG_COLOR     = "#ebebeb"
TITLE_COLOR  = "#8b1a1a"
SUB_COLOR    = "#555555"
GRAY_DOT     = "#aaaaaa"
ELITE_COLOR  = "#8b2500"
AXIS_COLOR   = "#333333"
GRID_COLOR   = "#cccccc"

PROSPECT_COLORS = [
    "#1b7837", "#762a83", "#e08214", "#2166ac", "#d6604d",
    "#4dac26", "#9970ab", "#f1a340", "#0571b0", "#ca0020",
    "#41ab5d", "#7b3294",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_seasons() -> pd.DataFrame:
    con = sqlite3.connect(str(CFB_DB))
    df = pd.read_sql("""
        SELECT
            s.player_id,
            p.full_name,
            p.position,
            p.declared_draft_year,
            s.season_year,
            s.games_played,
            s.rec_yards_per_team_pass_att,
            s.ppa_avg_pass,
            s.dominator_rating,
            s.usage_pass,
            COALESCE(r.recruit_year, NULL) AS recruit_year
        FROM cfb_player_seasons s
        JOIN players p ON p.id = s.player_id
        LEFT JOIN (
            SELECT player_id, MIN(recruit_year) AS recruit_year
            FROM recruiting GROUP BY player_id
        ) r ON r.player_id = s.player_id
        WHERE p.position IN ('WR','RB','TE')
          AND s.games_played >= 6
        ORDER BY s.player_id, s.season_year
    """, con)

    picks = pd.read_sql(
        "SELECT player_id FROM nfl_draft_picks WHERE draft_round <= 2", con
    )
    con.close()

    # year_out_of_hs: prefer recruit_year, fall back to min season_year
    min_sy = df.groupby("player_id")["season_year"].min().rename("min_sy")
    df = df.join(min_sy, on="player_id")
    df["year1"] = df["recruit_year"].fillna(df["min_sy"])
    df["year_out"] = (df["season_year"] - df["year1"] + 1).astype(int)
    df = df[(df["year_out"] >= 1) & (df["year_out"] <= 5)]

    elite_ids = set(picks["player_id"])
    df["is_elite"] = df["player_id"].isin(elite_ids)

    return df


def load_scores(position: str) -> list[str]:
    scores = pd.read_csv(SCORES_CSV)
    return scores[scores["position"] == position]["player_name"].tolist()


# ---------------------------------------------------------------------------
# Single chart
# ---------------------------------------------------------------------------

def draw_chart(cfg: dict, df: pd.DataFrame) -> None:
    pos     = cfg["position"]
    metric  = cfg["metric"]
    y_lo, y_hi = cfg["y_lim"]
    top_n   = cfg["top_n"]

    # Position-filtered data
    pos_df = df[df["position"] == pos].dropna(subset=[metric]).copy()

    # 2026 prospects + their names from DB
    p26_df = pos_df[pos_df["declared_draft_year"] == 2026].copy()
    p26_names = set(p26_df["full_name"].unique())

    # Match score-order names to DB full_name
    score_names = load_scores(pos)
    matched: list[tuple[str, str]] = []  # (display_name, db_full_name)
    seen_db = set()
    for sn in score_names:
        if sn in p26_names and sn not in seen_db:
            matched.append((sn, sn))
            seen_db.add(sn)
        else:
            # last-name fallback
            last = sn.split()[-1].lower()
            for n in p26_names:
                if n.split()[-1].lower() == last and n not in seen_db:
                    matched.append((sn, n))
                    seen_db.add(n)
                    break
        if len(matched) >= top_n:
            break

    # Background (non-2026) data
    bg_df = pos_df[pos_df["declared_draft_year"] != 2026].copy()

    # Clip background to y_lim for visual clarity
    bg_df = bg_df[(bg_df[metric] >= y_lo) & (bg_df[metric] <= y_hi)]

    # Elite average per year_out (clipped to y range)
    elite_avg = (
        bg_df[bg_df["is_elite"]]
        .groupby("year_out")[metric]
        .mean()
        .reindex([1, 2, 3, 4, 5])
        .clip(lower=y_lo, upper=y_hi)
    )

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.tick_params(colors=AXIS_COLOR, labelsize=9)

    rng = np.random.default_rng(42)

    # Gray background dots
    jitter = rng.uniform(-0.12, 0.12, len(bg_df))
    ax.scatter(
        bg_df["year_out"] + jitter,
        bg_df[metric],
        color=GRAY_DOT,
        s=14,
        alpha=0.55,
        linewidths=0,
        zorder=2,
        clip_on=True,
    )

    # Elite average dashed line (only draw if values exist inside y range)
    valid = elite_avg.dropna()
    valid = valid[(valid >= y_lo) & (valid <= y_hi)]
    if len(valid) >= 2:
        ax.plot(
            valid.index,
            valid.values,
            color=ELITE_COLOR,
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
            zorder=3,
            clip_on=True,
        )
        # Inline label placed at 97% of x-range, at line level
        last_v = valid.values[-1]
        ax.annotate(
            "Avg Round 1-2 picks",
            xy=(valid.index[-1], last_v),
            xytext=(8, 2),
            textcoords="offset points",
            fontsize=7.5,
            color=ELITE_COLOR,
            style="italic",
            va="center",
            annotation_clip=True,
            zorder=7,
        )

    # 2026 prospect dots + labels
    for i, (display_name, db_name) in enumerate(matched):
        color = PROSPECT_COLORS[i % len(PROSPECT_COLORS)]
        p_rows = p26_df[p26_df["full_name"] == db_name].dropna(subset=[metric]).sort_values("year_out")
        if p_rows.empty:
            continue

        # All dots for this prospect
        ax.scatter(
            p_rows["year_out"],
            p_rows[metric],
            color=color,
            s=60,
            zorder=5,
            linewidths=0.6,
            edgecolors="white",
            clip_on=True,
        )

        # Label on every dot
        for _, row in p_rows.iterrows():
            if y_lo <= row[metric] <= y_hi:
                ax.annotate(
                    display_name,
                    xy=(row["year_out"], row[metric]),
                    xytext=(6, 1),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=color,
                    fontweight="bold",
                    annotation_clip=True,
                    zorder=8,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground=BG_COLOR)],
                )

    # Enforce y limits AFTER all plotting
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel("Year out of High School", color=AXIS_COLOR, fontsize=10, labelpad=6)
    ax.set_ylabel(cfg["y_label"], color=AXIS_COLOR, fontsize=10, labelpad=6)

    # Title block
    ax.set_title(cfg["title"], color=TITLE_COLOR, fontsize=13, fontweight="bold",
                 loc="left", pad=10)
    ax.text(0.0, 1.012,
            "dynasty-prospect-model  |  Min. 6 Games Played  |  Top 2026 prospects highlighted",
            transform=ax.transAxes, color=SUB_COLOR, fontsize=8)

    # Attribution
    fig.text(0.985, 0.015, "Data: CFBD / nflverse",
             ha="right", va="bottom", fontsize=7, color=SUB_COLOR, style="italic")

    plt.tight_layout(pad=1.2)
    out_path = OUT_DIR / f"{cfg['stub']}.png"
    fig.savefig(out_path, dpi=150, bbox_inches=None, facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", choices=["WR", "RB", "TE"])
    args = parser.parse_args()

    print("Loading data...")
    df = load_all_seasons()
    print(f"  Loaded {len(df):,} player-seasons")

    for cfg in CHARTS:
        if args.position and cfg["position"] != args.position:
            continue
        print(f"Generating: {cfg['stub']}...")
        draw_chart(cfg, df)

    print(f"\nDone. Charts in: {OUT_DIR}")


if __name__ == "__main__":
    main()
