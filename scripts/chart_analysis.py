"""
Analytical charts for dynasty prospect evaluation.

Charts produced:
  1. Draft Capital vs B2S calibration (3-panel)
  2. ORBIT Value Map — 2026 class: capital vs model surplus
  3. Breakout Age vs B2S scatter (WR + 2026 overlaid)
  4. Career trajectory by outcome tier (hits / developing / misses)
  5. 2026 class depth vs historical classes (violin + box)
  6. Feature univariate R² bar chart (per position)

Output: output/charts/
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
CFB_DB     = Path(r"C:\Users\Ozimek\Documents\Claude\FF\cfb-prospect-db\ff.db")
DATA_DIR   = REPO_ROOT / "data"
OUT_DIR    = REPO_ROOT / "output" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG      = "#ebebeb"
TITLE_C = "#8b1a1a"
SUB_C   = "#555555"
GRID_C  = "#cccccc"
AXIS_C  = "#333333"

POS_COLORS = {"WR": "#2166ac", "RB": "#1b7837", "TE": "#762a83"}
TIER_COLORS = {"Hit (B2S≥10)": "#d6604d", "Developing (3–10)": "#f1a340", "Bust (B2S<3)": "#aaaaaa"}


def _style(ax, title, subtitle=""):
    ax.set_facecolor(BG)
    ax.yaxis.grid(True, color=GRID_C, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID_C)
    ax.tick_params(colors=AXIS_C, labelsize=9)
    ax.set_title(title, color=TITLE_C, fontsize=12, fontweight="bold", loc="left", pad=10)
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, color=SUB_C, fontsize=8)


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# 1. Draft Capital vs B2S calibration
# ---------------------------------------------------------------------------

def chart_capital_vs_b2s():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Draft Capital vs. B2S Score — Training Set (2011–2022)",
                 color=TITLE_C, fontsize=13, fontweight="bold", x=0.02, ha="left", y=1.01)

    rng = np.random.default_rng(42)

    for ax, pos in zip(axes, ["WR", "RB", "TE"]):
        df = pd.read_csv(DATA_DIR / f"training_{pos}.csv").dropna(subset=["draft_capital_score","b2s_score"])
        color = POS_COLORS[pos]

        # Jitter x slightly so stacked dots separate
        jx = rng.uniform(-0.4, 0.4, len(df))
        ax.scatter(df["draft_capital_score"] + jx, df["b2s_score"],
                   color=color, s=16, alpha=0.55, linewidths=0, zorder=3, clip_on=True)

        # Regression line
        x = df["draft_capital_score"].values
        y = df["b2s_score"].values
        m, b = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 100)
        ax.plot(xr, m * xr + b, color=AXIS_C, lw=1.2, ls="--", alpha=0.7, zorder=4)

        r, _ = pearsonr(x, y)
        # Bracket capitals into round tiers for median line
        bins = [0, 10, 30, 55, 75, 101]
        labels = ["R5–7", "R3–4", "R2", "R1 late", "R1 top"]
        df["tier"] = pd.cut(df["draft_capital_score"], bins=bins, labels=labels, right=True)
        tier_med = df.groupby("tier", observed=True)["b2s_score"].median()
        tier_mid = [5, 20, 42.5, 65, 88]
        ax.plot(tier_mid, tier_med.values, color=color, lw=2.2, marker="D",
                markersize=6, zorder=5, label="Median by capital tier")

        _style(ax, pos, f"r = {r:.2f} | n={len(df)}")
        ax.set_xlabel("Draft Capital Score", color=AXIS_C, fontsize=9)
        ax.set_ylabel("B2S Score (avg best 2 NFL seasons)", color=AXIS_C, fontsize=9)
        ax.set_xlim(-2, 100)
        ax.set_ylim(-0.5, df["b2s_score"].max() * 1.05)
        ax.legend(fontsize=8, framealpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save(fig, "ANALYSIS_01_capital_vs_b2s.png")


# ---------------------------------------------------------------------------
# 2. ORBIT Value Map — 2026 class
# ---------------------------------------------------------------------------

def chart_orbit_value_map():
    s26 = pd.read_csv(REPO_ROOT / "output" / "scores" / "scores_2026_ridge.csv")
    s26 = s26[s26["orbit_score"] > 0].copy()

    # We want ORBIT vs position_rank; over/under the OLS trendline = value/overpriced
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BG)

    # Fit line across all positions: ORBIT ~ position_rank
    valid = s26.dropna(subset=["position_rank", "orbit_score"])
    x_fit = valid["position_rank"].values
    y_fit = valid["orbit_score"].values
    m, b = np.polyfit(x_fit, y_fit, 1)
    xr = np.linspace(x_fit.min(), x_fit.max(), 200)
    ax.plot(xr, m * xr + b, color=AXIS_C, lw=1.2, ls="--", alpha=0.6, zorder=2, label="Expected ORBIT")

    # Shade above/below
    ax.fill_between(xr, m * xr + b, (m * xr + b) + 30, alpha=0.05, color="#1b7837")
    ax.fill_between(xr, (m * xr + b) - 30, m * xr + b, alpha=0.05, color="#d6604d")

    # Plot each prospect
    rng = np.random.default_rng(7)
    for pos, grp in s26.groupby("position"):
        color = POS_COLORS.get(pos, "#555555")
        jy = rng.uniform(-0.8, 0.8, len(grp))
        ax.scatter(grp["position_rank"], grp["orbit_score"] + jy,
                   color=color, s=28, alpha=0.75, linewidths=0.4,
                   edgecolors="white", zorder=4, label=pos)

    # Label notable divergences (biggest surplus above/below trend)
    for pos, grp in s26.groupby("position"):
        color = POS_COLORS.get(pos, "#555555")
        grp = grp.dropna(subset=["position_rank"])
        grp = grp.copy()
        grp["expected"] = m * grp["position_rank"] + b
        grp["surplus"] = grp["orbit_score"] - grp["expected"]

        # Top 4 surplus (value) + top 2 deficit (expensive) per position
        top_val = grp.nlargest(4, "surplus")
        top_exp = grp.nsmallest(2, "surplus")

        for _, row in pd.concat([top_val, top_exp]).iterrows():
            ax.annotate(
                row["player_name"],
                xy=(row["position_rank"], row["orbit_score"]),
                xytext=(5, 2), textcoords="offset points",
                fontsize=7, color=color, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=1.8, foreground=BG)],
                zorder=6,
            )

    _style(ax, "2026 ORBIT Value Map — Model vs. Draft Capital",
           "Above trend = model upside relative to draft cost  |  Below trend = draft capital leads model signal")
    ax.set_xlabel("Position Rank (consensus draft board)", color=AXIS_C, fontsize=10)
    ax.set_ylabel("ORBIT Score (0–100)", color=AXIS_C, fontsize=10)
    ax.set_xlim(0, s26["position_rank"].max() * 1.05)
    ax.set_ylim(-5, 105)

    handles = [mpatches.Patch(color=POS_COLORS[p], label=p) for p in ["WR","RB","TE"]]
    handles.append(plt.Line2D([0],[0], color=AXIS_C, ls="--", lw=1.2, label="Expected ORBIT"))
    ax.legend(handles=handles, fontsize=9, framealpha=0.5, loc="upper right")

    fig.text(0.985, 0.015, "Data: CFBD / nflverse", ha="right", va="bottom",
             fontsize=7, color=SUB_C, style="italic")
    plt.tight_layout(pad=1.2)
    save(fig, "ANALYSIS_02_orbit_value_map.png")


# ---------------------------------------------------------------------------
# 3. Breakout Age vs B2S (WR + 2026 overlaid)
# ---------------------------------------------------------------------------

def chart_breakout_age():
    wr = pd.read_csv(DATA_DIR / "training_WR.csv").dropna(subset=["breakout_age","b2s_score"])
    s26 = pd.read_csv(REPO_ROOT / "output" / "scores" / "scores_2026_ridge.csv")
    s26_wr = s26[s26["position"] == "WR"].dropna(subset=["breakout_age"])

    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor(BG)

    # Background: all training WRs, colored by b2s tier
    def tier(b):
        if b >= 10: return "Hit (B2S≥10)"
        if b >= 3:  return "Developing (3–10)"
        return "Bust (B2S<3)"

    wr["tier"] = wr["b2s_score"].apply(tier)
    rng = np.random.default_rng(99)

    for t, grp in wr.groupby("tier"):
        jx = rng.uniform(-0.06, 0.06, len(grp))
        jy = rng.uniform(-0.15, 0.15, len(grp))
        ax.scatter(grp["breakout_age"] + jx, grp["b2s_score"] + jy,
                   color=TIER_COLORS[t], s=22, alpha=0.55, linewidths=0,
                   zorder=3, label=t, clip_on=True)

    # Regression line
    x = wr["breakout_age"].values
    y = wr["b2s_score"].values
    m, b_int = np.polyfit(x, y, 1)
    xr = np.linspace(x.min(), x.max(), 100)
    ax.plot(xr, m * xr + b_int, color=TITLE_C, lw=1.4, ls="--", alpha=0.8, zorder=4)
    r, _ = pearsonr(x, y)

    # 2026 prospects as vertical dashed lines + label at top
    for _, row in s26_wr.iterrows():
        ax.axvline(row["breakout_age"], color="#2166ac", alpha=0.25, lw=0.8, zorder=2)
        if row["orbit_score"] >= 60:
            ax.text(row["breakout_age"] + 0.03, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 20,
                    row["player_name"], fontsize=6.5, color="#2166ac",
                    rotation=90, va="top", ha="left",
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=BG)])

    _style(ax, "Breakout Age vs. B2S Score — WR (2011–2022 Training)",
           f"r = {r:.2f} | Earlier breakout → higher NFL ceiling  |  Blue lines = 2026 WR prospects (ORBIT≥60)")
    ax.set_xlabel("Breakout Age (age in best college season)", color=AXIS_C, fontsize=10)
    ax.set_ylabel("B2S Score (avg best 2 NFL seasons)", color=AXIS_C, fontsize=10)
    ax.set_xlim(17.5, 24.5)
    ax.set_ylim(-0.5, wr["b2s_score"].max() * 1.05)

    handles = [mpatches.Patch(color=TIER_COLORS[t], label=t)
               for t in ["Hit (B2S≥10)", "Developing (3–10)", "Bust (B2S<3)"]]
    ax.legend(handles=handles, fontsize=9, framealpha=0.5)

    fig.text(0.985, 0.015, "Data: CFBD / nflverse", ha="right", va="bottom",
             fontsize=7, color=SUB_C, style="italic")
    plt.tight_layout(pad=1.2)
    save(fig, "ANALYSIS_03_breakout_age_b2s.png")


# ---------------------------------------------------------------------------
# 4. Career trajectory: hits vs. misses (WR from cfb DB)
# ---------------------------------------------------------------------------

def chart_trajectory_tiers():
    # Load training WR ids + b2s
    wr = pd.read_csv(DATA_DIR / "training_WR.csv")[["cfb_player_id","b2s_score","nfl_name"]].dropna()

    # Load per-season data from cfb DB
    con = sqlite3.connect(str(CFB_DB))
    seasons = pd.read_sql("""
        SELECT s.player_id, s.season_year, s.games_played,
               s.rec_yards_per_team_pass_att, s.ppa_avg_pass
        FROM cfb_player_seasons s
        JOIN players p ON p.id = s.player_id
        WHERE p.position = 'WR' AND s.games_played >= 6
    """, con)

    recruiting = pd.read_sql("""
        SELECT player_id, MIN(recruit_year) AS recruit_year
        FROM recruiting GROUP BY player_id
    """, con)
    con.close()

    # year_out
    min_sy = seasons.groupby("player_id")["season_year"].min().rename("min_sy")
    seasons = seasons.join(min_sy, on="player_id")
    seasons = seasons.join(recruiting.set_index("player_id")["recruit_year"], on="player_id")
    seasons["year1"] = seasons["recruit_year"].fillna(seasons["min_sy"])
    seasons["year_out"] = (seasons["season_year"] - seasons["year1"] + 1).astype(int)
    seasons = seasons[(seasons["year_out"] >= 1) & (seasons["year_out"] <= 5)]

    # Merge with training set
    merged = seasons.merge(wr, left_on="player_id", right_on="cfb_player_id")

    def tier(b):
        if b >= 10: return "Hit (B2S≥10)"
        if b >= 3:  return "Developing (3–10)"
        return "Bust (B2S<3)"

    merged["tier"] = merged["b2s_score"].apply(tier)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Career Production Trajectory by Outcome Tier — WR (2011–2022 Training)",
                 color=TITLE_C, fontsize=12, fontweight="bold", x=0.02, ha="left", y=1.01)

    for ax, metric, ylabel in zip(
        axes,
        ["rec_yards_per_team_pass_att", "ppa_avg_pass"],
        ["Rec Yds Per Team Pass Att", "PPA Pass (Avg)"]
    ):
        for tier_name, color in TIER_COLORS.items():
            grp = merged[merged["tier"] == tier_name].dropna(subset=[metric])
            stats = grp.groupby("year_out")[metric].agg(["median","sem"]).reindex([1,2,3,4,5])
            med = stats["median"].values
            se  = stats["sem"].fillna(0).values
            xvals = np.array([1,2,3,4,5])
            mask = ~np.isnan(med)
            ax.plot(xvals[mask], med[mask], color=color, lw=2.2, marker="o",
                    markersize=5, zorder=4, label=tier_name)
            ax.fill_between(xvals[mask],
                            (med - se)[mask], (med + se)[mask],
                            color=color, alpha=0.15, zorder=3)

        _style(ax, ylabel)
        ax.set_xlabel("Year out of High School", color=AXIS_C, fontsize=9)
        ax.set_ylabel(ylabel, color=AXIS_C, fontsize=9)
        ax.set_xticks([1,2,3,4,5])
        ax.set_xlim(0.7, 5.3)
        ax.legend(fontsize=8.5, framealpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save(fig, "ANALYSIS_04_trajectory_tiers.png")


# ---------------------------------------------------------------------------
# 5. 2026 class depth vs. historical classes
# ---------------------------------------------------------------------------

def chart_class_depth():
    # Load training sets for all positions
    all_train = []
    for pos in ["WR","RB","TE"]:
        df = pd.read_csv(DATA_DIR / f"training_{pos}.csv")
        df["pos"] = pos
        all_train.append(df)
    train = pd.concat(all_train, ignore_index=True)

    # Load 2026 prospects (scored)
    s26 = pd.read_csv(REPO_ROOT / "output" / "scores" / "scores_2026_ridge.csv")
    s26["draft_year"] = 2026
    s26 = s26.rename(columns={"best_rec_rate": "best_rec_rate", "position": "pos"})

    # Metric to compare: best_rec_rate (closest to production dominance)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("College Production Depth by Draft Class — Best Rec Rate\n(Gray bars = 2011–2022 training; colored = 2026 class)",
                 color=TITLE_C, fontsize=12, fontweight="bold", x=0.02, ha="left", y=1.04)

    for ax, pos in zip(axes, ["WR", "RB", "TE"]):
        color = POS_COLORS[pos]
        tr = train[train["pos"] == pos].copy()
        s26p = s26[s26["pos"] == pos].copy()

        # Historical: box per year
        years = sorted(tr["draft_year"].unique())
        positions = list(range(len(years)))

        data_per_year = [tr[tr["draft_year"] == y]["best_rec_rate"].dropna().values for y in years]

        bp = ax.boxplot(data_per_year, positions=positions, widths=0.55,
                        patch_artist=True, notch=False, showfliers=False,
                        medianprops=dict(color=AXIS_C, linewidth=1.5),
                        whiskerprops=dict(color=GRID_C),
                        capprops=dict(color=GRID_C),
                        flierprops=dict(marker=".", color=GRID_C, alpha=0.5))

        for patch in bp["boxes"]:
            patch.set_facecolor("#cccccc")
            patch.set_edgecolor(GRID_C)
            patch.set_alpha(0.7)

        # 2026 violin / strip next to last year
        x_2026 = len(years)
        s26_vals = s26p["best_rec_rate"].dropna().values
        if len(s26_vals) > 0:
            jitter = np.random.default_rng(55).uniform(-0.18, 0.18, len(s26_vals))
            ax.scatter([x_2026] * len(s26_vals) + jitter, s26_vals,
                       color=color, s=18, alpha=0.65, zorder=5)
            ax.boxplot([s26_vals], positions=[x_2026], widths=0.55,
                       patch_artist=True, notch=False, showfliers=False,
                       medianprops=dict(color="white", linewidth=2),
                       whiskerprops=dict(color=color),
                       capprops=dict(color=color),
                       boxprops=dict(facecolor=color, alpha=0.3, edgecolor=color))

        ax.set_xticks(list(positions) + [x_2026])
        ax.set_xticklabels([str(y)[2:] for y in years] + ["'26"], fontsize=8)
        ax.set_xlabel("Draft Class", color=AXIS_C, fontsize=9)
        ax.set_ylabel("Best Rec Rate (rec yds / team pass att)", color=AXIS_C, fontsize=9)
        _style(ax, pos)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "ANALYSIS_05_class_depth_history.png")


# ---------------------------------------------------------------------------
# 6. Feature univariate R² bar chart
# ---------------------------------------------------------------------------

def chart_univariate_r2():
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Univariate R² vs. B2S Score — Feature Importance by Position",
                 color=TITLE_C, fontsize=12, fontweight="bold", x=0.02, ha="left", y=1.01)

    # Human-readable feature names
    LABELS = {
        "log_draft_capital":           "Log Draft Capital",
        "draft_capital_score":         "Draft Capital Score",
        "overall_pick":                "Overall Pick #",
        "consensus_rank":              "Consensus Big Board Rank",
        "position_rank":               "Position Rank",
        "breakout_score_x_capital":    "Breakout Score × Capital",
        "capital_x_age":               "Capital × Age",
        "capital_x_dominator":         "Capital × Dominator",
        "best_breakout_score":         "Breakout Score",
        "best_rec_rate":               "Rec Yds Per Team Pass Att",
        "best_ppa_pass":               "PPA Pass (Avg)",
        "best_age":                    "Age at Best Season",
        "breakout_age":                "Breakout Age",
        "best_usage_pass":             "Pass Usage Rate",
        "college_fantasy_ppg":         "College Fantasy PPG",
        "career_yardage":              "Career Yardage",
        "best_dominator":              "Dominator Rating",
        "weight_lbs":                  "Weight (lbs)",
        "combined_ath":                "Combined Athletic Score",
        "speed_score":                 "Speed Score",
        "forty_time":                  "40-Yard Dash",
        "agility_score":               "Agility Score",
        "recruit_rating":              "Recruit Rating",
        "draft_premium":               "Draft Premium (pos adj.)",
        "early_declare":               "Early Declare",
        "power4_conf":                 "Power-4 Conference",
        "best_rec_td_pct":             "Rec TD Share",
        "best_total_yards_rate":       "Total Yards Rate",
    }

    for ax, pos in zip(axes, ["WR", "RB", "TE"]):
        color = POS_COLORS[pos]
        uni = pd.read_csv(REPO_ROOT / "output" / "analysis" / f"univariate_{pos}_2011_2022.csv")
        # top 15 by r2
        uni = uni.sort_values("r2", ascending=False).head(15)
        uni["label"] = uni["feature"].map(LABELS).fillna(uni["feature"])

        bars = ax.barh(range(len(uni)), uni["r2"].values,
                       color=color, alpha=0.75, zorder=3)

        # Error bars (CI)
        if "r2_ci_lo" in uni.columns and "r2_ci_hi" in uni.columns:
            xerr_lo = (uni["r2"] - uni["r2_ci_lo"]).clip(0).values
            xerr_hi = (uni["r2_ci_hi"] - uni["r2"]).clip(0).values
            ax.errorbar(uni["r2"].values, range(len(uni)),
                        xerr=[xerr_lo, xerr_hi],
                        fmt="none", color=AXIS_C, lw=0.9, capsize=2, zorder=4)

        ax.set_yticks(range(len(uni)))
        ax.set_yticklabels(uni["label"].values, fontsize=8)
        ax.set_xlabel("Univariate R² vs. B2S", color=AXIS_C, fontsize=9)
        ax.set_xlim(0, uni["r2"].max() * 1.2)
        ax.invert_yaxis()
        _style(ax, pos, "Lasso-selected features starred  |  LOYO-CV estimated")

        # Star Lasso-selected features
        LASSO = {
            "WR": {"log_draft_capital","draft_capital_score","breakout_score_x_capital",
                   "best_ppa_pass","best_age","early_declare"},
            "RB": {"log_draft_capital","capital_x_age","capital_x_dominator","position_rank",
                   "best_usage_pass","college_fantasy_ppg","career_yardage","weight_lbs","combined_ath"},
            "TE": {"overall_pick","draft_premium","capital_x_age","breakout_score_x_capital",
                   "position_rank","weight_lbs","agility_score","forty_time","recruit_rating"},
        }
        for i, feat in enumerate(uni["feature"].values):
            if feat in LASSO.get(pos, set()):
                ax.text(uni["r2"].iloc[i] + 0.003, i, "★",
                        va="center", fontsize=8, color=TITLE_C)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save(fig, "ANALYSIS_06_univariate_r2.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--chart", type=int, help="Only run chart N (1–6)")
    args = parser.parse_args()

    charts = {
        1: chart_capital_vs_b2s,
        2: chart_orbit_value_map,
        3: chart_breakout_age,
        4: chart_trajectory_tiers,
        5: chart_class_depth,
        6: chart_univariate_r2,
    }

    for n, fn in charts.items():
        if args.chart and n != args.chart:
            continue
        print(f"Generating chart {n}: {fn.__name__}...")
        fn()

    print(f"\nDone. Charts in: {OUT_DIR}")
