"""
Dynasty Prospect Viewer
=======================
A tkinter GUI for exploring 2026 draft class ZAP scores, Phase I scores,
capital deltas, and full prospect breakdowns.

Usage:
    python app/prospect_viewer.py
    python app/prospect_viewer.py --year 2025

Requires:
    - output/scores/scores_{year}_ridge.csv must exist
    - scripts/fit_model.py --all (and optionally --no-capital) must have run
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).parent.parent

# ── colours ────────────────────────────────────────────────────────────────
BG        = "#1a1d23"
PANEL     = "#22262e"
ACCENT    = "#2e86de"
GOLD      = "#f9ca24"
TEXT      = "#ecf0f1"
SUBTEXT   = "#95a5a6"
HIGH_RISK = "#e74c3c"
LOW_RISK  = "#2ecc71"
NEUTRAL   = "#7f8c8d"

POS_COLORS = {"WR": "#3498db", "RB": "#2ecc71", "TE": "#e67e22"}
RISK_COLORS = {"High Risk": HIGH_RISK, "Low Risk": LOW_RISK,
               "Neutral": NEUTRAL, "N/A": SUBTEXT}

# ── human-readable column labels ───────────────────────────────────────────
COL_LABELS = {
    "pos_rank": "Rank", "player_name": "Player", "position": "Pos",
    "zap_score": "ZAP", "phase1_zap": "Ph1", "capital_delta": "Delta",
    "risk": "Risk", "projected_b2s": "Proj B2S",
    "best_dominator": "Dominator", "best_rec_rate": "Rec Rate",
    "college_fantasy_ppg": "CFP PPG", "breakout_age": "BO Age",
    "best_age": "Best Age", "weight_lbs": "Weight",
    "speed_score": "Speed Score", "forty_time": "40yd",
    "broad_jump": "Broad Jump", "vertical_jump": "Vert",
    "consensus_rank": "Board", "overall_pick": "Proj Pick",
    "draft_capital_score": "Capital",
}

# ── radar chart config ─────────────────────────────────────────────────────
RADAR_FEATURES = {
    "WR": [
        ("Breakout Score",   "best_breakout_score"),
        ("Dominator",        "best_dominator"),
        ("YPRR",             "best_yprr"),
        ("Man YPRR",         "best_man_yprr"),
        ("Zone YPRR",        "best_zone_yprr"),
        ("Slot YPRR",        "best_slot_yprr"),
        ("Deep YPRR",        "best_deep_yprr"),
        ("Age Score",        "best_age_inv"),   # inverted: younger = better
    ],
    "RB": [
        ("Total Yds Rate",   "best_total_yards_rate"),
        ("Breakout Score",   "best_breakout_score"),
        ("Rec Rate",         "best_rec_rate"),
        ("Pass Usage",       "best_usage_pass"),
        ("YPRR",             "best_yprr"),
        ("Zone YPRR",        "best_zone_yprr"),
        ("Speed Score",      "speed_score"),
        ("Age Score",        "best_age_inv"),
    ],
    "TE": [
        ("Breakout Score",   "best_breakout_score"),
        ("Dominator",        "best_dominator"),
        ("YPRR",             "best_yprr"),
        ("Slot YPRR",        "best_slot_yprr"),
        ("40yd Speed",       "forty_inv"),       # inverted: faster = better
        ("Broad Jump",       "broad_jump"),
        ("Rec Grade",        "best_receiving_grade"),
        ("Age Score",        "best_age_inv"),
    ],
}


def _load_scores(year: int) -> pd.DataFrame:
    path = _ROOT / "output" / "scores" / f"scores_{year}_ridge.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Score file not found: {path}\n"
            f"Run: python scripts/score_class.py --year {year} --phase1"
        )
    df = pd.read_csv(path)
    # Ensure phase1 columns exist (graceful fallback)
    for col in ("phase1_zap", "capital_delta", "risk"):
        if col not in df.columns:
            df[col] = np.nan if col != "risk" else "N/A"
    return df


def _load_training_distributions() -> dict:
    """Load per-position per-feature percentile distributions from training CSVs."""
    data_dir = _ROOT / "data"
    dists = {}
    for pos in ("WR", "RB", "TE"):
        p = data_dir / f"{pos}_training.csv"
        if p.exists():
            dists[pos] = pd.read_csv(p)
    return dists


def _percentile_of(value, series: pd.Series) -> float:
    """Return percentile rank of value within series (0-100)."""
    if pd.isna(value) or len(series) == 0:
        return 50.0
    clean = series.dropna()
    if len(clean) == 0:
        return 50.0
    return float((clean < value).mean() * 100)


def _build_radar_values(row: pd.Series, pos: str,
                        train_df: pd.DataFrame | None) -> tuple:
    """Return (labels, values 0-1) for radar chart."""
    features = RADAR_FEATURES.get(pos, [])
    labels, vals = [], []

    # Add derived columns for radar
    row = row.copy()
    if "best_age" in row and pd.notna(row["best_age"]):
        row["best_age_inv"] = max(0.0, 26.0 - float(row["best_age"]))
    else:
        row["best_age_inv"] = np.nan

    if "forty_time" in row and pd.notna(row["forty_time"]):
        row["forty_inv"] = max(0.0, 5.5 - float(row["forty_time"]))
    else:
        row["forty_inv"] = np.nan

    for label, feat in features:
        val = row.get(feat, np.nan)
        if train_df is not None and feat in train_df.columns:
            pct = _percentile_of(val, train_df[feat]) / 100.0
        else:
            # Fallback: normalise to [0,1] using rough scale
            pct = 0.5 if pd.isna(val) else min(1.0, max(0.0, float(val) / 20.0))
        labels.append(label)
        vals.append(pct)

    return labels, vals


# ═══════════════════════════════════════════════════════════════════════════
# Main Application
# ═══════════════════════════════════════════════════════════════════════════

class ProspectViewer(tk.Tk):
    def __init__(self, year: int):
        super().__init__()
        self.year = year
        self.title(f"Dynasty Prospect Viewer — {year} Draft Class")
        self.configure(bg=BG)
        self.state("zoomed")   # maximise on Windows

        # Load data
        try:
            self.df = _load_scores(year)
        except FileNotFoundError as e:
            messagebox.showerror("Data Not Found", str(e))
            self.destroy()
            return

        self.train_dists = _load_training_distributions()
        self._sort_col = "zap_score"
        self._sort_asc = False
        self._active_pos = "All"
        self._filter_risk = "All"
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", self._on_search)

        self._build_ui()
        self._refresh_table()

    # ── Layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        self._build_topbar()

        # Main pane: left = controls + table, right = overview chart
        main = tk.Frame(self, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        left = tk.Frame(main, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = tk.Frame(main, bg=PANEL, bd=1, relief=tk.FLAT)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(8, 0))
        right.config(width=420)
        right.pack_propagate(False)

        self._build_controls(left)
        self._build_table(left)
        self._build_overview_panel(right)

    def _build_topbar(self):
        bar = tk.Frame(self, bg=ACCENT, height=52)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        tk.Label(
            bar, text=f"  Dynasty Prospect Model  ·  {self.year} Draft Class",
            font=("Helvetica", 16, "bold"), bg=ACCENT, fg="white",
        ).pack(side=tk.LEFT, padx=12, pady=8)

        # ZAP summary badges
        for pos in ("WR", "RB", "TE"):
            sub = self.df[self.df["position"] == pos]
            top = sub.nlargest(1, "zap_score")
            if not top.empty:
                r = top.iloc[0]
                lbl = (f"  {pos}#1: {r['player_name'].split()[-1]}"
                       f" ZAP={r['zap_score']:.0f}  ")
                tk.Label(
                    bar, text=lbl, font=("Helvetica", 10, "bold"),
                    bg=POS_COLORS[pos], fg="white",
                ).pack(side=tk.LEFT, padx=4, pady=10)

        # Search
        tk.Label(bar, text="Search:", bg=ACCENT, fg="white",
                 font=("Helvetica", 10)).pack(side=tk.RIGHT, padx=(0, 4))
        search_entry = tk.Entry(
            bar, textvariable=self._search_var,
            font=("Helvetica", 10), bg=PANEL, fg=TEXT,
            insertbackground=TEXT, relief=tk.FLAT, width=20,
        )
        search_entry.pack(side=tk.RIGHT, padx=4, pady=12, ipady=3)

    def _build_controls(self, parent):
        ctrl = tk.Frame(parent, bg=BG, pady=6)
        ctrl.pack(fill=tk.X)

        # Position filter
        tk.Label(ctrl, text="Position:", bg=BG, fg=SUBTEXT,
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(6, 2))
        for pos in ("All", "WR", "RB", "TE"):
            color = POS_COLORS.get(pos, ACCENT)
            btn = tk.Button(
                ctrl, text=pos,
                font=("Helvetica", 10, "bold"),
                bg=color if pos != "All" else ACCENT,
                fg="white", relief=tk.FLAT, padx=14, pady=4,
                cursor="hand2",
                command=lambda p=pos: self._filter_pos(p),
            )
            btn.pack(side=tk.LEFT, padx=3)

        # Risk filter
        tk.Label(ctrl, text="   Risk:", bg=BG, fg=SUBTEXT,
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(12, 2))
        for risk, color in (("All", SUBTEXT), ("High Risk", HIGH_RISK),
                             ("Neutral", NEUTRAL), ("Low Risk", LOW_RISK)):
            btn = tk.Button(
                ctrl, text=risk, font=("Helvetica", 9),
                bg=color, fg="white", relief=tk.FLAT, padx=10, pady=4,
                cursor="hand2",
                command=lambda r=risk: self._filter_risk_fn(r),
            )
            btn.pack(side=tk.LEFT, padx=2)

        # Stats strip
        self._stats_label = tk.Label(
            ctrl, text="", bg=BG, fg=SUBTEXT, font=("Helvetica", 9),
        )
        self._stats_label.pack(side=tk.RIGHT, padx=12)

    def _build_table(self, parent):
        cols = ("pos_rank", "player_name", "position", "zap_score",
                "phase1_zap", "capital_delta", "risk",
                "projected_b2s", "best_dominator", "best_rec_rate",
                "college_fantasy_ppg", "weight_lbs",
                "speed_score", "forty_time", "consensus_rank")

        self._table_cols = cols

        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "Prospects.Treeview",
            background=PANEL, foreground=TEXT, fieldbackground=PANEL,
            rowheight=26, font=("Helvetica", 10),
        )
        style.configure(
            "Prospects.Treeview.Heading",
            background=ACCENT, foreground="white",
            font=("Helvetica", 10, "bold"), relief=tk.FLAT,
        )
        style.map("Prospects.Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "white")])

        self._tree = ttk.Treeview(
            frame, columns=cols, show="headings",
            style="Prospects.Treeview", selectmode="browse",
        )

        widths = {
            "pos_rank": 46, "player_name": 180, "position": 46,
            "zap_score": 52, "phase1_zap": 52, "capital_delta": 56,
            "risk": 84, "projected_b2s": 72,
            "best_dominator": 72, "best_rec_rate": 72,
            "college_fantasy_ppg": 68, "weight_lbs": 58,
            "speed_score": 72, "forty_time": 52, "consensus_rank": 54,
        }

        for c in cols:
            label = COL_LABELS.get(c, c)
            self._tree.heading(
                c, text=label,
                command=lambda col=c: self._sort_by(col),
            )
            self._tree.column(c, width=widths.get(c, 70),
                              anchor=tk.CENTER, stretch=False)
        self._tree.column("player_name", anchor=tk.W, stretch=True)

        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL,
                            command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)

        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self._tree.bind("<Double-1>", self._on_double_click)
        self._tree.bind("<Return>", self._on_double_click)

        # Row tags for risk colouring
        self._tree.tag_configure("High Risk", foreground=HIGH_RISK)
        self._tree.tag_configure("Low Risk",  foreground=LOW_RISK)
        self._tree.tag_configure("Neutral",   foreground=TEXT)
        self._tree.tag_configure("alt", background="#262b35")

    def _build_overview_panel(self, parent):
        tk.Label(
            parent, text="Overview", font=("Helvetica", 12, "bold"),
            bg=PANEL, fg=TEXT,
        ).pack(anchor=tk.W, padx=10, pady=(10, 4))

        self._overview_fig = Figure(figsize=(4.2, 8), dpi=96,
                                    facecolor=PANEL)
        self._overview_canvas = FigureCanvasTkAgg(
            self._overview_fig, master=parent
        )
        self._overview_canvas.get_tk_widget().pack(
            fill=tk.BOTH, expand=True, padx=6, pady=6
        )
        self._draw_overview()

    # ── Data / filter helpers ───────────────────────────────────────────────

    def _filtered_df(self) -> pd.DataFrame:
        df = self.df.copy()
        if self._active_pos != "All":
            df = df[df["position"] == self._active_pos]
        if self._filter_risk != "All":
            df = df[df["risk"] == self._filter_risk]
        q = self._search_var.get().strip().lower()
        if q:
            df = df[df["player_name"].str.lower().str.contains(q, na=False)]
        return df.sort_values(
            self._sort_col, ascending=self._sort_asc, na_position="last"
        )

    def _filter_pos(self, pos: str):
        self._active_pos = pos
        self._refresh_table()
        self._draw_overview()

    def _filter_risk_fn(self, risk: str):
        self._filter_risk = risk
        self._refresh_table()

    def _sort_by(self, col: str):
        if self._sort_col == col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            self._sort_asc = False
        self._refresh_table()

    def _on_search(self, *_):
        self._refresh_table()

    # ── Table rendering ─────────────────────────────────────────────────────

    def _refresh_table(self):
        self._tree.delete(*self._tree.get_children())
        df = self._filtered_df()

        for i, (_, row) in enumerate(df.iterrows()):
            risk = row.get("risk", "Neutral") or "Neutral"
            values = []
            for c in self._table_cols:
                v = row.get(c, "")
                if pd.isna(v):
                    values.append("—")
                elif c in ("zap_score", "phase1_zap", "projected_b2s"):
                    values.append(f"{float(v):.1f}")
                elif c == "capital_delta" and pd.notna(v):
                    values.append(f"{float(v):+.1f}")
                elif c == "best_dominator" and pd.notna(v):
                    values.append(f"{float(v):.3f}")
                elif c == "best_rec_rate" and pd.notna(v):
                    values.append(f"{float(v):.3f}")
                elif c in ("college_fantasy_ppg", "speed_score") and pd.notna(v):
                    values.append(f"{float(v):.1f}")
                elif c == "forty_time" and pd.notna(v):
                    values.append(f"{float(v):.2f}")
                elif c in ("pos_rank", "consensus_rank",
                           "weight_lbs") and pd.notna(v):
                    try:
                        values.append(str(int(float(v))))
                    except (ValueError, TypeError):
                        values.append(str(v))
                else:
                    values.append(str(v))

            tag = risk if risk in ("High Risk", "Low Risk", "Neutral") else ""
            alt = "alt" if (i % 2 == 1 and not tag) else tag
            self._tree.insert("", tk.END, iid=str(row.get("player_name", i)),
                              values=values, tags=(alt,))

        n = len(df)
        hr = (df["risk"] == "High Risk").sum()
        lr = (df["risk"] == "Low Risk").sum()
        self._stats_label.config(
            text=f"{n} prospects  |  {hr} High Risk  |  {lr} Low Risk"
        )

    # ── Overview chart ──────────────────────────────────────────────────────

    def _draw_overview(self):
        fig = self._overview_fig
        fig.clear()

        df = self._filtered_df()
        positions = (
            [self._active_pos] if self._active_pos != "All"
            else ["WR", "RB", "TE"]
        )

        n_pos = len(positions)
        axes = fig.subplots(n_pos, 1)
        if n_pos == 1:
            axes = [axes]

        for ax, pos in zip(axes, positions):
            sub = df[df["position"] == pos].head(20)
            if sub.empty:
                ax.set_visible(False)
                continue

            players = [n.split()[-1] for n in sub["player_name"]]
            zap = sub["zap_score"].values
            ph1 = sub["phase1_zap"].values if "phase1_zap" in sub else None

            y = np.arange(len(players))
            color = POS_COLORS.get(pos, ACCENT)

            ax.set_facecolor(PANEL)
            ax.barh(y, zap, height=0.55, color=color, alpha=0.85,
                    label="ZAP")
            if ph1 is not None and not np.all(np.isnan(ph1.astype(float))):
                ax.scatter(ph1, y, color=GOLD, s=30, zorder=5,
                           label="Phase I", marker="D")

            # Risk shading
            for i, (_, row) in enumerate(sub.iterrows()):
                risk = row.get("risk", "Neutral")
                if risk == "High Risk":
                    ax.axhspan(i - 0.35, i + 0.35,
                               color=HIGH_RISK, alpha=0.08)
                elif risk == "Low Risk":
                    ax.axhspan(i - 0.35, i + 0.35,
                               color=LOW_RISK, alpha=0.08)

            ax.set_yticks(y)
            ax.set_yticklabels(players, fontsize=7.5, color=TEXT)
            ax.set_xlim(0, 105)
            ax.set_xlabel("Score", fontsize=8, color=SUBTEXT)
            ax.tick_params(colors=SUBTEXT, labelsize=7.5)
            for spine in ax.spines.values():
                spine.set_edgecolor(BG)
            ax.set_title(f"{pos} Top {len(sub)}",
                         color=color, fontsize=9, fontweight="bold",
                         loc="left", pad=4)
            ax.axvline(50, color=SUBTEXT, linewidth=0.6, linestyle="--")

            if pos == positions[0]:
                ax.legend(fontsize=7, facecolor=PANEL, labelcolor=TEXT,
                          framealpha=0.7, loc="lower right")

        fig.tight_layout(pad=1.2)
        self._overview_canvas.draw()

    # ── Player detail window ────────────────────────────────────────────────

    def _on_double_click(self, event):
        sel = self._tree.selection()
        if not sel:
            return
        name = sel[0]
        rows = self.df[self.df["player_name"] == name]
        if rows.empty:
            return
        PlayerDetail(self, rows.iloc[0], self.train_dists)

    # ── Filter + scroll helpers ─────────────────────────────────────────────

    def _on_search(self, *_):
        self._refresh_table()


# ═══════════════════════════════════════════════════════════════════════════
# Player Detail Window
# ═══════════════════════════════════════════════════════════════════════════

class PlayerDetail(tk.Toplevel):
    def __init__(self, parent, row: pd.Series, train_dists: dict):
        super().__init__(parent)
        self.row = row
        self.train_dists = train_dists
        pos = str(row.get("position", "WR"))

        self.title(f"{row['player_name']}  —  {pos}")
        self.configure(bg=BG)
        self.geometry("1020x720")
        self.resizable(True, True)

        self._build(pos)

    def _build(self, pos: str):
        # ── Header strip ──
        hdr = tk.Frame(self, bg=POS_COLORS.get(pos, ACCENT), height=62)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        name = self.row.get("player_name", "Unknown")
        team = self.row.get("best_team", "")
        yr   = self.row.get("best_season_year", "")
        tk.Label(
            hdr, text=f"  {name}",
            font=("Helvetica", 20, "bold"),
            bg=POS_COLORS.get(pos, ACCENT), fg="white",
        ).pack(side=tk.LEFT, padx=8, pady=8)
        tk.Label(
            hdr, text=f"{pos}  ·  {team}  ·  Best Season: {yr}",
            font=("Helvetica", 11),
            bg=POS_COLORS.get(pos, ACCENT), fg="white",
        ).pack(side=tk.LEFT, padx=4)

        # ZAP/Ph1/Delta badges
        for label, val, color in self._badge_data():
            frm = tk.Frame(hdr, bg=color, padx=10, pady=4)
            frm.pack(side=tk.RIGHT, padx=6, pady=10)
            tk.Label(frm, text=label, bg=color, fg="white",
                     font=("Helvetica", 8)).pack()
            tk.Label(frm, text=val, bg=color, fg="white",
                     font=("Helvetica", 14, "bold")).pack()

        # ── Main body: left stats, right charts ──
        body = tk.Frame(self, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = tk.Frame(body, bg=BG, width=340)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)

        right = tk.Frame(body, bg=BG)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_stat_panels(left, pos)
        self._build_charts(right, pos)

    def _badge_data(self):
        zap  = self.row.get("zap_score",     np.nan)
        ph1  = self.row.get("phase1_zap",    np.nan)
        delt = self.row.get("capital_delta", np.nan)
        risk = self.row.get("risk", "N/A")
        risk_col = RISK_COLORS.get(risk, NEUTRAL)

        badges = []
        if pd.notna(zap):
            badges.append(("ZAP Score", f"{zap:.0f}", ACCENT))
        if pd.notna(ph1):
            badges.append(("Phase I", f"{ph1:.0f}", "#8e44ad"))
        if pd.notna(delt):
            sign = "+" if delt >= 0 else ""
            badges.append((f"Capital Δ", f"{sign}{delt:.0f}", risk_col))
        badges.append(("Risk", str(risk), risk_col))
        return reversed(badges)

    def _build_stat_panels(self, parent, pos: str):
        def section(title):
            tk.Label(parent, text=title, bg=BG, fg=ACCENT,
                     font=("Helvetica", 10, "bold")).pack(
                anchor=tk.W, pady=(10, 2))

        def row_stat(label, val, pct_series=None):
            frm = tk.Frame(parent, bg=PANEL)
            frm.pack(fill=tk.X, pady=1)
            tk.Label(frm, text=f"  {label}",
                     bg=PANEL, fg=SUBTEXT,
                     font=("Helvetica", 9), width=22, anchor=tk.W
                     ).pack(side=tk.LEFT)
            disp = "—" if (val is None or (isinstance(val, float)
                                           and np.isnan(val))) else str(val)
            tk.Label(frm, text=disp, bg=PANEL, fg=TEXT,
                     font=("Helvetica", 9, "bold"), width=10, anchor=tk.E
                     ).pack(side=tk.LEFT)
            # Mini percentile bar
            if pct_series is not None and pd.notna(val):
                try:
                    pct = _percentile_of(float(val), pct_series)
                    bar_f = tk.Frame(frm, bg=PANEL)
                    bar_f.pack(side=tk.LEFT, padx=6, pady=2)
                    canvas = tk.Canvas(bar_f, width=70, height=10,
                                       bg=BG, highlightthickness=0)
                    canvas.pack()
                    fill_w = int(pct * 0.70)
                    bar_col = (LOW_RISK if pct >= 70
                               else GOLD if pct >= 40 else HIGH_RISK)
                    canvas.create_rectangle(0, 2, fill_w, 8,
                                            fill=bar_col, outline="")
                    canvas.create_text(74, 5, text=f"{pct:.0f}%",
                                       fill=SUBTEXT, font=("Helvetica", 7),
                                       anchor=tk.W)
                except (TypeError, ValueError):
                    pass

        r  = self.row
        td = self.train_dists.get(pos)

        def get(col):
            v = r.get(col, np.nan)
            return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v

        def fmt(v, decimals=2):
            if v is None:
                return None
            try:
                return f"{float(v):.{decimals}f}"
            except (ValueError, TypeError):
                return str(v)

        def ts(col):
            return td[col] if td is not None and col in td.columns else None

        section("Model Scores")
        row_stat("ZAP Score",     fmt(get("zap_score"), 1),    ts("b2s_score"))
        row_stat("Phase I ZAP",   fmt(get("phase1_zap"), 1))
        row_stat("Capital Delta",
                 f"{get('capital_delta'):+.1f}" if get("capital_delta") is not None else None)
        row_stat("Risk",          str(r.get("risk", "N/A")))
        row_stat("Proj B2S",      fmt(get("projected_b2s"), 2))
        row_stat("Proj Pick",     fmt(get("overall_pick"), 0))
        row_stat("Capital Score", fmt(get("draft_capital_score"), 1))

        section("College Production")
        row_stat("Breakout Score",  fmt(get("best_breakout_score"), 3), ts("best_breakout_score"))
        row_stat("Dominator",       fmt(get("best_dominator"), 3),       ts("best_dominator"))
        row_stat("Rec Rate",        fmt(get("best_rec_rate"), 4),        ts("best_rec_rate"))
        row_stat("CFP PPG",         fmt(get("college_fantasy_ppg"), 1),  ts("college_fantasy_ppg"))
        row_stat("Best Age",        fmt(get("best_age"), 1))
        row_stat("Early Declare",   str(int(get("early_declare") or 0)))
        row_stat("Power4 Conf",     str(int(get("power4_conf") or 0)) if get("power4_conf") is not None else "—")

        section("PFF Metrics")
        row_stat("YPRR",            fmt(get("best_yprr"), 2),            ts("best_yprr"))
        row_stat("Rec Grade",       fmt(get("best_receiving_grade"), 1), ts("best_receiving_grade"))
        row_stat("Routes/Game",     fmt(get("best_routes_per_game"), 1), ts("best_routes_per_game"))
        row_stat("Man YPRR",        fmt(get("best_man_yprr"), 2),        ts("best_man_yprr"))
        row_stat("Zone YPRR",       fmt(get("best_zone_yprr"), 2),       ts("best_zone_yprr"))
        row_stat("Man-Zone Delta",  fmt(get("best_man_zone_delta"), 2),  ts("best_man_zone_delta"))
        row_stat("Slot YPRR",       fmt(get("best_slot_yprr"), 2),       ts("best_slot_yprr"))
        row_stat("Deep YPRR",       fmt(get("best_deep_yprr"), 2),       ts("best_deep_yprr"))
        row_stat("Drop Rate",       fmt(get("best_drop_rate"), 1),       ts("best_drop_rate"))

        section("Combine / Physical")
        row_stat("Weight (lbs)",  fmt(get("weight_lbs"), 0),  ts("weight_lbs"))
        row_stat("Speed Score",   fmt(get("speed_score"), 1), ts("speed_score"))
        row_stat("40yd Dash",     fmt(get("forty_time"), 2),  ts("forty_time"))
        row_stat("Broad Jump",    fmt(get("broad_jump"), 0),  ts("broad_jump"))
        row_stat("Vert Jump",     fmt(get("vertical_jump"), 1), ts("vertical_jump"))

        section("Recruiting")
        row_stat("Recruit Rating", fmt(get("recruit_rating"), 4), ts("recruit_rating"))
        row_stat("Stars",          fmt(get("recruit_stars"), 0))
        row_stat("Natl Rank",      fmt(get("recruit_rank_national"), 0))

    def _build_charts(self, parent, pos: str):
        fig = Figure(figsize=(7, 5.5), dpi=96, facecolor=BG)

        # ── Radar chart ──────────────────────────────────────────
        ax_radar = fig.add_subplot(121, projection="polar")
        td = self.train_dists.get(pos)
        labels, vals = _build_radar_values(self.row, pos, td)

        if labels:
            N = len(labels)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]
            vals_plot = vals + vals[:1]

            ax_radar.set_facecolor(PANEL)
            ax_radar.plot(angles, vals_plot,
                          color=POS_COLORS.get(pos, ACCENT), linewidth=2)
            ax_radar.fill(angles, vals_plot,
                          color=POS_COLORS.get(pos, ACCENT), alpha=0.25)
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(labels, size=7.5, color=TEXT)
            ax_radar.set_yticks([0.25, 0.50, 0.75, 1.0])
            ax_radar.set_yticklabels(["25", "50", "75", "100"],
                                     size=6, color=SUBTEXT)
            ax_radar.set_ylim(0, 1)
            ax_radar.tick_params(colors=SUBTEXT)
            ax_radar.grid(color=SUBTEXT, alpha=0.3)
            ax_radar.set_title("Percentile Profile\nvs Training Set",
                               color=TEXT, size=9, pad=14)
            for spine in ax_radar.spines.values():
                spine.set_edgecolor(BG)
            ax_radar.set_facecolor(PANEL)

        # ── ZAP vs Phase I gauge ─────────────────────────────────
        ax_gauge = fig.add_subplot(122)
        ax_gauge.set_facecolor(PANEL)
        ax_gauge.set_xlim(-0.6, 0.6)
        ax_gauge.set_ylim(0, 110)
        ax_gauge.axis("off")

        zap  = self.row.get("zap_score", np.nan)
        ph1  = self.row.get("phase1_zap", np.nan)
        delt = self.row.get("capital_delta", np.nan)
        risk = self.row.get("risk", "N/A")
        risk_col = RISK_COLORS.get(risk, NEUTRAL)

        def _gauge_bar(ax, x, y_bot, height, color, label, value):
            ax.bar(x, height, bottom=y_bot, width=0.35, color=color,
                   alpha=0.85)
            ax.text(x, y_bot + height + 2.5,
                    f"{value:.0f}" if pd.notna(value) else "—",
                    ha="center", va="bottom", color=TEXT,
                    fontsize=16, fontweight="bold")
            ax.text(x, y_bot - 5, label, ha="center", va="top",
                    color=SUBTEXT, fontsize=9)

        if pd.notna(zap):
            _gauge_bar(ax_gauge, -0.2, 0, float(zap),
                       POS_COLORS.get(pos, ACCENT), "ZAP", zap)
        if pd.notna(ph1):
            _gauge_bar(ax_gauge, 0.2, 0, float(ph1),
                       "#8e44ad", "Phase I", ph1)

        if pd.notna(delt):
            sign = "+" if delt >= 0 else ""
            ax_gauge.text(0, 56,
                          f"Δ Capital\n{sign}{delt:.0f}",
                          ha="center", va="center", color=risk_col,
                          fontsize=13, fontweight="bold")
            ax_gauge.text(0, 44, risk, ha="center", va="center",
                          color=risk_col, fontsize=10)

        ax_gauge.axhline(50, color=SUBTEXT, linewidth=0.7,
                         linestyle="--", alpha=0.5)
        ax_gauge.text(0.55, 51, "50th pct", color=SUBTEXT,
                      fontsize=7, va="bottom")
        ax_gauge.set_title("ZAP vs Phase I", color=TEXT, size=9, pad=6)

        fig.tight_layout(pad=1.5)

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Dynasty Prospect Viewer GUI")
    parser.add_argument("--year", type=int, default=2026,
                        help="Draft year to display (default: 2026)")
    args = parser.parse_args()

    app = ProspectViewer(year=args.year)
    app.mainloop()


if __name__ == "__main__":
    main()
