"""
JJ 2026 Guide Feature Experiments
Tests new metrics from JJ's 2026 guide against our training data via LOYO-CV.
"""
import os, sys, json, warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

BASE = r"C:\Users\Ozimek\Documents\Claude\FF\dynasty-prospect-model"
TARGET = "b2s_score"

def loyo_cv(df, features, target=TARGET):
    df2 = df.dropna(subset=features + [target, "draft_year"]).copy()
    years = sorted(df2["draft_year"].unique())
    preds, actuals = [], []
    for yr in years:
        train = df2[df2["draft_year"] != yr]
        test  = df2[df2["draft_year"] == yr]
        if len(test) < 3:
            continue
        X_tr = train[features].values
        y_tr = train[target].values
        X_te = test[features].values
        y_te = test[target].values
        pipe = Pipeline([("sc", StandardScaler()),
                         ("ridge", RidgeCV(alphas=np.logspace(-1, 4, 60)))])
        pipe.fit(X_tr, y_tr)
        yhat = pipe.predict(X_te)
        preds.extend(yhat.tolist())
        actuals.extend(y_te.tolist())
    if not preds:
        return np.nan, np.nan
    overall = r2_score(actuals, preds)
    rho, _ = spearmanr(actuals, preds)
    return round(overall, 4), round(rho, 4)

def current_loyo(pos):
    with open(os.path.join(BASE, "models", "metadata.json")) as f:
        meta = json.load(f)
    return meta[pos]["ridge_loyo_r2"], meta[pos]["selected_features"]

def run_experiment(pos, df, candidate_features, label, base_r2, base_features):
    all_feats = list(dict.fromkeys(base_features + candidate_features))
    avail = [f for f in all_feats if f in df.columns]
    r2, rho = loyo_cv(df, avail)
    delta = round(r2 - base_r2, 4)
    sign = "+" if delta >= 0 else ""
    print(f"  {label}: LOYO R2={r2:.4f}  delta={sign}{delta:.4f}  Spearman={rho:.4f}")
    return r2, delta

# ── load data ──────────────────────────────────────────────────────────────
wr = pd.read_csv(os.path.join(BASE, "data", "training_WR.csv"))
rb = pd.read_csv(os.path.join(BASE, "data", "training_RB.csv"))
te = pd.read_csv(os.path.join(BASE, "data", "training_TE.csv"))

# ── compute derived features ───────────────────────────────────────────────
for d in [wr, rb, te]:
    d["log_draft_capital"] = np.log(d["draft_capital_score"].clip(lower=0.01) + 1)
    # height-adjusted speed score: SS * (height/74)^1.5
    d["height_adj_speed_score"] = np.where(
        d["speed_score"].notna() & d["height_inches"].notna(),
        d["speed_score"] * (d["height_inches"] / 74.0) ** 1.5,
        np.nan
    )
    d["height_ss_x_capital"] = d["height_adj_speed_score"] * d["log_draft_capital"]
    d["teammate_score_x_capital"] = d["teammate_score"] * d["log_draft_capital"]
    # JJ-style capital curve: log(1000/(pick+1))
    d["log_pick_capital"] = np.log(1000.0 / (d["overall_pick"] + 1))

# WR: age+SOS adjusted fantasy PPG
def age_mult(age):
    if age <= 19: return 1.25
    if age <= 20: return 1.12
    if age <= 21: return 1.0
    if age <= 22: return 0.88
    return 0.76

wr["adj_fantasy_ppg"] = wr.apply(
    lambda r: r["college_fantasy_ppg"] * age_mult(r["best_age"]) *
              max(0.70, min(1.30, 1.0 + (r["best_sp_plus"] - 5.0) / 100.0))
    if pd.notna(r["college_fantasy_ppg"]) else np.nan, axis=1
)

# RB: total yards per team play proxy
rb["rush_rate"] = (rb["best_usage_rush"] /
                   (rb["best_usage_rush"] + rb["best_usage_pass"]).replace(0, np.nan))
rb["total_plays_adj_yards_rate"] = rb["best_total_yards_rate"] * (1 - rb["rush_rate"].fillna(0.5))

SEP = "=" * 68

# ── WR ─────────────────────────────────────────────────────────────────────
print(SEP)
print("WR EXPERIMENTS")
print(SEP)
base_r2_wr, base_feats_wr = current_loyo("WR")
print(f"  Baseline LOYO R2: {base_r2_wr:.4f}")
print(f"  Current features: {base_feats_wr}\n")
results = {}
wr_exps = {
    "E1: +teammate_score": ["teammate_score"],
    "E2: +teammate_score_x_capital": ["teammate_score_x_capital"],
    "E3: +adj_fantasy_ppg (age+SOS)": ["adj_fantasy_ppg"],
    "E4: +log_pick_capital (JJ curve)": ["log_pick_capital"],
    "E5: +height_adj_speed_score": ["height_adj_speed_score"],
    "E6: all new WR features": ["teammate_score","teammate_score_x_capital",
                                 "adj_fantasy_ppg","log_pick_capital","height_adj_speed_score"],
}
for lbl, feats in wr_exps.items():
    r2, d = run_experiment("WR", wr, feats, lbl, base_r2_wr, base_feats_wr)
    results["WR_" + lbl] = {"r2": r2, "delta": d}

# ── RB ─────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("RB EXPERIMENTS")
print(SEP)
base_r2_rb, base_feats_rb = current_loyo("RB")
print(f"  Baseline LOYO R2: {base_r2_rb:.4f}")
print(f"  Current features: {base_feats_rb}\n")
rb_exps = {
    "E1: +teammate_score": ["teammate_score"],
    "E2: +teammate_score_x_capital": ["teammate_score_x_capital"],
    "E3: +speed_score": ["speed_score"],
    "E4: +height_adj_speed_score": ["height_adj_speed_score"],
    "E5: +total_plays_adj_yards_rate": ["total_plays_adj_yards_rate"],
    "E6: +log_pick_capital": ["log_pick_capital"],
    "E7: all new RB features": ["teammate_score","teammate_score_x_capital",
                                 "speed_score","height_adj_speed_score",
                                 "total_plays_adj_yards_rate","log_pick_capital"],
}
for lbl, feats in rb_exps.items():
    r2, d = run_experiment("RB", rb, feats, lbl, base_r2_rb, base_feats_rb)
    results["RB_" + lbl] = {"r2": r2, "delta": d}

# ── TE ─────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("TE EXPERIMENTS")
print(SEP)
base_r2_te, base_feats_te = current_loyo("TE")
print(f"  Baseline LOYO R2: {base_r2_te:.4f}")
print(f"  Current features: {base_feats_te}\n")
te_exps = {
    "E1: +teammate_score": ["teammate_score"],
    "E2: +teammate_score_x_capital": ["teammate_score_x_capital"],
    "E3: +height_adj_speed_score": ["height_adj_speed_score"],
    "E4: +height_ss_x_capital": ["height_ss_x_capital"],
    "E5: +log_pick_capital": ["log_pick_capital"],
    "E6: +speed_score (raw)": ["speed_score"],
    "E7: +height_inches alone": ["height_inches"],
    "E8: +speed+height (JJ TE proxy)": ["speed_score","height_inches","height_adj_speed_score"],
    "E9: height_adj_ss + x_cap": ["height_adj_speed_score","height_ss_x_capital"],
    "E10: all new TE features": ["teammate_score","teammate_score_x_capital",
                                  "height_adj_speed_score","height_ss_x_capital",
                                  "log_pick_capital","speed_score","height_inches"],
}
for lbl, feats in te_exps.items():
    r2, d = run_experiment("TE", te, feats, lbl, base_r2_te, base_feats_te)
    results["TE_" + lbl] = {"r2": r2, "delta": d}

# ── PHASE C — JJ 2026 GUIDE GAP ANALYSIS ───────────────────────────────────
print()
print(SEP)
print("PHASE C — JJ 2026 GUIDE GAP ANALYSIS")
print(SEP)

# ── Phase C derived features ────────────────────────────────────────────────

# TE: Inverse athletic interaction (JJ: lower athleticism + strong capital = better signal)
te["inv_height_adj_ss_x_capital"] = (
    te["log_draft_capital"] / te["height_adj_speed_score"].clip(lower=1)
)
# Normalized low-athleticism interaction: peaks when capital is high AND athleticism is low
_ss_min = te["height_adj_speed_score"].quantile(0.05)
_ss_max = te["height_adj_speed_score"].quantile(0.95)
te["height_adj_ss_norm"] = (
    (te["height_adj_speed_score"] - _ss_min) / (_ss_max - _ss_min)
).clip(0, 1)
te["capital_x_low_athleticism"] = te["log_draft_capital"] * (1 - te["height_adj_ss_norm"])
# Negative athleticism term — Ridge will find the right sign direction
te["neg_height_adj_ss"] = -te["height_adj_speed_score"]

# RB: capital swap interaction terms (log_pick_capital replaces log_draft_capital)
rb["pick_capital_x_age"] = rb["log_pick_capital"] * rb["best_age"]
rb["total_yards_rate_x_pick_capital"] = rb["best_total_yards_rate"] * rb["log_pick_capital"]

# ── WR Phase C ──────────────────────────────────────────────────────────────
print()
print("WR PHASE C")
print("-" * 50)
wr_phase_c = {
    "C1: +best_usage_rush (gadget)":    ["best_usage_rush"],
    "C2: +career_rush_yards":           ["career_rush_yards"],
    "C3: +career_rush_attempts":        ["career_rush_attempts"],
    "C4: +weight_lbs":                  ["weight_lbs"],
    "C5: +height_inches":               ["height_inches"],
    "C6: +weight+height":               ["weight_lbs", "height_inches"],
    "C7: +rush+size combo":             ["best_usage_rush", "career_rush_yards",
                                         "weight_lbs", "height_inches"],
}
for lbl, feats in wr_phase_c.items():
    r2, d = run_experiment("WR", wr, feats, lbl, base_r2_wr, base_feats_wr)
    results["WR_" + lbl] = {"r2": r2, "delta": d}

# Spearman check for rush involvement in WR subgroup (sparse metric sanity check)
_wr_rush = wr[wr["career_rush_yards"].fillna(0) > 50].copy()
if len(_wr_rush) >= 20:
    _rho, _pval = spearmanr(_wr_rush["career_rush_yards"].fillna(0),
                             _wr_rush["b2s_score"].fillna(0))
    print(f"\n  WR rush subset (career_rush_yards > 50, n={len(_wr_rush)}): "
          f"Spearman rho={_rho:.3f}  p={_pval:.3f}")

# ── RB Phase C ──────────────────────────────────────────────────────────────
print()
print("RB PHASE C")
print("-" * 50)

# C3: capital swap — log_pick_capital replaces log_draft_capital across all terms
_capital_swap_map = {
    "log_draft_capital": "log_pick_capital",
    "capital_x_age": "pick_capital_x_age",
    "total_yards_rate_x_capital": "total_yards_rate_x_pick_capital",
}
base_feats_rb_swapped = [_capital_swap_map.get(f, f) for f in base_feats_rb]
_avail_swap = [f for f in base_feats_rb_swapped if f in rb.columns]
_r2_swap, _rho_swap = loyo_cv(rb, _avail_swap)
_d_swap = round(_r2_swap - base_r2_rb, 4)
_sign = "+" if _d_swap >= 0 else ""
print(f"  C3: capital swap (log_pick replaces log_draft): LOYO R2={_r2_swap:.4f}  "
      f"delta={_sign}{_d_swap:.4f}  Spearman={_rho_swap:.4f}")
results["RB_C3: capital swap"] = {"r2": _r2_swap, "delta": _d_swap}

rb_phase_c = {
    "C1: +weight_lbs":                  ["weight_lbs"],
    "C2: +best_breakout_score":         ["best_breakout_score"],
    "C4: +teammate_score (confirm)":    ["teammate_score"],
    "C5: +teammate_score_x_capital":    ["teammate_score_x_capital"],
    "C6: all new RB signals":           ["weight_lbs", "best_breakout_score",
                                         "teammate_score", "teammate_score_x_capital"],
}
for lbl, feats in rb_phase_c.items():
    r2, d = run_experiment("RB", rb, feats, lbl, base_r2_rb, base_feats_rb)
    results["RB_" + lbl] = {"r2": r2, "delta": d}

# ── TE Phase C ──────────────────────────────────────────────────────────────
print()
print("TE PHASE C")
print("-" * 50)
te_phase_c = {
    "C1: +inv_height_adj_ss_x_capital":  ["inv_height_adj_ss_x_capital"],
    "C2: +capital_x_low_athleticism":    ["capital_x_low_athleticism"],
    "C3: +neg_height_adj_ss":            ["neg_height_adj_ss"],
    "C4: inv_ath + capital_x_low_ath":   ["inv_height_adj_ss_x_capital",
                                           "capital_x_low_athleticism"],
}
for lbl, feats in te_phase_c.items():
    r2, d = run_experiment("TE", te, feats, lbl, base_r2_te, base_feats_te)
    results["TE_" + lbl] = {"r2": r2, "delta": d}

# ── SUMMARY ────────────────────────────────────────────────────────────────
print()
print(SEP)
print("SUMMARY")
print(SEP)
print("Positive delta (>= +0.002):")
for k, v in sorted(results.items(), key=lambda x: -x[1]["delta"]):
    if v["delta"] >= 0.002:
        print(f"  {k}: +{v['delta']:.4f}  (R2={v['r2']:.4f})")
print()
print("Negative delta (confirmed no-signal):")
for k, v in sorted(results.items(), key=lambda x: x[1]["delta"]):
    if v["delta"] < -0.001:
        print(f"  {k}: {v['delta']:.4f}  (R2={v['r2']:.4f})")

# ── JJ vs Our comparison ───────────────────────────────────────────────────
print()
print(SEP)
print("JJ ZAP vs OUR ORBIT SCORES (2026 pre-draft)")
print(SEP)
comparisons = [
    ("Jordyn Tyson WR",   92.2, 85.7),
    ("Carnell Tate WR",   84.2, 92.4),
    ("Makai Lemon WR",    80.2, 93.3),
    ("KC Concepcion WR",  73.4, None),
    ("Omar Cooper WR",    66.3, None),
    ("Denzel Boston WR",  66.1, None),
    ("Chris Brazzell WR", 61.6, None),
    ("Germie Bernard WR", 60.4, None),
    ("Elijah Sarratt WR", 57.4, None),
    ("Zachariah Branch WR", 57.0, None),
    ("Jeremiyah Love RB", 93.9, 97.3),
    ("Emmett Johnson RB", 69.7, 83.0),
    ("Jadarian Price RB", 64.2, None),
    ("Mike Washington RB",61.0, None),
    ("Jonah Coleman RB",  58.7, 68.0),
    ("Nick Singleton RB", 51.0, None),
    ("Kaytron Allen RB",  41.0, None),
    ("Kenyon Sadiq TE",   97.8, 99.0),
    ("Eli Stowers TE",    79.2, 95.9),
    ("Max Klare TE",      56.2, 82.5),
    ("Michael Trigg TE",  48.8, None),
    ("Sam Roush TE",      44.7, None),
    ("Justin Joly TE",    40.8, None),
]
print(f"  {'Player':<25} {'JJ':>6} {'Ours':>6} {'Delta':>8}")
print("  " + "-" * 49)
for name, jj_z, our_z in comparisons:
    d_str = f"{our_z - jj_z:+.1f}" if our_z is not None else "  N/A"
    our_str = f"{our_z:.1f}" if our_z is not None else "  N/A"
    print(f"  {name:<25} {jj_z:>6.1f} {our_str:>6} {d_str:>8}")

print("\nDone.")
