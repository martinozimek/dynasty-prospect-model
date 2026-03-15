"""
Phase A Stress Tests — dynasty-prospect-model
==============================================
Runs all five Phase A diagnostics from RESEARCH_PLAN.md.
No model changes are made. Results are written to output/stress_test/.

Tests:
  A1  Bootstrap Lasso stability (1000 resamples, selection frequency per feature)
  A2  Feature knockout (one-at-a-time LOYO R² drop)
  A3  ZAP calibration curve (predicted decile vs actual B2S mean)
  A4  Label permutation test (shuffle B2S → expect LOYO R²≈0)
  A5  Capital decomposition (capital-only vs Phase I vs full model LOYO R²)

Usage:
    python scripts/stress_test.py --all
    python scripts/stress_test.py --position WR --test A1
    python scripts/stress_test.py --all --n-boot 200   # faster
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from analyze import engineer_features, load_training
from config import get_data_dir
from fit_model import (
    _CANDIDATE_FEATURES,
    _CANDIDATE_FEATURES_NOCAP,
    _RIDGE_ALPHA_RANGE,
    _lasso_select,
    _loyo_cv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_POSITIONS = ("WR", "RB", "TE")
TARGET = "b2s_score"

OUT_DIR = _ROOT / "output" / "stress_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ridge_pipe(alpha_range):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("mdl", RidgeCV(alphas=alpha_range, cv=5)),
    ])


def _loyo_r2(df, features, target=TARGET, alpha_range=None):
    """Quick LOYO R² for a feature set. No nested Lasso (for speed in knock-out loops)."""
    years = sorted(df[target].dropna().index)  # not used directly
    years = sorted(df["draft_year"].dropna().unique())
    all_actual, all_pred = [], []
    for yr in years:
        train = df[df["draft_year"] != yr]
        test  = df[df["draft_year"] == yr]
        tr = train[train[target].notna()]
        te = test[test[target].notna()]
        if len(tr) < 10 or len(te) < 2:
            continue
        used = [f for f in features if f in tr.columns]
        if not used:
            continue
        ar = alpha_range if alpha_range is not None else np.logspace(-1, 2, 30)
        pipe = _ridge_pipe(ar)
        pipe.fit(tr[used].values, tr[target].values)
        preds = np.clip(pipe.predict(te[used].values), 0, None)
        all_actual.extend(te[target].values.tolist())
        all_pred.extend(preds.tolist())
    if len(all_actual) < 5:
        return float("nan")
    return float(r2_score(all_actual, all_pred))


def _load_pos(pos, data_dir, start_year=2014, end_year=None):
    df_raw = load_training(pos, data_dir, start_year, end_year)
    return engineer_features(df_raw)


def _available_candidates(df, candidates):
    return [c for c in candidates
            if c in df.columns
            and df[c].notna().sum() >= 5
            and df[c].nunique(dropna=True) > 1]


def _outer_lasso_features(df, candidates, target=TARGET):
    """Run outer Lasso on available candidates, return selected list."""
    avail = _available_candidates(df, candidates)
    labeled = df[df[target].notna()]
    return _lasso_select(labeled[avail], labeled[target])


# ---------------------------------------------------------------------------
# A1 — Bootstrap Lasso stability
# ---------------------------------------------------------------------------

def test_a1_bootstrap(pos, df, candidates, n_boot=1000, seed=42, target=TARGET):
    log.info("A1 Bootstrap Lasso [%s] n_boot=%d ...", pos, n_boot)
    rng = np.random.default_rng(seed)
    avail = _available_candidates(df, candidates)
    labeled = df[df[target].notna()].reset_index(drop=True)
    n = len(labeled)

    counts = {f: 0 for f in avail}
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = labeled.iloc[idx]
        # require enough variance in resample
        boot_avail = [f for f in avail
                      if boot[f].notna().sum() >= 5
                      and boot[f].nunique(dropna=True) > 1]
        selected = _lasso_select(boot[boot_avail], boot[target], cv_folds=3)
        for f in selected:
            counts[f] = counts.get(f, 0) + 1
        if (i + 1) % 200 == 0:
            log.info("  A1 [%s] %d/%d", pos, i + 1, n_boot)

    results = sorted(
        [{"feature": f, "freq": counts[f] / n_boot, "count": counts[f]}
         for f in avail],
        key=lambda x: -x["freq"]
    )
    return results


# ---------------------------------------------------------------------------
# A2 — Feature knockout
# ---------------------------------------------------------------------------

def test_a2_knockout(pos, df, features, alpha_range, target=TARGET):
    log.info("A2 Knockout [%s] baseline features: %s", pos, features)
    labeled = df[df[target].notna()].copy()
    baseline = _loyo_r2(labeled, features, target, alpha_range)
    log.info("  A2 [%s] baseline LOYO R²=%.4f", pos, baseline)

    results = [{"feature": "BASELINE", "loyo_r2": baseline, "delta": 0.0}]
    for f in features:
        remaining = [x for x in features if x != f]
        if not remaining:
            continue
        r2 = _loyo_r2(labeled, remaining, target, alpha_range)
        delta = r2 - baseline
        log.info("  A2 [%s] remove %-40s → R²=%.4f  delta=%+.4f", pos, f, r2, delta)
        results.append({"feature": f, "loyo_r2": r2, "delta": delta})

    results.sort(key=lambda x: x["delta"])  # most damaging removal first
    return results


# ---------------------------------------------------------------------------
# A3 — ZAP calibration curve
# ---------------------------------------------------------------------------

def test_a3_calibration(pos, df, features, alpha_range, target=TARGET, n_bins=10):
    log.info("A3 Calibration [%s] ...", pos)
    labeled = df[df[target].notna()].copy()
    years = sorted(labeled["draft_year"].dropna().unique())
    records = []

    for yr in years:
        train = labeled[labeled["draft_year"] != yr]
        test  = labeled[labeled["draft_year"] == yr]
        if len(train) < 10 or len(test) < 2:
            continue
        used = [f for f in features if f in train.columns]
        pipe = _ridge_pipe(alpha_range)
        pipe.fit(train[used].values, train[target].values)
        train_preds = np.clip(pipe.predict(train[used].values), 0, None)
        test_preds  = np.clip(pipe.predict(test[used].values), 0, None)

        for pred, actual in zip(test_preds, test[target].values):
            zap = float(np.mean(train_preds <= pred) * 100)
            records.append({"zap": zap, "pred_b2s": pred, "actual_b2s": actual})

    df_cal = pd.DataFrame(records)
    if df_cal.empty:
        return []

    df_cal["zap_bin"] = pd.cut(df_cal["zap"], bins=n_bins, labels=False)
    results = []
    for b in range(n_bins):
        grp = df_cal[df_cal["zap_bin"] == b]
        if grp.empty:
            continue
        lo = b * (100 / n_bins)
        hi = (b + 1) * (100 / n_bins)
        results.append({
            "bin_label": f"{lo:.0f}–{hi:.0f}",
            "n": len(grp),
            "pred_b2s_mean": round(float(grp["pred_b2s"].mean()), 2),
            "actual_b2s_mean": round(float(grp["actual_b2s"].mean()), 2),
            "bias": round(float(grp["pred_b2s"].mean() - grp["actual_b2s"].mean()), 2),
        })

    # Overall Spearman rank correlation
    rho, p = spearmanr(df_cal["pred_b2s"], df_cal["actual_b2s"])
    top12_threshold = df_cal["actual_b2s"].quantile(0.75)  # approx top-25%
    df_cal["predicted_top12"] = df_cal["zap"] >= 75
    df_cal["actual_top12"]    = df_cal["actual_b2s"] >= top12_threshold
    top12_hit_rate = float((df_cal["predicted_top12"] & df_cal["actual_top12"]).sum() /
                           df_cal["predicted_top12"].sum()) if df_cal["predicted_top12"].sum() > 0 else float("nan")
    top12_precision_naive = float(df_cal["actual_top12"].mean())

    return {
        "bins": results,
        "spearman_rho": round(float(rho), 4),
        "spearman_p": round(float(p), 4),
        "top25pct_hit_rate_at_zap75": round(top12_hit_rate, 3),
        "top25pct_base_rate": round(top12_precision_naive, 3),
    }


# ---------------------------------------------------------------------------
# A4 — Label permutation test
# ---------------------------------------------------------------------------

def test_a4_permutation(pos, df, features, alpha_range, target=TARGET, n_perm=100, seed=42):
    log.info("A4 Permutation [%s] n_perm=%d ...", pos, n_perm)
    rng = np.random.default_rng(seed)
    labeled = df[df[target].notna()].copy()
    real_r2 = _loyo_r2(labeled, features, target, alpha_range)
    log.info("  A4 [%s] real LOYO R²=%.4f", pos, real_r2)

    perm_r2s = []
    for i in range(n_perm):
        shuffled = labeled.copy()
        shuffled[target] = rng.permutation(shuffled[target].values)
        r2 = _loyo_r2(shuffled, features, target, alpha_range)
        perm_r2s.append(r2)
        if (i + 1) % 25 == 0:
            log.info("  A4 [%s] %d/%d perm_mean=%.4f", pos, i + 1, n_perm, float(np.nanmean(perm_r2s)))

    perm_arr = np.array(perm_r2s)
    p_value = float(np.mean(perm_arr >= real_r2))
    return {
        "real_loyo_r2": round(real_r2, 4),
        "perm_mean": round(float(np.nanmean(perm_arr)), 4),
        "perm_std":  round(float(np.nanstd(perm_arr)), 4),
        "perm_max":  round(float(np.nanmax(perm_arr)), 4),
        "p_value":   round(p_value, 4),
        "signal_confirmed": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# A5 — Capital decomposition
# ---------------------------------------------------------------------------

def _capital_only_features(pos):
    """Features from the capital model that are capital/market signals only."""
    cap = [
        "log_draft_capital", "draft_capital_score", "draft_round", "overall_pick",
        "draft_tier", "is_top16_rb", "draft_premium",
        "consensus_rank", "position_rank",
        "capital_x_age", "capital_x_dominator",
        "rec_rate_x_capital", "breakout_score_x_capital",
        "total_yards_rate_x_capital", "college_fpg_x_capital",
        "yprr_x_capital", "deep_target_x_capital", "deep_yprr_x_capital",
        "man_delta_x_capital", "slot_rate_x_capital",
        "combined_ath_x_capital", "breakout_score_x_dominator",
    ]
    return cap


def test_a5_decomposition(pos, df, full_features, nocap_features, alpha_range, target=TARGET):
    log.info("A5 Capital decomposition [%s] ...", pos)
    labeled = df[df[target].notna()].copy()

    # Full model
    r2_full = _loyo_r2(labeled, full_features, target, alpha_range)

    # Phase I (no capital) — use the nocap candidate pool, run Lasso to select
    nocap_avail = _available_candidates(labeled, nocap_features)
    nocap_selected = _lasso_select(labeled[nocap_avail], labeled[target])
    r2_nocap = _loyo_r2(labeled, nocap_selected, target, alpha_range)

    # Capital only — run Lasso on capital signals to find which ones matter
    cap_candidates = [f for f in _capital_only_features(pos) if f in labeled.columns
                      and labeled[f].notna().sum() >= 5
                      and labeled[f].nunique(dropna=True) > 1]
    if cap_candidates:
        cap_selected = _lasso_select(labeled[cap_candidates], labeled[target])
        if not cap_selected:
            cap_selected = cap_candidates
    else:
        cap_selected = []
    r2_cap = _loyo_r2(labeled, cap_selected, target, alpha_range) if cap_selected else float("nan")

    log.info("  A5 [%s] capital-only=%.4f  nocap(Phase I)=%.4f  full=%.4f",
             pos, r2_cap, r2_nocap, r2_full)
    log.info("  A5 [%s] Phase I features: %s", pos, nocap_selected)
    log.info("  A5 [%s] capital features: %s", pos, cap_selected)

    return {
        "capital_only_loyo_r2":  round(r2_cap, 4),
        "phase1_loyo_r2":        round(r2_nocap, 4),
        "full_model_loyo_r2":    round(r2_full, 4),
        "capital_features":      cap_selected,
        "phase1_features":       nocap_selected,
        "incremental_over_cap":  round(r2_full - r2_cap, 4),
        "incremental_over_nocap": round(r2_full - r2_nocap, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase A stress tests.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--position", choices=["WR", "RB", "TE"])
    parser.add_argument("--test", choices=["A1", "A2", "A3", "A4", "A5"],
                        help="Run a single test (default: all)")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap resamples for A1 (default: 1000)")
    parser.add_argument("--n-perm", type=int, default=100,
                        help="Permutations for A4 (default: 100)")
    parser.add_argument("--start-year", type=int, default=2014)
    args = parser.parse_args()

    data_dir = get_data_dir()
    positions = [args.position] if args.position else list(_POSITIONS)
    run_tests = ([args.test] if args.test else ["A1", "A2", "A3", "A4", "A5"])

    all_results = {}

    for pos in positions:
        log.info("===== %s =====", pos)
        df = _load_pos(pos, data_dir, args.start_year)
        alpha_range = _RIDGE_ALPHA_RANGE[pos]

        # Get the current production model's selected features from metadata
        meta_path = _ROOT / "models" / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            full_features = meta.get(pos, {}).get("selected_features", [])
        else:
            full_features = []

        if not full_features:
            log.warning("  No metadata found for %s — running outer Lasso", pos)
            full_features = _outer_lasso_features(df, _CANDIDATE_FEATURES[pos])

        log.info("  Production model features (%d): %s", len(full_features), full_features)

        pos_results = {"full_features": full_features}

        if "A1" in run_tests:
            r = test_a1_bootstrap(pos, df, _CANDIDATE_FEATURES[pos], n_boot=args.n_boot)
            pos_results["A1_bootstrap"] = r
            log.info("  A1 top-10 stability:")
            for x in r[:10]:
                log.info("    %-45s  %.1f%%", x["feature"], x["freq"] * 100)

        if "A2" in run_tests:
            r = test_a2_knockout(pos, df, full_features, alpha_range)
            pos_results["A2_knockout"] = r

        if "A3" in run_tests:
            r = test_a3_calibration(pos, df, full_features, alpha_range)
            pos_results["A3_calibration"] = r
            if isinstance(r, dict):
                log.info("  A3 [%s] Spearman rho=%.4f (p=%.4f)", pos,
                         r["spearman_rho"], r["spearman_p"])
                log.info("  A3 [%s] Top-25pct hit rate @ ZAP>=75: %.1f%% (base=%.1f%%)",
                         pos, r["top25pct_hit_rate_at_zap75"] * 100,
                         r["top25pct_base_rate"] * 100)
                for b in r["bins"]:
                    bias_str = f"{b['bias']:+.2f}"
                    log.info("    ZAP %s  n=%3d  pred=%.2f  actual=%.2f  bias=%s",
                             b["bin_label"], b["n"], b["pred_b2s_mean"],
                             b["actual_b2s_mean"], bias_str)

        if "A4" in run_tests:
            r = test_a4_permutation(pos, df, full_features, alpha_range, n_perm=args.n_perm)
            pos_results["A4_permutation"] = r
            log.info("  A4 [%s] real R²=%.4f  perm_mean=%.4f  p=%.4f  signal=%s",
                     pos, r["real_loyo_r2"], r["perm_mean"], r["p_value"],
                     "CONFIRMED" if r["signal_confirmed"] else "NOT CONFIRMED")

        if "A5" in run_tests:
            nocap_cands = _CANDIDATE_FEATURES_NOCAP[pos]
            r = test_a5_decomposition(pos, df, full_features, nocap_cands, alpha_range)
            pos_results["A5_decomposition"] = r
            log.info("  A5 [%s] cap-only=%.4f  phase1=%.4f  full=%.4f",
                     pos, r["capital_only_loyo_r2"], r["phase1_loyo_r2"],
                     r["full_model_loyo_r2"])
            log.info("  A5 [%s] incremental over capital-only: %+.4f",
                     pos, r["incremental_over_cap"])

        all_results[pos] = pos_results

    # Save JSON results
    out_path = OUT_DIR / "stress_test_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    log.info("Results saved: %s", out_path)

    # Print summary table
    print("\n" + "=" * 80)
    print("  Phase A Stress Test Summary")
    print("=" * 80)
    for pos, res in all_results.items():
        print(f"\n  {pos} (features: {res['full_features']})")
        if "A4_permutation" in res:
            p = res["A4_permutation"]
            print(f"    A4 Permutation: real R²={p['real_loyo_r2']:.4f}  "
                  f"perm_mean={p['perm_mean']:.4f}  p={p['p_value']:.4f}  "
                  f"{'SIGNAL CONFIRMED' if p['signal_confirmed'] else 'WEAK SIGNAL'}")
        if "A5_decomposition" in res:
            d = res["A5_decomposition"]
            print(f"    A5 Decomp:   cap-only={d['capital_only_loyo_r2']:.4f}  "
                  f"phase1={d['phase1_loyo_r2']:.4f}  "
                  f"full={d['full_model_loyo_r2']:.4f}  "
                  f"incr={d['incremental_over_cap']:+.4f}")
        if "A3_calibration" in res and isinstance(res["A3_calibration"], dict):
            c = res["A3_calibration"]
            print(f"    A3 Calibr:   Spearman rho={c['spearman_rho']:.4f}  "
                  f"top25%_hit={c['top25pct_hit_rate_at_zap75']:.1%}  "
                  f"base={c['top25pct_base_rate']:.1%}")
    print("=" * 80)


if __name__ == "__main__":
    main()
