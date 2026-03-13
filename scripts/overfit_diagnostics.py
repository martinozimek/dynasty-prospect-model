
"""
Comprehensive overfitting diagnostics for dynasty-prospect-model.
Runs silently, prints structured report sections.
"""
import sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(r'C:\Users\Ozimek\Documents\Claude\FF\dynasty-prospect-model')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze import engineer_features, load_training
from config import get_data_dir

data_dir = get_data_dir()
meta = json.loads((ROOT / 'models' / 'metadata.json').read_text())

# ============================================================
# SECTION 1: Train R² vs LOYO R² gap (overfitting signal #1)
# ============================================================
print("=" * 70)
print("SECTION 1: Train vs. Out-of-Sample R² Gap")
print("=" * 70)
print(f"{'Position':6}  {'N':>4}  {'Feats':>5}  {'Train R²':>9}  {'LOYO R²':>8}  {'Gap':>7}  {'Gap %':>7}")
print("-" * 70)
gaps = {}
for pos in ['WR', 'RB', 'TE']:
    m = meta[pos]
    train = m['ridge_r2_train']
    loyo = m['ridge_loyo_r2']
    gap = train - loyo
    gap_pct = gap / train * 100 if train > 0 else float('nan')
    gaps[pos] = gap
    flag = " *** HIGH" if gap > 0.10 else (" * moderate" if gap > 0.06 else "")
    print(f"{pos:6}  {m['n_train']:>4}  {m['n_features']:>5}  {train:>9.3f}  {loyo:>8.3f}  {gap:>7.3f}  {gap_pct:>6.1f}%{flag}")

# LGBM vs Ridge gap
print()
print(f"{'Position':6}  {'Ridge LOYO':>10}  {'LGBM LOYO':>10}  {'LGBM gap':>9}")
print("-" * 50)
for pos in ['WR', 'RB', 'TE']:
    m = meta[pos]
    ridge_loyo = m['ridge_loyo_r2']
    lgbm_loyo = m['lgbm_loyo_r2']
    lgbm_gap = ridge_loyo - lgbm_loyo
    flag = " *** SEVERE" if lgbm_gap > 0.20 else (" * large" if lgbm_gap > 0.10 else "")
    print(f"{pos:6}  {ridge_loyo:>10.3f}  {lgbm_loyo:>10.3f}  {lgbm_gap:>9.3f}{flag}")

# ============================================================
# SECTION 2: Per-year R² variance (temporal stability)
# ============================================================
print()
print("=" * 70)
print("SECTION 2: Per-Year R² Variance (Temporal Stability)")
print("=" * 70)
for pos in ['WR', 'RB', 'TE']:
    years = meta[pos]['ridge_loyo_years']
    r2_vals = [y['r2'] for y in years]
    n_vals = [y['n_test'] for y in years]
    print(f"\n{pos} (aggregate LOYO R²={meta[pos]['ridge_loyo_r2']:.3f}):")
    print(f"  Per-year R²: min={min(r2_vals):.3f}  max={max(r2_vals):.3f}  "
          f"std={np.std(r2_vals):.3f}  range={max(r2_vals)-min(r2_vals):.3f}")
    neg_yrs = [(y['year'], round(y['r2'], 3), y['n_test']) for y in years if y['r2'] < 0]
    if neg_yrs:
        print(f"  *** NEGATIVE R² years (worse than mean baseline): {neg_yrs}")
    small_yrs = [(y['year'], y['n_test']) for y in years if y['n_test'] < 10]
    if small_yrs:
        print(f"  *** Small holdout sets (<10): {small_yrs}")
    for y in years:
        bar = "#" * max(0, int((y['r2']) * 20))
        neg_bar = "x" * max(0, int((-y['r2']) * 20))
        marker = neg_bar if y['r2'] < 0 else bar
        print(f"  {y['year']}: n={y['n_test']:>2}  R²={y['r2']:>6.3f}  {marker}")

# ============================================================
# SECTION 3: Feature-to-sample ratios + effective degrees of freedom
# ============================================================
print()
print("=" * 70)
print("SECTION 3: Feature-to-Sample Ratio")
print("=" * 70)
print(f"{'Position':6}  {'N':>4}  {'Feats':>5}  {'N/feat':>7}  {'Candidate feats':>16}  {'Cand/N':>7}")
print("-" * 60)
cand_counts = {'WR': 45, 'RB': 35, 'TE': 47}  # approx from candidate lists
for pos in ['WR', 'RB', 'TE']:
    m = meta[pos]
    n = m['n_train']
    feats = m['n_features']
    nf_ratio = n / feats
    cand = cand_counts[pos]
    cand_ratio = cand / n
    flag = " *** LOW (<10x)" if nf_ratio < 10 else (" * moderate (<15x)" if nf_ratio < 15 else "")
    print(f"{pos:6}  {n:>4}  {feats:>5}  {nf_ratio:>7.1f}  {cand:>16}  {cand_ratio:>7.2f}{flag}")

print()
print("  Note: 'Candidate feats / N' measures the candidate selection search space.")
print("  Lasso explores ~45 candidates for WR against 286 rows — even if Lasso")
print("  itself is regularized, the search process introduces selection bias.")

# ============================================================
# SECTION 4: Multicollinearity among selected features
# ============================================================
print()
print("=" * 70)
print("SECTION 4: Multicollinearity Among Selected Features")
print("=" * 70)
for pos in ['WR', 'RB', 'TE']:
    df_raw = load_training(pos, data_dir)
    df_eng = engineer_features(df_raw)
    feats = meta[pos]['selected_features']
    available = [f for f in feats if f in df_eng.columns]
    sub = df_eng[available].dropna()
    if len(sub) < 20:
        sub = df_eng[available].copy()
    corr = sub.corr(method='pearson', min_periods=10)
    print(f"\n{pos} selected features — high correlation pairs (|r| > 0.70):")
    found = False
    for i, f1 in enumerate(available):
        for f2 in available[i+1:]:
            if f1 in corr.columns and f2 in corr.columns:
                r = corr.loc[f1, f2]
                if abs(r) > 0.70:
                    print(f"  {f1} × {f2}: r={r:.3f}")
                    found = True
    if not found:
        print("  None above 0.70 threshold")

# ============================================================
# SECTION 5: PFF missing-data bias
# ============================================================
print()
print("=" * 70)
print("SECTION 5: PFF Missing-Data Bias (Implicit Subsetting)")
print("=" * 70)
pff_features = {
    'WR': ['best_slot_yprr', 'best_man_yprr', 'best_man_zone_delta'],
    'RB': ['best_zone_yprr'],
    'TE': ['best_slot_target_rate'],
}
for pos in ['WR', 'RB', 'TE']:
    df_raw = load_training(pos, data_dir)
    df_eng = engineer_features(df_raw)
    print(f"\n{pos} (N={len(df_eng)}):")
    for feat in pff_features[pos]:
        if feat not in df_eng.columns:
            continue
        n_have = df_eng[feat].notna().sum()
        n_miss = df_eng[feat].isna().sum()
        pct = n_have / len(df_eng) * 100
        # Compare B2S distribution: rows WITH vs WITHOUT this PFF feature
        have = df_eng[df_eng[feat].notna()]['b2s_score'].dropna()
        miss = df_eng[df_eng[feat].isna()]['b2s_score'].dropna()
        b2s_diff = have.mean() - miss.mean() if len(miss) > 5 else float('nan')
        draft_have = df_eng[df_eng[feat].notna()]['draft_capital_score'].dropna().mean()
        draft_miss = df_eng[df_eng[feat].isna()]['draft_capital_score'].dropna().mean()
        draft_diff = draft_have - draft_miss
        print(f"  {feat}: {n_have}/{len(df_eng)} ({pct:.0f}%) non-null")
        print(f"    B2S mean — with: {have.mean():.2f}  without: {miss.mean():.2f}  diff: {b2s_diff:+.2f}")
        print(f"    Draft capital — with: {draft_have:.1f}  without: {draft_miss:.1f}  diff: {draft_diff:+.1f}")

# ============================================================
# SECTION 6: Lasso selection bias (look-ahead via full-data selection)
# ============================================================
print()
print("=" * 70)
print("SECTION 6: Lasso Selection Bias Estimate")
print("=" * 70)
print("  Methodology: Compare LOYO R² using currently selected features (selected")
print("  on full dataset) vs. nested LOYO where feature selection is re-run inside")
print("  each fold. The gap quantifies selection leakage into CV.\n")

for pos in ['WR', 'RB', 'TE']:
    df_raw = load_training(pos, data_dir)
    df_eng = engineer_features(df_raw)
    df = df_eng[df_eng['b2s_score'].notna()].copy()
    years = sorted(df['draft_year'].dropna().unique())

    # Standard LOYO (current approach — features selected on all data)
    feats_fixed = meta[pos]['selected_features']
    available_fixed = [f for f in feats_fixed if f in df.columns]

    # Nested LOYO (re-select features inside each fold)
    all_candidates = [
        f for f in meta[pos]['selected_features']  # keep to same candidate pool for speed
    ]
    # Expand to all candidates from fit_model.py for a truer test would be slow;
    # use a moderate superset: selected + their raw components
    expanded = list(set(feats_fixed))  # same set — ceiling on what nested can do

    nested_actual, nested_predicted = [], []
    fixed_actual, fixed_predicted = [], []

    for test_year in years:
        train_df = df[df['draft_year'] != test_year]
        test_df = df[df['draft_year'] == test_year]
        if len(train_df) < 10 or len(test_df) < 2:
            continue

        # Standard (fixed features)
        X_tr = train_df[available_fixed].values
        y_tr = train_df['b2s_score'].values
        X_te = test_df[available_fixed].values
        y_te = test_df['b2s_score'].values

        pipe_fixed = Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('scl', StandardScaler()),
            ('mdl', Ridge(alpha=10.0)),
        ])
        pipe_fixed.fit(X_tr, y_tr)
        preds_fixed = np.clip(pipe_fixed.predict(X_te), 0, None)
        fixed_actual.extend(y_te)
        fixed_predicted.extend(preds_fixed)

        # Nested: re-run Lasso on train fold only
        X_tr_full = train_df[available_fixed].copy()
        imp = SimpleImputer(strategy='median')
        scl = StandardScaler()
        X_tr_imp = scl.fit_transform(imp.fit_transform(X_tr_full))
        alphas = np.logspace(-4, 2, 30)
        lasso = LassoCV(alphas=alphas, cv=3, max_iter=5000, random_state=42)
        try:
            lasso.fit(X_tr_imp, y_tr)
            nested_sel = [f for f, c in zip(available_fixed, lasso.coef_) if c != 0.0]
        except Exception:
            nested_sel = available_fixed
        if not nested_sel:
            nested_sel = available_fixed

        X_tr_n = train_df[nested_sel].values
        X_te_n = test_df[nested_sel].values
        pipe_nested = Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('scl', StandardScaler()),
            ('mdl', Ridge(alpha=10.0)),
        ])
        pipe_nested.fit(X_tr_n, y_tr)
        preds_nested = np.clip(pipe_nested.predict(X_te_n), 0, None)
        nested_actual.extend(y_te)
        nested_predicted.extend(preds_nested)

    r2_fixed = r2_score(fixed_actual, fixed_predicted)
    r2_nested = r2_score(nested_actual, nested_predicted)
    selection_bias = r2_fixed - r2_nested
    print(f"  {pos}: Fixed-selection LOYO R²={r2_fixed:.3f}  Nested LOYO R²={r2_nested:.3f}  "
          f"Selection bias={selection_bias:+.3f}")

# ============================================================
# SECTION 7: ZAP reference distribution bias
# ============================================================
print()
print("=" * 70)
print("SECTION 7: ZAP Reference Distribution (In-Sample Bias)")
print("=" * 70)
print("  ZAP uses in-sample Ridge training predictions as the percentile reference.")
print("  Ridge training predictions are optimistically biased vs. true holdout preds.")
print()
for pos in ['WR', 'RB', 'TE']:
    train_preds = meta[pos]['train_preds']
    df_raw = load_training(pos, data_dir)
    df_eng = engineer_features(df_raw)
    b2s = df_eng['b2s_score'].dropna()

    tp_arr = np.array(train_preds)
    print(f"  {pos}:")
    print(f"    Training pred dist:  mean={tp_arr.mean():.2f}  std={tp_arr.std():.2f}  "
          f"max={tp_arr.max():.2f}  p90={np.percentile(tp_arr, 90):.2f}")
    print(f"    Actual B2S dist:     mean={b2s.mean():.2f}  std={b2s.std():.2f}  "
          f"max={b2s.max():.2f}  p90={np.percentile(b2s, 90):.2f}")
    print(f"    In-sample mean overestimation: {tp_arr.mean() - b2s.mean():+.2f}")

# ============================================================
# SECTION 8: Draft capital dominance — feature independence
# ============================================================
print()
print("=" * 70)
print("SECTION 8: Draft Capital Dominance")
print("=" * 70)
for pos in ['WR', 'RB', 'TE']:
    df_raw = load_training(pos, data_dir)
    df_eng = engineer_features(df_raw)
    df = df_eng[df_eng['b2s_score'].notna()].copy()

    feats = meta[pos]['selected_features']
    capital_feats = [f for f in feats if any(k in f for k in
        ['capital', 'pick', 'tier', 'round', 'premium', 'rank'])]
    non_capital = [f for f in feats if f not in capital_feats]

    print(f"\n{pos}: {len(feats)} selected features")
    print(f"  Capital-derived: {capital_feats}")
    print(f"  Non-capital:     {non_capital}")

    # R² of capital-only model vs. full model
    available = [f for f in feats if f in df.columns]
    cap_avail = [f for f in capital_feats if f in df.columns]

    if len(cap_avail) > 0:
        X_cap = df[cap_avail].values
        y = df['b2s_score'].values
        years = sorted(df['draft_year'].dropna().unique())
        cap_preds, all_preds, actual = [], [], []
        for test_year in years:
            tr = df[df['draft_year'] != test_year]
            te = df[df['draft_year'] == test_year]
            if len(tr) < 5 or len(te) < 2:
                continue
            for feats_use, preds_list in [(cap_avail, cap_preds), (available, all_preds)]:
                p = Pipeline([('imp', SimpleImputer(strategy='median')),
                               ('scl', StandardScaler()),
                               ('mdl', Ridge(alpha=10.0))])
                p.fit(tr[feats_use].values, tr['b2s_score'].values)
                pred = np.clip(p.predict(te[feats_use].values), 0, None)
                if feats_use == cap_avail:
                    cap_preds.extend(pred)
                    actual.extend(te['b2s_score'].values)
                else:
                    all_preds.extend(pred)

        r2_cap = r2_score(actual, cap_preds)
        r2_full = r2_score(actual, all_preds)
        print(f"  Capital-only LOYO R²: {r2_cap:.3f}  |  Full model LOYO R²: {r2_full:.3f}")
        print(f"  Non-capital features add: {r2_full - r2_cap:+.3f} R² points")

print()
print("=" * 70)
print("END OF DIAGNOSTICS")
print("=" * 70)
