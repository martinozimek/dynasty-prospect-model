# Tight End Model — N Expansion Roadmap

**Status**: Active (Step 7e of overfitting remediation plan, 2026-03-13)
**Root constraint**: TE N=97 across 9 draft years (2014–2022) — binding limitation.

---

## Current State

After 2014+ training cutoff (Step 1), TE has:
- **N=97** labeled training rows
- **9 draft years** (2014–2022), smallest folds: 2014 (n=7), 2016 (n=6)
- **10 selected features** → N/feat ratio of 9.7x (below 15x safety threshold)
- **LOYO R²=0.408** but inflated by lucky 2016 fold (n=6, R²=0.826)
- Rolling-window R² (train≤2018, test≥2019) is the honest stability measure

No amount of regularization or CV tuning fully overcomes N=97.
**The path forward is more data.**

---

## Expansion Path

### 1. Natural class maturation (~2027)
- **2023 draft class eligible ~2027** (3 complete NFL seasons by end of 2026)
  - Expected: ~10–12 TE rows
  - Prerequisite: nfl-fantasy-db B2S computation for 2023 class
- **2024 draft class eligible ~2028**
  - Expected: ~10–12 rows

### 2. FCS-drafted players (~5–10 rows/year)
- Check nflverse + PFF college coverage for FCS players who were drafted
- Example: Dalton Kincaid (San Diego) — verify if PFF has his college data
- If FCS coverage exists (even partial), adds 5–10 rows per draft year back to 2014
- Implementation: update `build_training_set.py` to not filter FCS if PFF data present

### 3. Target N: 150 TE rows (2014–2024 classes)
- At 150 rows: 10 features → **15x N/feat ratio** (safe threshold)
- Enables running full-candidate Lasso reliably without selection instability
- Timeline: realistically 2028 for natural class maturation to 150

---

## Interim Mitigations (already applied)

| Mitigation | Status | Effect |
|---|---|---|
| 2014+ cutoff (Step 1) | ✅ Applied | Removed era bias; accepted N drop 127→97 |
| VIF pruning (Step 7a) | ✅ Applied | Removed `combined_ath_x_capital` (r=0.981) + `draft_premium` (r=−0.953) |
| Alpha tuning α∈[1,1000] (Step 7b) | ✅ Applied | Higher shrinkage for small N |
| Rolling-window CV (Step 7c) | ✅ Applied | More honest than LOYO for tiny folds |
| LGBM retired (Step 7d) | ✅ Applied | Gap >0.10 — using Ridge only for TE |
| Nested Lasso CV (Step 3) | ✅ Applied | Honest LOYO (selection bias was ~0.002) |

---

## Current TE Model Limitations (communicate to users)

1. **N=97 is small**: each 1-point change in ZAP score is within model noise
2. **LOYO R²=0.408 is inflated** by the 2016 holdout (n=6, R²=0.826 — luck)
3. **Rolling-window R²** (reported in score output) is the honest estimate
4. **Feature count = 10** at the N/feat safety threshold — further feature additions require more data
5. **TEs are inherently harder to model**: high variance in NFL usage decisions, blocking requirements, positional scarcity all create noise that college data can't capture

---

## Monitoring

- After 2023 NFL season concludes (March 2027): run `populate_nfl.py` to update B2S
- Re-run `build_training_set.py` — check if N increases to ~107+
- Re-run `overfit_diagnostics.py` — track if TE N/feat ratio improves
- Revisit LGBM retirement decision once N≥120

---

*Created: 2026-03-13 | Next review: 2027 Q1*
