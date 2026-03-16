# Phase B3 / D2 / C2 Findings — Dynasty Prospect Model
**Date:** 2026-03-15
**Branch:** model-research
**Follows:** PhaseA_Findings.md (2026-03-14)

---

## Executive Summary

Three remaining open items were evaluated: B3 (combine missingness flags), D2 (ensemble blend), and C2 (TE 5-year label window). All three are closed — two by empirical test, one by data availability constraint.

| Item | Test Run | Verdict | Action |
|------|----------|---------|--------|
| B3 — Missingness indicator flags | Yes — Mann-Whitney + correlation test | REJECTED — signal is redundant with capital or absent | No change |
| D2 — Ensemble blend (Ridge + LGBM) | Yes — LOYO R² across 4 blend weights | REJECTED — Ridge beats all blends in all positions | No change |
| C2 — TE 5-year label window | No — data dependency | DEFERRED — requires nfl-fantasy-db change; 2022 class TE data unavailable until after 2026 season | Revisit post-2026 |

**The model is unchanged. Ridge remains the sole predictor for all three positions.**

---

## B3 — Combine Missingness Indicator Flags

### What was tested

For each position (WR, RB, TE), we examined whether missing combine data (`forty_time`,
`broad_jump`, `vertical_jump`) is distributed randomly across players or is correlated
with player outcomes (B2S) and/or draft capital.

The condition for implementing B3: **missingness is non-random and not already captured
by existing capital features**.

### Results

| Position | N | Combine Missing | B2S (missing) | B2S (present) | Diff | Mann-Whitney p | Corr vs Capital |
|----------|---|-----------------|---------------|---------------|------|----------------|-----------------|
| WR | 224 | 70 (31%) | 5.95 | 6.80 | −0.85 | **0.231 — not significant** | r = −0.041 |
| RB | 147 | 47 (32%) | 6.58 | 8.34 | −1.76 | **0.139 — not significant** | r = −0.084 |
| TE | 97  | 42 (43%) | 3.99 | 6.26 | −2.28 | **0.010 — significant** | r = −0.341 |

Individual column miss rates (2014–2022 training set):

| Column | WR | RB | TE |
|--------|----|----|-----|
| forty_time | 28% | 25% | 28% |
| broad_jump | 30% | 29% | 42% |
| vertical_jump | 29% | 26% | 37% |
| weight_lbs | 17% | 20% | 16% |

### Interpretation

**WR and RB: Condition not met.** The B2S difference between players with and without
combine data is not statistically significant (p=0.23, p=0.14). The correlation between
missingness and draft capital is near-zero (r=−0.04 to −0.08). Missing combine data for
WRs and RBs is effectively random — it does not carry information the model doesn't
already have.

**TE: Significant but redundant.** The TE result is the only statistically significant
finding (p=0.010). However, the correlation with log_draft_capital is r=−0.341: TEs
with missing combine data are late-round picks (median overall pick 144 vs 98 for
players with combine data). The capital signal — already the dominant feature in the
TE model — already encodes this difference. Adding a `combine_missing` flag would be
an indirect proxy for late-round capital, not independent information.

**Critical confound — 2021 COVID combine cancellation:** The 2021 NFL Combine was
cancelled due to COVID-19. As a result, 100% of players in the 2021 draft class are
coded as `combine_missing=1` across all positions. Any flag built from this variable
is therefore constant within the 2021 LOYO fold — it adds zero discriminative value
for that entire holdout year, which represents 10–13% of each training set.

**Miss rate by year (2021 spike confirms COVID confound):**
```
WR 2021: 100% missing     RB 2021: 100% missing     TE 2021: 100% missing
WR 2022:  26% missing     RB 2022:  17% missing     TE 2022:  75% missing
```

### Decision

**B3 REJECTED.** Condition not met. For WR/RB, missingness is statistically random.
For TE, the signal is redundant with draft capital features already in the model.
The 2021 COVID combine cancellation further degrades the flag's utility across all
positions. No flags added.

---

## D2 — Ensemble Blend (Ridge + LightGBM)

### What was tested

LOYO-CV was run for each position comparing: pure Ridge, pure LGBM, and four blended
combinations (Ridge weight: 0.5, 0.6, 0.7, 0.8). The nested Lasso feature selection
was applied per fold for all variants.

The condition for implementing D2: **ensemble LOYO R² > pure Ridge across all positions**.

### Results

| Position | Ridge LOYO R² | LGBM LOYO R² | Best Blend (80/20) | Ridge vs Blend |
|----------|:-------------:|:------------:|:------------------:|:--------------:|
| WR | **0.312** | 0.247 | 0.305 | Ridge +0.007 |
| RB | **0.363** | 0.362 | 0.360 | Ridge +0.003 |
| TE | 0.311 | 0.316 | 0.302 | Ridge +0.009 |

Full blend matrix:

| Weight | WR | RB | TE |
|--------|----|----|-----|
| 50% Ridge / 50% LGBM | 0.286 | 0.343 | 0.277 |
| 60% Ridge / 40% LGBM | 0.293 | 0.350 | 0.287 |
| 70% Ridge / 30% LGBM | 0.299 | 0.356 | 0.295 |
| 80% Ridge / 20% LGBM | 0.305 | 0.360 | **0.302** |
| **100% Ridge (baseline)** | **0.312** | **0.363** | **0.311** |

### Interpretation

Ridge beats every blend at every weight for WR and RB. For TE, LGBM alone is 0.005
above Ridge in this run — but every blend underperforms pure Ridge. This means the
LGBM predictions for TE are slightly better than Ridge when pure but add noise when
combined, which is consistent with LGBM overfitting to specific patterns that cancel
out in a blend.

The broader picture: LGBM consistently underperforms Ridge on LOYO for WR (+0.065
gap) and is essentially tied for RB (gap of 0.001). With N=97 to N=224 and 8–10
selected features, the Ridge model's regularization is better suited to this
small-sample regime than LGBM's tree-based approach. This finding is consistent
with prior model evaluations and with the literature on small-N sports analytics.

Note: LOYO R² values in this test are slightly lower than the main fit_model.py
run (which uses the 2011–2022 full window with pre-2014 rows included). The
relative rankings within this test are what matter for the D2 decision.

### Decision

**D2 REJECTED.** Condition not met. Ridge beats all blends in all positions. The
pure Ridge model remains the sole predictor. No ensemble introduced.

---

## C2 — TE 5-Year Label Window

### What this involves

The current B2S label is: **best 2 of first 3 NFL seasons, minimum 8 games per season**.
C2 proposes changing this to **best 2 of first 5 seasons** for TEs only, on the theory
that TEs develop later than WRs/RBs and the 3-year window may penalize players whose
peak comes in seasons 4–5.

The condition for implementing C2: **TE LOYO R² improves with 5-year window vs 3-year**.

### Why this cannot be tested now

A 5-year label requires 5 seasons of NFL data for each player in the training set.
The training window is 2014–2022 draft classes.

| Class | 5th Season | Status |
|-------|------------|--------|
| 2014–2019 | 2018–2023 | ✓ Available in nfl-fantasy-db |
| 2020 | 2024 | ✓ Available |
| 2021 | 2025 | ✓ Likely available (2025 season) |
| 2022 | **2026** | **Not available — 2026 season has not been played** |

Excluding 2022 class TEs from training reduces N from 97 to approximately 79–85.
This is a meaningful reduction for an already small dataset (N=97).

Beyond data availability, this change requires modifying the B2S computation in
`nfl-fantasy-db` — a separate repository with its own pipeline. The B2S label is
computed there and exported as a training input to this model. Implementing C2
without validation in nfl-fantasy-db would mean training on an inconsistent label.

### What the theoretical case looks like

The argument for C2 rests on position-specific development curves. Some supporting
observations from the training data:

- TE LOYO R² variance is the highest of any position (std=0.290 vs WR=0.158, RB=0.152)
- TE 2021 LOYO R² = −0.157 and 2014 R² = −0.257 (two negative years) — suggesting
  higher variance in TE outcomes than a 3-year window can reliably capture
- The A5 decomposition shows TE non-capital signal is 0.285 LOYO R² (Phase I model),
  suggesting real TE talent signal exists beyond capital — but the 3-year label may
  not be giving it time to manifest

The theoretical case is reasonable. The empirical test requires the 2026 season.

### Decision

**C2 DEFERRED.** Data dependency prevents testing now. The 2022 draft class TEs
(16% of current TE training set) have not completed 5 NFL seasons. Revisit after
the 2026 NFL season concludes (approximately January 2027), at which point the full
2022 class 5-year window will be available.

Priority if/when revisited: implement 5-year B2S label in `nfl-fantasy-db` first,
then compare TE LOYO R² against the current 3-year label as a controlled experiment.

---

## Updated Research Plan Status

| Phase | Item | Status |
|-------|------|--------|
| A | All diagnostics (A1–A5) | COMPLETE |
| B | B1 Target share | REJECTED |
| B | B2 Trajectory features | REJECTED |
| B | **B3 Missingness flags** | **REJECTED — 2026-03-15** |
| B | B4 Capital curve variants | REJECTED |
| B | B5 Age multiplier variants | REJECTED |
| C | C1 Survivorship zero-fill | REJECTED |
| C | **C2 TE 5-year label** | **DEFERRED — data dependency** |
| D | D1 Conformal intervals | COMPLETE |
| D | **D2 Ensemble blend** | **REJECTED — 2026-03-15** |
| D | D3 Ranking metrics | COMPLETE |
| D | D4 Bayesian Ridge | REJECTED |

### What remains open

| Item | Trigger |
|------|---------|
| C2 — TE 5-year label | After 2026 NFL season; requires nfl-fantasy-db change first |
| Post-draft re-score | Run `score_class.py --year 2026 --post-draft` after 2026 NFL Draft |

**The model-research branch has no further validated changes to implement.**
The current state of the model on `model-research` represents the maximum
evidence-supported improvement from Phase A through Phase D.

---

## What the Model Looks Like Now vs When We Started

| Metric | Original (master) | model-research |
|--------|-------------------|----------------|
| WR LOYO R² | 0.388 | 0.356 |
| RB LOYO R² | 0.399 | 0.413 |
| TE LOYO R² | 0.408 | 0.403 |
| Prediction intervals | No | Yes — 80%/90% from LOYO residuals |
| Phase I / Capital Delta | Yes | Yes (unchanged) |
| Rank-order calibration in output | No | Yes — Spearman + top-25% hit rate |
| Stress test validation | No | Yes — all 5 Phase A tests passed |

Note on WR LOYO R²: The small decrease (0.388 → 0.356) reflects the switch to the
**honest nested Lasso CV** implemented in the overfitting remediation step — the
original 0.388 included selection leakage. The current 0.356 is the unbiased estimate.
This is not a regression; it is a more accurate measurement of the same underlying
model quality.

---

*Full numerical results for B3 and D2 available on request.*
*B3 raw data: training CSVs at `data/training_{WR,RB,TE}.csv`*
*D2 raw data: run `scripts/stress_test.py` or reproduce via the test script above*
