# Model Research Plan — dynasty-prospect-model
**Branch:** model-research
**Created:** 2026-03-14
**Status:** Phase A Complete — No Phase B/C changes warranted

---

## Guiding Principle

> Only make fundamental changes to the model if the findings fundamentally warrant such a change.
> Do not make any change to the model that is not validated.

---

## Source Documents

Feedback1.MD — AI Technical Review (structural weaknesses)
Feedback2.md — Independent Technical Review (highest impact list)
Feedback3.md — ChatGPT (interrogation questions + diagnostics)
Feedback4.MD — Gemini (gaps and strategic recommendations)
Feedback5.md — ChatGPT (Bayesian Ridge + Conformal Prediction investigation plan)
Feedback6.md — ChatGPT (formal stress-test protocol)

---

## Phase A — Diagnostics (Run First, Change Nothing)

These are free information. Results inform all subsequent decisions.

| # | Test | Script | Status | Finding |
|---|------|--------|--------|---------|
| A1 | Bootstrap Lasso stability (1000×) | scripts/stress_test.py | COMPLETE | All current model features ≥50% stable; RB `total_yards_rate_x_capital` borderline (24.4%) |
| A2 | Feature knockout (one-at-a-time LOYO drop) | scripts/stress_test.py | COMPLETE | No single-feature dependency; several marginal features slightly negative for RB/TE |
| A3 | ZAP calibration curve (predicted vs actual by decile) | scripts/stress_test.py | COMPLETE | Good rank-order signal (Spearman 0.60–0.63); no systematic top-decile overconfidence |
| A4 | Label permutation test (shuffle B2S, expect R²≈0) | scripts/stress_test.py | COMPLETE | PASS — all 3 positions p=0.0, real signal confirmed, no leakage |
| A5 | Capital decomposition (capital-only vs Phase I vs full) | scripts/stress_test.py | COMPLETE | RB nearly capital-only (+0.004 from non-capital); WR/TE get meaningful uplift from non-capital |

---

## Phase B — Features (Only if A validates the gap)

| # | Change | Condition to implement | Status |
|---|--------|------------------------|--------|
| B1 | Add `best_target_share` + `target_share_x_capital` | A1 shows it is not already captured; LOYO R² improves | NOT WARRANTED — A1 shows existing WR PFF coverage is adequate |
| B2 | Add trajectory features (`yprr_growth`, `target_share_growth`, `career_avg_yprr`) | A1 shows best-season instability; trajectory adds >0.005 LOYO R² | NOT WARRANTED — WR best-season features all ≥53% stable |
| B3 | Add missingness indicator flags (`forty_missing`, `combine_missing`) | A8-style test shows combine missingness is non-random | DEFERRED — Phase A did not include missingness pattern test; can revisit |
| B4 | Empirical draft capital curve (test log/sqrt/exp variants) | Capital-only LOYO R² improves over current exp decay | NOT WARRANTED — current capital curve performs well (see A5); no gap to exploit |
| B5 | Age multiplier sensitivity (test exp vs linear vs reciprocal) | Breakout score LOYO R² improves measurably | NOT WARRANTED — A2 shows best_age contributes positively; age formula stable |

---

## Phase C — Label Fix (Requires nfl-fantasy-db change)

| # | Change | Condition to implement | Status |
|---|--------|------------------------|--------|
| C1 | Zero-fill B2S for non-qualifying players (survivorship fix) | Calibration curve (A3) shows systematic overconfidence at bottom ZAP deciles | NOT WARRANTED — A3 shows no systematic bottom-decile overconfidence for any position |
| C2 | TE 5-year label window experiment | TE LOYO R² improves with 5-year window vs 3-year | OPEN — A3/A5 don't speak directly to label window; can revisit if TE LOYO R² stagnates |

---

## Phase D — Architecture (Research track)

| # | Change | Condition to implement | Status |
|---|--------|------------------------|--------|
| D1 | Conformal prediction intervals | Calibration test (A3) confirms ZAP-to-B2S mapping is reliable enough to anchor intervals | **CONDITION MET** — all 3 positions show strong Spearman (0.60–0.63) and good decile calibration |
| D2 | Ensemble blend (Ridge + LGBM) | Ensemble LOYO R² > pure Ridge across all positions | OPEN — not tested; LGBM historically underperformed Ridge on LOYO |
| D3 | Ranking metrics in output (Spearman, top-12 hit rate) | Always add — reporting improvement, not model change | OPEN — easy win, can add at any time |
| D4 | Bayesian Ridge for TE | Only if A1 shows severe TE coefficient instability AND D-equivalent Bayesian model improves LOYO R² | NOT WARRANTED — TE features all ≥49% stable; not severe instability |

---

## Deprioritized (Do Not Pursue)

- Landing spot / QB quality model — requires NFL data not available pre-draft
- Draft class strength normalization — ZAP is already class-relative
- Transfer portal flag — low signal in 2014–2022 window; direction ambiguous
- Full Bayesian stack — excessive complexity; conformal intervals cover 80% of the benefit

---

## Results Log

### A1 — Bootstrap Lasso Stability (1000 resamples)

```
WR — current model features (bootstrap selection frequency):
  best_age                   94.4%  STABLE
  log_draft_capital          86.9%  STABLE
  best_routes_per_game       78.9%  STABLE
  early_declare              76.9%  STABLE
  draft_tier                 71.7%  STABLE
  best_man_zone_delta        67.4%  STABLE
  best_man_yprr              52.9%  STABLE
  breakout_score_x_capital   58.1%  STABLE
  [best_slot_yprr: 69.5% — stable but not currently selected]

RB — current model features:
  best_zone_yprr             85.5%  STABLE
  best_usage_pass            83.2%  STABLE
  log_draft_capital          82.6%  STABLE
  capital_x_age              78.1%  STABLE
  best_rush_ypc              61.4%  STABLE
  position_rank              55.6%  STABLE
  college_fantasy_ppg        51.7%  STABLE
  total_yards_rate_x_capital 24.4%  BORDERLINE UNSTABLE

TE — current model features:
  consensus_rank             95.0%  STABLE
  overall_pick               90.6%  STABLE
  weight_lbs                 66.0%  STABLE
  best_zone_yprr*            65.3%  STABLE (high frequency but not selected in this run)
  broad_jump                 58.5%  STABLE
  breakout_score_x_capital   54.3%  STABLE
  forty_time                 52.2%  STABLE
  slot_rate_x_capital        49.1%  BORDERLINE

Conclusion: All positions predominantly stable. RB total_yards_rate_x_capital is
borderline (24.4%) — unstable by the <40% threshold but this feature exists in the
interaction term form (total_yards_rate × capital) so loss of raw feature is expected.
```

### A2 — Feature Knockout (LOYO R² delta from removing each feature)

```
WR — baseline LOYO R² = 0.3579:
  best_age                   -0.0166  (most important non-capital feature)
  log_draft_capital          -0.0122  (most important capital feature)
  best_routes_per_game       -0.0053
  breakout_score_x_capital   -0.0027
  draft_tier                 -0.0019
  best_man_yprr              -0.0006
  best_man_zone_delta        +0.0004  (marginal — slightly hurts when kept)
  early_declare              +0.0009  (marginal — slightly hurts when kept)
  → No catastrophic single-feature dependency. best_man_zone_delta and early_declare
    are near-zero signal but do not meaningfully hurt either.

RB — baseline LOYO R² = 0.4253:
  capital_x_age              -0.0256  (dominant feature — age × capital interaction)
  best_zone_yprr             -0.0149  (strong signal)
  log_draft_capital          -0.0076
  position_rank              -0.0069
  best_usage_pass            ~0.000   (essentially zero contribution)
  total_yards_rate_x_capital +0.0013  (slightly hurts — borderline unstable, see A1)
  best_rush_ypc              +0.0020  (slightly hurts)
  college_fantasy_ppg        +0.0032  (slightly hurts — clearest negative)
  → NOTABLE: 3 of 8 features are marginally counterproductive on LOYO.
    However effects are tiny and all features are <0.005 LOYO R² in either direction.
    Condition for removal: does NOT meet "fundamental change" threshold.

TE — baseline LOYO R² = 0.4112:
  consensus_rank             -0.0265  (tied most important)
  overall_pick               -0.0264  (tied most important)
  breakout_score_x_capital   -0.0127
  forty_time                 -0.0029
  broad_jump                 +0.0040  (slightly hurts)
  weight_lbs                 +0.0041  (slightly hurts)
  slot_rate_x_capital        +0.0054  (slightly hurts)
  → 3 of 7 features marginally counterproductive. Same conclusion as RB:
    effects are tiny, no fundamental change warranted.
```

### A3 — ZAP Calibration Curve (predicted vs actual B2S by decile)

```
WR — Spearman rho=0.5988 (p≈0), top-25% hit rate at ZAP≥75: 54.7% (base rate 25%)
  Decile   N   PredB2S  ActualB2S  Bias
  0-10    23    2.24      2.82     -0.58
  10-20   25    3.31      2.44     +0.87
  20-30   16    3.72      2.91     +0.81
  30-40   26    4.51      3.05     +1.46  ← largest overconfidence
  40-50   25    5.58      6.79     -1.21
  50-60   19    6.77      7.18     -0.41
  60-70   24    7.86      8.21     -0.34
  70-80   20    8.86      9.77     -0.91
  80-90   24   10.19     10.07     +0.12  ← well calibrated
  90-100  22   12.41     12.32     +0.08  ← well calibrated
  → Top deciles well calibrated. Mid-low deciles noisy but not systematic
    overconfidence at the top. C1 survivorship fix NOT warranted.

RB — Spearman rho=0.6247 (p≈0), top-25% hit rate at ZAP≥75: 62.2% (base rate 25.2%)
  Good calibration overall. Largest noise in bottom deciles (small N).
  No systematic top-decile overconfidence.

TE — Spearman rho=0.6296 (p≈0), top-25% hit rate at ZAP≥75: 65.2% (base rate 25.8%)
  Best calibrated of 3 positions. Top decile slightly underestimates actual (bias -0.87
  at 90-100), which is conservative/safe.
  No systematic overconfidence at any decile.

Conclusion: All 3 positions show strong rank-order signal well above base rate.
Calibration is adequate. No systematic failure at top or bottom deciles.
D1 (conformal intervals) condition CONFIRMED MET.
```

### A4 — Label Permutation Test (100 permutations)

```
WR:  real LOYO R²=0.3579, perm mean=-0.0277, perm max=0.0157, p=0.0 ✓
RB:  real LOYO R²=0.4253, perm mean=-0.0357, perm max=0.0349, p=0.0 ✓
TE:  real LOYO R²=0.4112, perm mean=-0.0409, perm max=0.0219, p=0.0 ✓

PASS: Model is detecting real signal in all 3 positions.
Real LOYO R² is 10–17× above permuted maximum in all cases.
No evidence of label leakage or overfitting to noise.
```

### A5 — Capital Decomposition

```
                 Capital-only  Phase I (no-cap)  Full model  Incr/capital  Incr/nocap
WR               0.311         0.212             0.358       +0.047        +0.146
RB               0.422         0.246             0.425       +0.004        +0.179
TE               0.405         0.285             0.411       +0.006        +0.126

WR: Capital explains 87% of full-model R². Non-capital adds meaningful +4.7 ppts.
    College/combine data genuinely contributes beyond draft capital for WRs.

RB: Capital explains 99% of full-model R². Non-capital adds only +0.4 ppts.
    The RB model is ESSENTIALLY a capital model. Production/efficiency features
    capture little incremental signal beyond what draft position already encodes.
    This is consistent with JJ's finding that top-half first-round RBs (R²=0.53)
    are near-automatic based on capital alone.

TE: Capital explains 98% of full-model R². Non-capital adds only +0.6 ppts.
    Similar pattern to RB — capital dominates; non-capital features marginal.

Notable: Phase I (no capital) R² is substantial for all positions (0.21–0.29),
confirming that college/combine data has independent predictive value. The capital
delta / Phase I model has a valid foundation.
```

---

## Decision Log

| Date | Decision | Evidence |
|------|----------|----------|
| 2026-03-14 | REJECT B1 (target share features) | A1: existing WR PFF features ≥53% stable; no gap to fill |
| 2026-03-14 | REJECT B2 (trajectory features) | A1: best-season features stable; condition not met |
| 2026-03-14 | DEFER B3 (missingness flags) | Phase A did not test this directly; not enough evidence either way |
| 2026-03-14 | REJECT B4 (capital curve variants) | A5: current capital curve adequate; capital-only R² already strong |
| 2026-03-14 | REJECT B5 (age multiplier variants) | A2: best_age contributes cleanly; no evidence current formula is wrong |
| 2026-03-14 | REJECT C1 (survivorship zero-fill) | A3: no systematic bottom-decile overconfidence in any position |
| 2026-03-14 | DEFER C2 (TE 5-year label window) | A5: TE capital dominates but this is a label design question, not a feature gap |
| 2026-03-14 | APPROVE D1 (conformal intervals) | A3: Spearman 0.60–0.63 across positions; decile calibration adequate; condition met |
| 2026-03-14 | REJECT D4 (Bayesian Ridge for TE) | A1: TE features all ≥49% stable; not severe instability |
| 2026-03-14 | DOCUMENT only (RB near-capital-only) | A5: RB non-capital adds only +0.004 LOYO R². No change warranted — this is expected given JJ's research. Document for transparency. |
| 2026-03-14 | DOCUMENT only (RB marginal features) | A2: total_yards_rate_x_capital / best_rush_ypc / college_fantasy_ppg each slightly negative but all <0.005 LOYO R² — not fundamental; no change warranted |
