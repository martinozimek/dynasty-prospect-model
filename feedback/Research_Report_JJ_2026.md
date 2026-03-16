# Research Report — JJ 2026 Pre-Draft Guide Analysis
**Date**: 2026-03-15
**Source**: JJ Zachariason, *Late Round Podcast 2026 Pre-Draft Prospect Guide*
**Purpose**: Extract new methodology insights from JJ's 2026 guide, test candidate features against our training data via LOYO-CV, and produce adoption/rejection recommendations.

---

## 1. Methodology Insights from the 2026 Guide

### 1.1 New ZAP Scaling (2026 Change)

JJ changed from a pure percentile-based ZAP (0–100 percentile rank) to a **talent-gap-preserving scale** that uses absolute score distances rather than ranked buckets. The practical effect: a cluster of similar prospects no longer gets spread across a wide percentile range just because they are adjacent in rank.

**2026 ZAP Categories:**

| Category | Range |
|---|---|
| Legendary Performer | 90–100 |
| Elite Producer | 75–90 |
| Weekly Starter | 60–75 |
| Flex Play | 40–60 |
| Benchwarmer | 30–40 |
| Waiver Wire Add | 20–30 |
| Dart Throw | 0–20 |

This aligns with our own 2026 ORBIT approach (we already moved away from strict percentile scaling). No model change required.

---

### 1.2 WR Model (2026)

JJ's WR model has not materially changed from 2025. Key inputs remain:
- Draft capital (primary predictor)
- Breakout Score (age + SOS adjusted receiving production)
- Adjusted best-season fantasy PPG (age + SOS adjusted)
- College conference (Power 4 flag)
- Teammate Score (sum of college teammate draft capital)

**New 2026 WR input**: adjusted best-season fantasy PPG — the player's single best college season's fantasy PPG, multiplied by an age adjustment and a SOS (SP+) multiplier. This is JJ's attempt to capture peak production ceiling rather than career average.

**JJ on WR athleticism**: explicitly states athleticism is **not important** for WRs. Speed Score, 40 time, and combine metrics are excluded from his WR model. Our model similarly found forty_time, speed_score, and broad_jump all non-selected by Lasso for WR.

---

### 1.3 RB Model (2026)

**New 2026 RB input**: *total yards per team play* — rushing yards plus receiving yards divided by total team offensive plays (not team pass attempts). This is meaningfully different from our `best_total_yards_rate` which uses team pass attempts in the denominator.

JJ's denominator (total team plays) removes the advantage held by pass-heavy offenses and better captures true usage within any offensive system. Our denominator (team pass attempts) slightly biases in favor of pass-heavy offenses.

**Draft capital blend**: For RBs, JJ uses an 83/17 blend of actual pick capital and consensus mock draft capital. This reduces the influence of positional devaluation (late-round RBs can still have high consensus mock scores).

---

### 1.4 TE Model (2026) — Most Substantive Update

JJ's TE model is the most developed new methodology in the 2026 guide. It is organized around **three interacting areas**:

**Area 1 — Draft Capital**
Primary predictor as in other positions. However, the *meaning* of capital shifts based on athlete profile:
- **Athletic TE + strong capital**: capital confirming what film already shows
- **Less athletic TE + strong capital**: capital is *more* informative — the team has seen route-running, spatial awareness, and translatable skills that athleticism testing cannot capture
- **Athletic TE without production**: penalized — athleticism without college output is a red flag at TE

**Area 2 — Athleticism (Height-Adjusted Speed Score)**
JJ applies a height adjustment to traditional Speed Score to reward tall TEs with good speed:

> `Height-Adjusted Speed Score = Speed Score × (height_inches / 74)^k`

JJ does not disclose the exponent `k`. From known values in the 2026 guide:
- Sam Roush (6'6", 267 lbs, 4.70 40): Height-Adj SS = 112.3
- Jack Endries (6'5", 245 lbs, 4.62 40): Height-Adj SS = 116.0

Our calibration using `k=1.5` recovers approximately correct ordering.

**Area 3 — Production**
- Breakout Score (same formula as WR: age + SOS adjusted receiving yards/team pass att)
- Best-season yards per team pass attempt (adjusted)
- Height and weight independently (heavier = bonus for run-blocking / YAC; height = separation opportunity)

**Key Hunter Henry vs Michael Mayer insight** (both pick 35, similar profiles):
- Henry (6'5", 250, 4.68 → height-adj SS ~96): ZAP 85.6
- Mayer (6'5", 249, 4.66 → height-adj SS ~103): ZAP 73.0

Counterintuitively, **lower athleticism + same draft capital = higher score** because capital's signal is more reliable when athleticism isn't driving the pick. Teams that use a premium pick on a less-athletic TE have seen something else they value highly.

---

### 1.5 Teammate Score

JJ uses a weighted sum of college teammate draft capital as an input to all three position models. The idea: players who dominated alongside drafted teammates were producing against real NFL-level coverage / with reduced defensive attention, both of which deflate traditional production metrics.

Our training data has `teammate_score` at **100% coverage** for all positions (224/224 WR, 199/199 RB, 126/126 TE rows).

---

### 1.6 Draft Capital Delta

JJ's Draft Capital Delta is defined as:
> `ZAP Score − Draft Capital Score (0–100 percentile of pick)`

Pre-draft: uses projected pick from NFLMockDraftDatabase.com consensus.
Post-draft: uses actual pick.

JJ's own framing (2025 guide): "Draft Capital Delta can help us find gems, but remember it's not the key metric we should be working off of. ZAP score will always be the top priority."

Our Phase I no-capital model produces a richer version of this concept: delta = ORBIT − Phase1_ORBIT (full model with no capital signal vs full model with capital).

---

## 2. Empirical Experiment Results (LOYO-CV)

All experiments run via `scripts/jj_feature_experiments.py`. Baseline R² from `models/metadata.json`. Each experiment adds candidate features to the existing selected feature set and re-runs RidgeCV LOYO-CV.

**Baselines (metadata.json):**
- WR: LOYO R² = 0.3563
- RB: LOYO R² = 0.4131
- TE: LOYO R² = 0.4034

---

### 2.1 WR Experiments

| Experiment | LOYO R² | Delta | Spearman | Decision |
|---|---|---|---|---|
| E1: +teammate_score | 0.3516 | −0.0047 | 0.5955 | ✗ Reject |
| E2: +teammate_score_x_capital | 0.3394 | −0.0169 | 0.5896 | ✗ Reject |
| E3: +adj_fantasy_ppg (age+SOS) | 0.3455 | −0.0108 | 0.5813 | ✗ Reject |
| E4: +top_season_ppg_adj | 0.9012 | +0.5449 | 0.9423 | ✗ **DATA LEAKAGE** |
| E5: +log_pick_capital (JJ curve) | 0.3495 | −0.0068 | 0.5957 | ✗ Reject |
| E6: +height_adj_speed_score | 0.3562 | −0.0001 | 0.5967 | ✗ Reject |
| E7: all new WR features | 0.3322 | −0.0241 | 0.5811 | ✗ Reject |

**WR conclusion**: No new JJ features provide signal beyond our current WR feature set. `adj_fantasy_ppg` adds nothing beyond what `best_routes_per_game` + `best_breakout_score` (via `breakout_score_x_capital`) already capture. Height-adjusted Speed Score is flat (near-zero delta), consistent with JJ's own view that WR athleticism is not predictive.

---

### 2.2 RB Experiments

| Experiment | LOYO R² | Delta | Spearman | Decision |
|---|---|---|---|---|
| E1: +teammate_score | 0.4688 | **+0.0557** | 0.6571 | ✓ Candidate |
| E2: +teammate_score_x_capital | 0.4700 | **+0.0569** | 0.6581 | ✓ Candidate |
| E3: +speed_score | 0.4159 | +0.0028 | 0.6239 | ~ Marginal |
| E4: +height_adj_speed_score | 0.4131 | +0.0000 | 0.6222 | ✗ Reject |
| E5: +total_plays_adj_yards_rate | 0.4280 | **+0.0149** | 0.6323 | ✓ Candidate |
| E6: +log_pick_capital (JJ curve) | 0.4815 | **+0.0684** | 0.6642 | ✓ Candidate (strongest) |
| E7: all new RB features | 0.4957 | **+0.0826** | 0.6761 | ✓ Candidate |

**RB conclusion**: Four features show meaningful positive signal. The strongest — `log_pick_capital` — suggests JJ's raw pick capital curve outperforms our current `log_draft_capital` formulation for RBs. `teammate_score` and its capital interaction both add ~+0.057. Including all new RB features simultaneously produces +0.083 combined improvement, suggesting complementary (not collinear) contributions.

---

### 2.3 TE Experiments

| Experiment | LOYO R² | Delta | Spearman | Decision |
|---|---|---|---|---|
| E1: +teammate_score | 0.3919 | −0.0115 | 0.6126 | ✗ Reject |
| E2: +teammate_score_x_capital | 0.3882 | −0.0152 | 0.6071 | ✗ Reject |
| E3: +height_adj_speed_score | 0.3890 | −0.0144 | 0.6018 | ✗ Reject |
| E4: +height_ss_x_capital | 0.3908 | −0.0126 | 0.6084 | ✗ Reject |
| E5: +log_pick_capital | 0.3956 | −0.0078 | 0.6232 | ✗ Reject |
| E6: +speed_score (raw) | 0.3986 | −0.0048 | 0.6217 | ✗ Reject |
| E7: +height_inches alone | 0.3936 | −0.0098 | 0.6049 | ✗ Reject |
| E8: +speed+height (JJ TE proxy) | 0.3876 | −0.0158 | 0.6014 | ✗ Reject |
| E9: height_adj_ss + x_cap | 0.3851 | −0.0183 | 0.6013 | ✗ Reject |
| E10: all new TE features | 0.3766 | −0.0268 | 0.5936 | ✗ Reject |

**TE conclusion**: All JJ TE features reduce model performance. Height-adjusted Speed Score — JJ's central TE athleticism construct — consistently hurts the model (−0.0144 alone, −0.0183 with capital interaction). This is counterintuitive given JJ's detailed methodology, but consistent with our existing TE model which already includes `forty_time`, `broad_jump`, and `agility_score` as selected features. JJ's height adjustment may be capturing a signal that our model already encodes through the combination of raw combine metrics + weight.

The N=126-127 TE sample also limits our ability to detect subtle composite features — JJ's guide likely has access to a longer training window.

---

## 3. Critical Finding: `top_season_ppg` Data Leakage

**Severity: CRITICAL — must remove from training data**

The column `top_season_ppg` in `data/training_WR.csv` (and likely RB/TE CSVs) is **post-outcome data**:

- **Correlation with b2s_score**: 0.983
- **Verification**: Odell Beckham's `top_season_ppg` = 24.75 = exactly his `year1_ppg` (NFL Year 1 PPG)
- **Mechanism**: `top_season_ppg` appears to be the player's best NFL season PPG among years 1–3, which is directly derived from the B2S label computation

When `top_season_ppg_adj` was included in WR E4, LOYO R² jumped to 0.9012 — a nonsensical result that confirmed the leakage. This column must be **removed from all training CSVs** and from any feature candidate lists.

**Action item**: Audit `build_training_set.py` to identify where `top_season_ppg` is joined in. If it comes from the NFL outcomes table, exclude it from the training feature export. The column name is deceptive — "top season PPG" sounds like a college metric but is an NFL outcome.

---

## 4. JJ ZAP vs Our ORBIT Score Comparison (2026 Pre-Draft)

| Player | JJ ZAP | Our ORBIT | Delta | Note |
|---|---|---|---|---|
| **WR** | | | | |
| Jordyn Tyson | 92.2 | 85.7 | −6.5 | JJ higher |
| Carnell Tate | 84.2 | 92.4 | +8.2 | We score higher |
| Makai Lemon | 80.2 | 93.3 | +13.1 | We score substantially higher |
| KC Concepcion | 73.4 | N/A | — | Not in our DB |
| Omar Cooper | 66.3 | N/A | — | |
| Denzel Boston | 66.1 | N/A | — | |
| Chris Brazzell | 61.6 | N/A | — | |
| Germie Bernard | 60.4 | N/A | — | |
| Elijah Sarratt | 57.4 | N/A | — | |
| Zachariah Branch | 57.0 | N/A | — | |
| **RB** | | | | |
| Jeremiyah Love | 93.9 | 97.3 | +3.4 | Aligned at top |
| Emmett Johnson | 69.7 | 83.0 | +13.3 | We score higher |
| Jadarian Price | 64.2 | N/A | — | |
| Mike Washington | 61.0 | N/A | — | |
| Jonah Coleman | 58.7 | 68.0 | +9.3 | We score higher |
| Nick Singleton | 51.0 | N/A | — | |
| Kaytron Allen | 41.0 | N/A | — | |
| **TE** | | | | |
| Kenyon Sadiq | 97.8 | 99.0 | +1.2 | Aligned at top |
| Eli Stowers | 79.2 | 95.9 | +16.7 | We score substantially higher |
| Max Klare | 56.2 | 82.5 | +26.3 | We score dramatically higher |
| Michael Trigg | 48.8 | N/A | — | |
| Sam Roush | 44.7 | N/A | — | |
| Justin Joly | 40.8 | N/A | — | |

**Pattern analysis:**

- **WR**: Our model and JJ's are broadly aligned on the top player (Tate/Tyson near top). The main divergence is Makai Lemon — JJ scores him at 80.2 (Elite Producer boundary), we score him at 93.3 (Legendary). The gap likely reflects JJ's age/SOS-adjusted PPG input pulling Lemon down (younger players get a boost in JJ's system, but Lemon's raw production may not clear JJ's threshold).

- **RB**: Strong agreement at the top (Love #1 both systems). We systematically score the mid-tier RBs higher (Johnson +13, Coleman +9). This could reflect JJ's total-team-plays denominator being more conservative for pass-heavy offenses.

- **TE**: Our TE model consistently overscores relative to JJ's for mid-tier players. Klare +26.3 and Stowers +16.7 are large divergences. JJ's TE model relies more heavily on the **interaction** between capital and athleticism — a less athletic TE needs stronger capital confirmation to score highly. Our TE model may not be penalizing athleticism-deficient players as aggressively as JJ does. The key experiment suggested by this divergence: test whether the `height_adj_speed_score × capital` interaction can *replace* some existing TE athleticism features rather than adding to them.

---

## 5. Recommended Actions

### 5.1 Immediate — Data Integrity

**Remove `top_season_ppg` from training data.** Audit `build_training_set.py` for the column's origin. This is a post-outcome NFL variable that should never appear in a training feature set. Remove from all position CSVs and from any candidate feature list.

---

### 5.2 Phase C Experiment Candidates — RB Only

The RB results are the strongest signal from this research cycle. Three experiments are worth running as formal LOYO tests:

**Experiment C-RB-1: log_pick_capital (JJ's raw pick curve)**
`log_pick_capital = log(1000 / (overall_pick + 1))`
- LOYO delta: +0.0684 (largest single-feature improvement in this research cycle)
- Our current RB model uses `log_draft_capital = log(draft_capital_score + 1)`, which is an exponential decay of pick. JJ uses raw pick in a different log transform.
- **Test**: replace `log_draft_capital` in RB feature set with `log_pick_capital` and measure LOYO R². If improvement holds, this becomes the new capital encoding for RB.

**Experiment C-RB-2: teammate_score and teammate_score_x_capital**
- LOYO delta: +0.057 (E1), +0.057 (E2)
- 100% coverage in training data — no missingness concern.
- **Test**: add to RB feature candidate pool in Lasso selection. Lasso will determine whether both survive or just one.
- **Caution**: +0.057 improvement with N=199 — estimate the standard error of LOYO R² before claiming signal. Bootstrap the LOYO folds to confirm the delta is above noise floor.

**Experiment C-RB-3: total_plays_adj_yards_rate**
- LOYO delta: +0.015
- Conceptually sound — removes pass-offense bias from our denominator.
- **Test**: recompute using `total team plays` rather than `team_pass_att` as denominator. Requires adding total_team_plays to `cfb-prospect-db`. Determine whether the improvement is from the denominator change or the `(1 - rush_rate)` weighting factor (the current test confounds both).

---

### 5.3 Reject — WR and TE

All JJ WR features (teammate_score, adj_fantasy_ppg, height_adj_speed_score, log_pick_capital) show negative LOYO delta and should not be added to the WR model.

All JJ TE features show negative LOYO delta. However, the **structural insight** from JJ's TE model — that capital's meaning depends on the athlete's profile — is worth preserving as an interpretive lens even without a formal feature addition. When manually evaluating TE prospects, apply the JJ heuristic: a team using a premium pick on a less-athletic TE is signaling strong non-measurable confidence.

---

### 5.4 Monitoring — Score Divergence Investigation

The systematic over-scoring of TE mid-tier players (Stowers +16.7, Klare +26.3) relative to JJ's system warrants investigation. When actual 2026 TE outcomes are available (2027+), compare which system better predicted the true outcomes for this cohort. This is a deferred experiment but should be tracked.

---

## 6. Summary Table

| Finding | Status | Priority |
|---|---|---|
| `top_season_ppg` is data leakage — remove | **Critical** | Immediate |
| RB `log_pick_capital` — strong positive signal (+0.068) | Phase C experiment | High |
| RB `teammate_score` — moderate positive signal (+0.057) | Phase C experiment | Medium |
| RB `total_plays_adj_yards_rate` — marginal positive (+0.015) | Phase C experiment | Low |
| WR new features — all negative or flat | Rejected | Done |
| TE new features — all negative | Rejected | Done |
| JJ TE athleticism × capital interaction — interpretive lens | Conceptual | Ongoing |
| Our TE model overshoots mid-tier vs JJ | Monitor | Deferred |

---

## 7. Research Quality Assessment

This research cycle surface-tested JJ's 2026 innovations against our training data. The LOYO-CV framework appropriately guards against overfitting. Several cautions apply:

1. **Small TE sample (N=126)** means any individual TE feature test has wide confidence intervals. The consistent negative direction across 10 TE experiments suggests genuine absence of signal, not noise.

2. **RB teammate_score** shows the largest legitimate gain (+0.057–0.068). Before adoption, bootstrap the LOYO folds to compute a confidence interval on the improvement. If the CI includes zero, the correct conclusion is "no statistically detectable signal at this sample size."

3. **JJ's TE model may be better calibrated for mid-tier prospects** because he has access to more training data and a different (potentially longer) label window. Our TE model's over-scoring of Stowers and Klare may reflect a true model weakness we cannot identify from the experiments here.

4. **The data leakage finding** (`top_season_ppg`) is the most important output of this research cycle. If this feature was inadvertently informing any prior model diagnostics, those results need to be re-evaluated after the column is removed.

---

*Research conducted by Claude Code. Script: `scripts/jj_feature_experiments.py`. Guide source: JJ Zachariason 2026 Pre-Draft Prospect Guide.*
