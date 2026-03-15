# Feedback1
## External Review of Dynasty Prospect Model

Author of reviewed model: Martin Ozimek  
Model: Dynasty Prospect Model v1.0  
Reviewer: AI Technical Review  

---

# Executive Summary

The dynasty prospect model demonstrates a **strong statistical foundation and thoughtful architecture**.

Particularly strong components include:

- Leave-One-Year-Out cross-validation (correct temporal validation)
- Explicit separation of **capital vs non-capital signals** via the Phase I framework
- Good production and efficiency metrics
- Regularization strategy (Lasso selection → Ridge prediction)
- Clean reproducibility and data documentation

Overall structural quality is high and comparable to strong academic sports analytics work.

However, several areas present opportunities for improvement:

- Target label construction
- Production feature selection
- Modeling assumptions
- Missing opportunity-share variables
- Small-sample stability

Estimated potential improvement after upgrades:

WR LOYO R²  
0.38 → ~0.43–0.46

---

# Major Structural Weaknesses

## 1. Target Label: Survivorship Bias

Current label:

**B2S (Best Two of First Three Seasons)** with minimum 8 games played.

Players with fewer than two qualifying seasons are removed from training.

### Issue

This creates **survivorship bias**.

Players who:

- never earn snaps
- leave the league quickly
- suffer early injuries

are excluded entirely.

The model therefore learns:

> Predict production **conditional on surviving**

instead of:

> Predict probability of fantasy success.

This particularly biases evaluation of:

- Day 3 WRs
- developmental TEs
- rotational RBs

### Suggested Alternatives

**Option A — Zero-Filled B2S**

Treat non-qualifying seasons as **0 PPG** instead of removing the player.

**Option B — Hit Rate Target**

Binary classification:


Hit = 1 if any season ≥ 12 PPG
Hit = 0 otherwise


**Option C — Peak Production**


max_ppg_first3years


This avoids survivorship bias while retaining signal.

---

## 2. Age Multiplier in Breakout Score

Current formula:


age_mult = max(0, 26 − age)


Example:

Age | Multiplier
--- | ---
19 | 7
20 | 6
21 | 5
22 | 4

This implies a 19-year-old breakout is **75% more valuable than a 22-year-old breakout**.

### Issue

This scaling is likely too aggressive.

Historical research shows the age effect is **nonlinear and diminishing**, not linear.

### Suggested Replacement

Example smoother functions:


age_weight = exp(-0.25*(age-20))


or


age_weight = 1 / (1 + 0.5*(age-20))


These preserve early-breakout advantages without extreme scaling.

---

## 3. Draft Capital Decay Constant

Current capital score:


DraftCapital = 100 * exp(-0.023*(pick − 1))


### Issue

The decay constant (0.023) is **not estimated from data**.

This imposes an assumed capital value curve rather than learning it from historical outcomes.

### Suggested Fix

Estimate the relationship empirically:

Fit


E[B2S | pick]


Alternatively use simpler variables:


log(pick)
draft_round
pick_percentile


Empirically estimated curves typically outperform hand-specified ones.

---

## 4. Best-Season Feature Construction

Most features use **only the best season**.

Examples:

- best_yprr
- best_breakout_score
- best_routes_per_game

### Issue

This discards information about:

- trajectory
- consistency
- development curve

Example:

Player A (gradual improvement)

FR: 1.2 YPRR  
SO: 1.4  
JR: 1.6  
SR: 1.8  

Player B (single spike)

FR: 0.4  
SO: 3.2  
JR: 0.5  

Best-season metrics treat them similarly.

They should not be treated similarly.

### Suggested Additions


career_avg_yprr
top2_avg_yprr
yprr_growth
routes_growth
target_share_growth


Trajectory features are often extremely predictive.

---

## 5. Combine Metric Imputation

Missing combine metrics are median-imputed.

### Issue

Combine missingness is **not random**.

Players skipping drills are often:

- elite prospects preserving draft position
- poor testers avoiding exposure
- injured prospects

Median imputation merges these groups incorrectly.

### Suggested Fix

Add missingness indicators:


forty_missing
agility_missing
vertical_missing


These often improve model performance.

---

## 6. Sample Size vs Feature Pool

Training sizes:

Position | N
---|---
WR | 224
RB | 147
TE | 97

Candidate features approach ~40.

This creates **high-variance conditions**.

### Suggested Alternative

Use **ElasticNet** instead of two-stage:


Lasso → Ridge


ElasticNet handles correlated predictors more smoothly.

---

## 7. Phase I Independence Limitations

Phase I removes draft capital.

However, several retained metrics (e.g., YPRR) correlate strongly with capital.

Thus Phase I represents:

> A no-explicit-capital model

not

> A fully independent evaluation.

This limitation is unavoidable but worth acknowledging.

---

# Major Missing Predictor

## Target Share

One of the strongest predictors in WR prospect modeling is **target share**.


target_share = targets / team_pass_attempts


### Why It Matters

Target share measures **offensive role and coaching trust**.

Example:

Player | Targets | Team Attempts | Share
---|---|---|---
A | 90 | 550 | 16%
B | 90 | 350 | 26%

Player B historically has a far higher NFL hit rate.

Yardage metrics miss this distinction.

### Recommended Features


best_target_share
career_target_share
target_share_growth


### Interaction Feature


target_share_x_capital


This would likely compete strongly with breakout_score_x_capital.

---

# Additional Feature Ideas

## Weighted Target Share

Example:


weighted_target_share =
0.6 * final_year_share +
0.3 * junior_year_share +
0.1 * sophomore_year_share


Captures development trajectory.

---

## Trajectory Features


yprr_slope
target_share_growth
routes_growth


These signals are frequently strong predictors.

---

# Model Output Improvements

## Prediction Uncertainty

Current outputs are point estimates only.

Adding bootstrap prediction intervals would improve risk evaluation.

Example:


Predicted B2S: 13.5 PPG
95% interval: [7.8, 18.9]


This is especially valuable for dynasty decision-making.

---

# Tight End Model Risk

TE sample size:


N ≈ 97


Many predictors are combine metrics.

Risk:

The model may largely learn:


fast tight ends succeed


Which is already widely known.

More production-based features or larger sample sizes may help.

---

# Additional Potential Improvements

## Quarterback Quality Adjustment

College production could be adjusted for QB quality:

Examples:


team_pass_efficiency
QB draft capital
team passing EPA


---

## Draft Class Strength Normalization

Some draft classes are unusually strong.

Normalizing prospect scores relative to class distribution may improve stability.

---

# Overall Assessment

Category | Evaluation
---|---
Conceptual Design | Excellent
Data Engineering | Excellent
Feature Engineering | Good
Statistical Assumptions | Moderate
Target Variable | Needs Improvement

Overall Model Quality:

**Strong (~8/10)**

With several improvements capable of significantly raising predictive performance.

---

# Highest Priority Improvements

1. Add **target share features**
2. Remove survivorship bias from target variable
3. Replace **linear age multiplier**
4. Add **trajectory features**
5. Estimate **draft capital curve empirically**

These changes are the most likely to produce large performance gains.