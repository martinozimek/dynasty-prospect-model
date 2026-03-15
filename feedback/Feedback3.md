# Feedback3 --- Dynasty Prospect Model Review

Author: ChatGPT Date: 2026‑03‑14

This document compiles a full technical review of the dynasty prospect
model methodology, assumptions, and validation framework. The goal is to
provide targeted questions and diagnostics that interrogate the model's
validity, stability, and predictive reliability.

The feedback is organized into the following sections:

1.  Overall structural assessment
2.  High‑priority model interrogation questions
3.  Feature engineering critiques
4.  Data and label risks
5.  Validation and robustness tests
6.  Model improvement suggestions
7.  Key diagnostics checklist

------------------------------------------------------------------------

# 1. Overall Structural Assessment

The model architecture is strong relative to most public dynasty
prospect models. Several methodological choices significantly increase
credibility:

Strengths

• Temporal validation using Leave‑One‑Year‑Out (LOYO) cross‑validation\
• Lasso feature selection followed by Ridge regression for stability\
• Explicit separation of capital and non‑capital models (Phase I
framework)\
• Integration of PFF route‑efficiency metrics\
• Explicit assumption inventory and transparent methodology
documentation

These design choices address many common modeling pitfalls such as
random data splits, hidden leakage, and opaque feature engineering.

However, three structural risks remain:

1.  Draft capital signal dominance\
2.  Feature instability due to small sample sizes\
3.  Label noise driven by opportunity rather than talent

The following sections describe diagnostics that should be run to
evaluate these risks.

------------------------------------------------------------------------

# 2. High‑Priority Interrogation Questions

## 2.1 Draft Capital Dependence

Test how much predictive signal comes from draft capital alone.

Diagnostic:

Train three models using LOYO validation:

1.  Capital‑only features\
2.  Non‑capital features only\
3.  Full model

Compare R² values.

If the capital‑only model explains most of the predictive power, the
model is primarily capturing NFL team evaluation rather than discovering
independent signals.

------------------------------------------------------------------------

## 2.2 Interaction Term Audit

The strongest predictors in the current system involve capital
interactions.

Examples:

• breakout_score_x_capital\
• capital_x_age

Test the necessity of these terms.

Diagnostic:

Retrain the model excluding all interaction terms and compare LOYO R².

Possible outcomes:

• Minimal performance drop → interactions unnecessary\
• Large performance drop → interactions capture important nonlinear
relationships

------------------------------------------------------------------------

## 2.3 Lasso Feature Stability

Small datasets combined with correlated features can cause Lasso
selection instability.

Diagnostic:

Bootstrap the training dataset 1000 times and record which features are
selected by Lasso.

Example output:

  feature                    selection frequency
  -------------------------- ---------------------
  log_draft_capital          100%
  breakout_score_x_capital   85%
  best_man_yprr              42%
  best_routes_per_game       35%

Features appearing in fewer than \~40% of bootstrap runs are likely
unstable predictors.

------------------------------------------------------------------------

## 2.4 Era Stability

The training window assumes statistical stationarity from 2014--2022.

However college football has changed significantly during this period
due to:

• spread offenses • increased passing rates • transfer portal effects

Diagnostic:

Train on 2014‑2018 and test on 2019‑2022, then reverse.

Compare performance and feature importance shifts.

------------------------------------------------------------------------

## 2.5 Outcome Definition Sensitivity

The B2S metric assumes that early career peak performance best captures
dynasty value.

However fantasy PPG is strongly influenced by opportunity.

Diagnostic:

Re‑run the pipeline using alternative labels:

• Peak season PPG\
• Top‑24 season indicator\
• Career WAR proxy\
• First‑contract snap share

Compare model stability across labels.

------------------------------------------------------------------------

## 2.6 Opportunity Bias Test

Fantasy production strongly correlates with playing time.

Diagnostic:

Measure correlations between:

• predicted B2S • draft capital • average snap share in first three
seasons

If predicted success correlates strongly with snap share, the model
primarily predicts opportunity allocation.

------------------------------------------------------------------------

## 2.7 Best‑Season Selection Bias

The model selects the "best season" using receiving yards per team pass
attempt for all positions.

This biases RB evaluation toward receiving backs.

Diagnostic:

Compare predictive power using three selection methods:

• current best season • most recent season • career weighted average

------------------------------------------------------------------------

## 2.8 Missing Data Bias

Median imputation assumes missing combine data is random.

In reality:

• elite prospects often skip testing • poor testers hide results

Diagnostic:

Compare distributions of combine metrics for players with missing versus
recorded measurements.

If distributions differ significantly, median imputation introduces
bias.

------------------------------------------------------------------------

## 2.9 Archetype Blind Spot Analysis

Prospect models often fail for specific player archetypes.

Examples:

WR • contested catch specialists • slot receivers

RB • receiving backs • committee backs

TE • low‑production elite athletes

Diagnostic:

Cluster prospects using k‑means on size, athleticism, and production
features and analyze prediction errors per cluster.

------------------------------------------------------------------------

## 2.10 Out‑of‑Distribution Detection

Extreme athletic profiles cause model extrapolation.

Diagnostic:

Compute Mahalanobis distance of each prospect's feature vector from the
training distribution.

Flag players beyond the 95th percentile as low‑confidence predictions.

------------------------------------------------------------------------

## 2.11 Calibration Analysis

ZAP scores assume percentile predictions correspond to actual
performance distributions.

Diagnostic:

Group historical players by predicted ZAP decile and compare predicted
versus actual B2S averages.

If top deciles systematically overpredict outcomes, calibration
adjustments are needed.

------------------------------------------------------------------------

## 2.12 Historical Miss Analysis

The most informative diagnostic is studying model failures.

Diagnostic:

Identify the 20 largest historical prediction errors and analyze their
feature profiles.

Questions to answer:

• Which features drove the incorrect predictions? • Do certain
archetypes consistently produce errors?

------------------------------------------------------------------------

# 3. Feature Engineering Critiques

## Breakout Score Reconstruction

The breakout score formula was reconstructed from a podcast description
rather than verified algebraically.

Testing variants:

• rec_rate only • rec_rate + age adjustment • rec_rate + SOS adjustment
• full formula

This tests whether each component contributes predictive value.

------------------------------------------------------------------------

## Draft Capital Curve

The exponential decay constant used in the draft capital formula was not
estimated from data.

Recommendation:

Fit an optimal decay parameter directly using regression against B2S
outcomes.

------------------------------------------------------------------------

## Recruiting Rating Independence

Recruiting ratings are assumed independent from draft capital.

However recruiting rank likely influences:

• college opportunity • draft capital

Diagnostic:

Compute correlation between recruit_rating and draft capital.

------------------------------------------------------------------------

# 4. Data and Label Risks

Several structural data limitations remain.

Major risks include:

• Small TE sample size • PFF coverage inconsistencies across eras •
Median imputation bias • Approximate age estimation • Use of team pass
attempts as RB denominator • Potential breakout formula inaccuracies

Each of these should be stress‑tested via sensitivity analysis.

------------------------------------------------------------------------

# 5. Validation Improvements

Additional validation approaches that would strengthen the model:

1.  Bootstrapped cross‑validation distributions
2.  Out‑of‑distribution detection for extreme prospects
3.  Calibration curves for predicted percentiles
4.  Rolling‑window forward prediction tests

These diagnostics ensure that the model generalizes rather than
memorizes historical patterns.

------------------------------------------------------------------------

# 6. Potential Model Improvements

Two structural upgrades could significantly improve model robustness.

## Bayesian Ridge Regression

Advantages:

• naturally handles small sample sizes • produces uncertainty estimates
• stabilizes coefficient estimates

------------------------------------------------------------------------

## Conformal Prediction Intervals

Conformal prediction would allow the model to produce prediction
intervals for B2S outcomes.

Example output:

ZAP = 82\
Expected B2S = 12.3 PPG\
80% prediction interval = \[7.8, 15.6\]

This improves interpretability and communicates uncertainty to users.

------------------------------------------------------------------------

# 7. Key Diagnostic Checklist

The following tests should be run before finalizing the model.

1.  Capital signal decomposition
2.  Interaction term necessity test
3.  Lasso feature stability bootstrap
4.  Era stability validation
5.  Label sensitivity analysis
6.  Opportunity bias test
7.  Best‑season selection comparison
8.  Missing data bias test
9.  Archetype clustering error analysis
10. Out‑of‑distribution detection
11. ZAP calibration curves
12. Largest historical miss analysis

If these diagnostics confirm stability, the model can be considered
methodologically robust.

------------------------------------------------------------------------

End of Feedback3.
