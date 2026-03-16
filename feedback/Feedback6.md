# Feedback6 --- Model Stress Test Protocol

Author: ChatGPT Date: 2026-03-14

This document defines a **formal stress-testing protocol** for the
Dynasty Prospect Model. The goal is to evaluate robustness, identify
failure modes, and ensure the model's predictive performance is not the
result of overfitting, unstable feature selection, or structural bias.

The tests below mirror validation procedures commonly used in
professional forecasting systems and sports analytics pipelines.

The protocol is organized into the following sections:

1.  Feature Stability Tests
2.  Model Sensitivity Tests
3.  Data Integrity Tests
4.  Out-of-Distribution Detection
5.  Label Robustness Tests
6.  Calibration and Reliability Tests
7.  Failure Mode Analysis
8.  Reporting Template

------------------------------------------------------------------------

# 1. Feature Stability Tests

Small datasets and correlated predictors can cause unstable feature
selection.

## 1.1 Bootstrap Feature Selection

Procedure:

1.  Bootstrap the training dataset **1000 times**.
2.  Run the full Lasso feature selection process on each sample.
3.  Record which features are selected.

Output:

  Feature                    Selection Frequency
  -------------------------- ---------------------
  log_draft_capital          100%
  breakout_score_x_capital   88%
  best_man_yprr              43%
  best_routes_per_game       37%

Interpretation:

Features selected in **\<40% of bootstrap runs** are unstable and may
represent noise.

------------------------------------------------------------------------

## 1.2 Coefficient Variance Test

Procedure:

1.  Fit the Ridge model on each bootstrap sample.
2.  Record coefficient values.

Metrics:

• mean coefficient • standard deviation • coefficient sign stability

Large variance indicates unstable predictors.

------------------------------------------------------------------------

# 2. Model Sensitivity Tests

## 2.1 Feature Knockout Test

Purpose:

Determine whether the model depends heavily on any single feature.

Procedure:

1.  Train the full model normally.
2.  Retrain the model **removing one feature at a time**.
3.  Measure LOYO R² change.

Output example:

  Removed Feature            LOYO R²
  -------------------------- ---------
  none (baseline)            0.39
  log_draft_capital          0.18
  breakout_score_x_capital   0.34
  best_man_yprr              0.38

Interpretation:

Large drops indicate structural dependence on that feature.

------------------------------------------------------------------------

## 2.2 Interaction Term Sensitivity

Procedure:

1.  Remove all capital interaction terms.
2.  Retrain the model.
3.  Compare predictive performance.

This determines whether interactions are capturing genuine signal or
simply reshaping the capital curve.

------------------------------------------------------------------------

# 3. Data Integrity Tests

## 3.1 Random Feature Injection

Purpose:

Ensure the model is not capturing noise patterns.

Procedure:

1.  Add a randomly generated feature column.
2.  Run Lasso feature selection.

Expected result:

Random features should be selected **≈0% of the time**.

If selected frequently, the model is overfitting.

------------------------------------------------------------------------

## 3.2 Label Permutation Test

Purpose:

Test whether the model detects real signal.

Procedure:

1.  Shuffle B2S labels randomly.
2.  Train the model.

Expected result:

R² should collapse to **≈0**.

If R² remains high, leakage exists.

------------------------------------------------------------------------

# 4. Out-of-Distribution Detection

Extreme prospects can cause model extrapolation.

## 4.1 Mahalanobis Distance Test

Procedure:

1.  Compute Mahalanobis distance for each prospect's feature vector
    relative to the training feature distribution.

Flag prospects exceeding the **95th percentile distance**.

These predictions should be treated as low confidence.

------------------------------------------------------------------------

## 4.2 Feature Boundary Test

Procedure:

For each feature:

1.  Identify training set minimum and maximum values.
2.  Flag prospects exceeding those bounds.

These represent true extrapolation scenarios.

------------------------------------------------------------------------

# 5. Label Robustness Tests

The B2S label captures early fantasy production but may be noisy.

## 5.1 Alternative Label Experiments

Re-run the full modeling pipeline using:

• peak season PPG • top-24 season indicator • career WAR proxy •
first-contract snap share

Compare LOYO R² across labels.

Large performance swings indicate label fragility.

------------------------------------------------------------------------

## 5.2 Opportunity Bias Test

Procedure:

Compute correlation between:

• predicted B2S • draft capital • snap share in first three seasons

High correlation with snap share indicates the model predicts
**opportunity allocation rather than talent**.

------------------------------------------------------------------------

# 6. Calibration and Reliability Tests

## 6.1 ORBIT Calibration Curve

Procedure:

1.  Group prospects by predicted ORBIT decile.
2.  Compute actual B2S averages for each group.

Output:

  ORBIT Bin   Predicted Mean   Actual Mean
  --------- ---------------- -------------
  90--100   14.8             11.3
  80--90    13.6             12.5

If predicted values consistently exceed actual values, the model is
overconfident.

------------------------------------------------------------------------

## 6.2 Residual Distribution Analysis

Procedure:

Analyze distribution of prediction errors.

Metrics:

• mean absolute error • skewness • heavy tails

Heavy tails suggest unpredictable prospect types.

------------------------------------------------------------------------

# 7. Failure Mode Analysis

## 7.1 Largest Prediction Errors

Identify the **20 largest prediction misses**.

For each case record:

• predicted B2S • actual B2S • draft capital • feature vector •
archetype classification

Goal:

Identify systematic blind spots.

Examples might include:

• contested catch receivers • small slot WRs • committee RBs • athletic
low-production TEs

------------------------------------------------------------------------

## 7.2 Archetype Clustering

Procedure:

Cluster players based on:

• size • athleticism • production metrics

Compute model error by cluster.

Clusters with high error reveal structural blind spots.

------------------------------------------------------------------------

# 8. Reporting Template

Each stress test should produce a standardized report.

Suggested structure:

1.  Test description
2.  Diagnostic plots
3.  Key metrics
4.  Interpretation
5.  Recommended action

------------------------------------------------------------------------

# Final Goal

The stress test suite should answer three critical questions:

1.  **Is the model stable?**
2.  **Is the model discovering real signal?**
3.  **Where does the model fail?**

If the model passes these tests, its predictive validity is
significantly more credible.

------------------------------------------------------------------------

End of Feedback6.
