# Feedback5 --- Investigation Plan for Structural Model Upgrades

Author: ChatGPT Date: 2026-03-14

This document outlines a structured investigation plan for two model
upgrades commonly used in professional sports analytics and forecasting
systems:

1.  Hierarchical Bayesian Ridge Regression
2.  Conformal Prediction Intervals

The goal is not to immediately replace the current model architecture,
but to evaluate whether these techniques meaningfully improve predictive
stability, interpretability, and uncertainty communication.

The current model architecture is already strong (Lasso → Ridge with
LOYO validation). These upgrades should be evaluated as extensions
rather than replacements.

------------------------------------------------------------------------

# Upgrade 1: Hierarchical Bayesian Ridge Regression

## Motivation

The current model uses Ridge regression with regularization to stabilize
coefficient estimates in a small‑sample environment. However, Ridge
still produces single point estimates and does not explicitly model
uncertainty in parameter estimates.

Hierarchical Bayesian regression can improve performance in small
datasets by:

• shrinking noisy coefficients toward global priors\
• pooling information across related parameters\
• producing posterior distributions instead of single estimates

This is widely used in sports analytics where sample sizes are small
(e.g., baseball player projections, basketball lineup models, and NFL
EPA models).

------------------------------------------------------------------------

## Key Hypothesis

A hierarchical Bayesian version of the model will:

1.  reduce coefficient instability caused by small N
2.  produce more stable predictions across draft classes
3.  generate uncertainty intervals for predicted outcomes

------------------------------------------------------------------------

## Investigation Plan

### Step 1 --- Establish Baseline

Before implementing a Bayesian model, compute diagnostic baselines from
the current system.

Metrics to record:

• LOYO R² (current Ridge model) • LOYO MAE • coefficient variance across
LOYO folds • prediction variance across LOYO folds

This baseline allows comparison against Bayesian models.

------------------------------------------------------------------------

### Step 2 --- Implement Bayesian Ridge Prototype

Implement a Bayesian version of the existing Ridge model using a
probabilistic framework such as:

• PyMC • Stan • NumPyro

Basic structure:

B2S \~ Normal(Xβ, σ)

β \~ Normal(0, τ²)

τ \~ HalfNormal(σ_prior)

This setup mimics Ridge regression but estimates the shrinkage parameter
from the data.

Compare results to classical Ridge regression.

Evaluation metrics:

• LOYO R² • prediction stability across folds • coefficient posterior
variance

------------------------------------------------------------------------

### Step 3 --- Introduce Hierarchical Structure

Extend the model by introducing hierarchical structure across positions.

Example hierarchy:

β_position \~ Normal(β_global, τ_position)

Where:

β_global = global coefficient mean across all positions\
β_position = position‑specific coefficients

This allows the model to share information across WR/RB/TE while still
learning position‑specific effects.

Hypothesis:

This will improve stability for TE where N is smallest.

------------------------------------------------------------------------

### Step 4 --- Evaluate Posterior Predictive Distributions

Bayesian models produce distributions rather than single predictions.

For each prospect compute:

• posterior mean B2S • 80% credible interval • 95% credible interval

Compare interval widths across positions.

Expected results:

TE intervals should be wider due to lower sample size.

------------------------------------------------------------------------

### Step 5 --- Stability Testing

Compare Ridge vs Bayesian Ridge across several diagnostics.

Diagnostics:

• prediction variance across LOYO folds • coefficient stability across
folds • out‑of‑sample predictive performance • sensitivity to feature
inclusion

If Bayesian models produce significantly more stable coefficients
without sacrificing predictive power, they may replace Ridge as the
primary model.

------------------------------------------------------------------------

# Upgrade 2: Conformal Prediction Intervals

## Motivation

The current system produces a percentile score (ORBIT) and a predicted B2S
value but provides no uncertainty estimate.

Conformal prediction provides **distribution‑free prediction intervals**
that guarantee statistical coverage regardless of model assumptions.

Advantages:

• easy to implement • model‑agnostic • produces calibrated uncertainty
intervals

This technique is increasingly used in forecasting and machine learning
systems that require reliable uncertainty estimates.

------------------------------------------------------------------------

## Key Hypothesis

Conformal prediction intervals will:

1.  produce calibrated uncertainty bounds around predicted B2S
2.  reveal when the model is extrapolating beyond the training
    distribution
3.  improve interpretability of ORBIT scores

------------------------------------------------------------------------

## Investigation Plan

### Step 1 --- Compute Residual Distribution

Using the existing LOYO predictions:

Compute residuals:

residual = \|actual_B2S − predicted_B2S\|

Store the empirical distribution of residuals.

------------------------------------------------------------------------

### Step 2 --- Construct Conformal Interval

For each new prediction:

prediction_interval = predicted_B2S ± quantile(residuals, q)

Where q corresponds to the desired coverage level.

Example:

80% interval → q = 0.80 residual quantile

------------------------------------------------------------------------

### Step 3 --- Evaluate Calibration

Test coverage on historical predictions.

Procedure:

1.  compute intervals for all LOYO predictions
2.  check percentage of actual outcomes within interval

Expected result:

• 80% intervals should contain \~80% of outcomes • 90% intervals should
contain \~90% of outcomes

If coverage is incorrect, recalibrate interval quantiles.

------------------------------------------------------------------------

### Step 4 --- Interval Width Analysis

Analyze interval widths across:

• positions (WR vs RB vs TE) • draft capital tiers • extreme athletic
profiles

Hypothesis:

• TE intervals will be wider • late‑round players will have wider
intervals • extreme athletic outliers will produce larger uncertainty
ranges

------------------------------------------------------------------------

### Step 5 --- Integrate with ORBIT Score

Translate prediction intervals into ORBIT uncertainty.

Example output:

Prospect A

ORBIT: 84\
Predicted B2S: 13.2 PPG\
80% interval: \[8.5, 16.1\]

This communicates uncertainty without changing the core scoring system.

------------------------------------------------------------------------

# Experimental Comparison Framework

To evaluate both upgrades fairly, the following metrics should be
tracked:

  Metric                  Ridge      Bayesian Ridge   Ridge + Conformal
  ----------------------- ---------- ---------------- -------------------
  LOYO R²                 baseline   compare          same
  LOYO MAE                baseline   compare          same
  prediction stability    baseline   compare          same
  uncertainty intervals   none       posterior        conformal
  interpretability        high       moderate         high

------------------------------------------------------------------------

# Decision Criteria

Adopt Bayesian Ridge if:

• LOYO performance is equal or better • coefficient stability improves •
prediction variance decreases

Adopt conformal intervals if:

• coverage calibration is accurate • intervals provide meaningful
uncertainty signals

These upgrades can also coexist:

Bayesian model → predictions\
Conformal method → calibrated intervals

------------------------------------------------------------------------

# Expected Outcome

The most likely result is:

• Ridge remains the best core predictor • Conformal intervals provide
the most practical improvement

Bayesian models may improve TE stability but require more computational
complexity.

Therefore the recommended path is:

1.  implement conformal intervals first
2.  experiment with Bayesian ridge as a secondary research track

------------------------------------------------------------------------

End of Feedback5.
