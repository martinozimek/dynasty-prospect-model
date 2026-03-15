# Dynasty Prospect Model -- External Review & Feedback (Feedback2)

Author: Independent Technical Review\
Date: 2026

------------------------------------------------------------------------

# Executive Summary

Your dynasty prospect model is **well-designed, transparent, and far
more rigorous than most public fantasy football models**. The
combination of:

-   strong documentation
-   reproducible data pipelines
-   LOYO cross‑validation
-   Lasso feature selection
-   Ridge regression stability
-   a Phase I (no-capital) model

is excellent practice and reflects **professional-level modeling
discipline**.

However, several structural improvements could materially increase
predictive power and reduce bias:

**Highest Impact Improvements**

1.  Add **target share and career YPRR features**
2.  Ensemble **LightGBM with Ridge**
3.  Output **probabilistic predictions (elite / starter / bust)**
4.  Replace **single best-season signal with weighted career
    production**
5.  Add **Peak-5 label alongside B2S** to capture late developers

Estimated improvement potential: **\~15--25% predictive gain**.

------------------------------------------------------------------------

# 1. Training Target (B2S)

Current label:

Best 2 of first 3 seasons PPR PPG.

Strengths: - reduces rookie volatility - captures early career peak
value - aligns with dynasty time horizons

## Issue 1 --- Opportunity Bias

Late developers are penalized.

Examples historically include players who developed after year 3 (e.g.,
WRs with slower development curves or TEs).

### Recommended Fix

Create dual targets:

B2S_3yr = current label\
Peak5 = best season within first 5 NFL seasons

Composite label:

DynastyValue = 0.7 \* B2S_3yr + 0.3 \* Peak5

This preserves early-career signal while allowing late breakouts.

------------------------------------------------------------------------

## Issue 2 --- Survivorship Bias

Players who never earn playing time may be excluded if seasons do not
qualify.

This causes the model to train mostly on successful outcomes.

### Fix

Treat missing qualifying seasons as **0 PPG** rather than removing them.

------------------------------------------------------------------------

## Issue 3 --- Position Development Curves

Typical peak ages:

WR: 24--26\
RB: 23--25\
TE: 26--28

Your B2S definition underestimates TE value.

### Fix

Position-specific targets:

WR/RB → Best 2 of first 3 seasons\
TE → Best 2 of first 5 seasons

------------------------------------------------------------------------

# 2. Data Engineering

Your data stack is strong:

-   CFBD college production
-   PFF route metrics
-   nflverse combine data
-   recruiting ratings
-   consensus big boards

This produces excellent coverage.

However, one design introduces statistical bias.

------------------------------------------------------------------------

## Best Season Selection Bias

Current rule:

best_season = argmax(receiving_yards / team_pass_attempts)

Problem:\
Players with **one anomalous season** receive disproportionate
influence.

Examples:

-   transfer players
-   injury‑shortened seasons
-   role changes

### Recommended Fix

Use weighted career signal:

ProductionScore =\
0.6 \* best_season\
+ 0.4 \* career_average

Alternative:

ProductionScore = max(best_season, career_avg \* 1.1)

------------------------------------------------------------------------

# 3. Feature Engineering

Your feature selection is strong and aligns with literature.

Key good signals included:

-   YPRR
-   breakout age proxies
-   team production share
-   combine metrics
-   recruiting ratings
-   man/zone efficiency splits

However several **elite predictors are missing**.

------------------------------------------------------------------------

## Missing Feature 1 --- Target Share

target_share = targets / team_pass_attempts

Historically one of the **most predictive WR metrics**.

------------------------------------------------------------------------

## Missing Feature 2 --- Career YPRR

Single-season YPRR introduces noise.

Better signals:

career_yprr\
career_routes

------------------------------------------------------------------------

## Missing Feature 3 --- Early Declare

Years in college strongly correlate with NFL success.

Example features:

declare_age\
years_in_college

------------------------------------------------------------------------

## Missing Feature 4 --- Size-Speed Score

Instead of raw combine metrics:

SpeedScore = weight \* (200 / forty_time\^4)

Captures functional athleticism better than raw metrics.

------------------------------------------------------------------------

## Missing Feature 5 --- True Dominator

Receiving-only dominator misses RB rushing dominance.

Use full dominator metric:

dominator = (yards_share + TD_share) / 2

------------------------------------------------------------------------

# 4. Modeling Architecture

Current pipeline:

LassoCV → Ridge Regression → LOYO CV

This is **clean, interpretable, and defensible**.

Strengths:

-   Lasso removes weak predictors
-   Ridge stabilizes coefficients
-   LOYO prevents temporal leakage

However the model is **fully linear**.

------------------------------------------------------------------------

## Limitation: Nonlinear Relationships

Prospect signals behave nonlinearly.

Example:

YPRR \< 1.5 → weak\
1.5--2.5 → moderate\
\>3.0 → elite

Linear models struggle with these thresholds.

------------------------------------------------------------------------

## Recommended Upgrade

Train multiple models:

Ridge\
ElasticNet\
LightGBM

Ensemble prediction:

Prediction =\
0.4 Ridge\
0.4 LightGBM\
0.2 ElasticNet

This captures nonlinear interactions while preserving interpretability.

------------------------------------------------------------------------

# 5. Cross Validation

Your use of **Leave-One-Year-Out CV** is excellent.

Training structure:

train: all draft classes except year X\
test: draft class X

This correctly simulates real-world forecasting.

Many public models incorrectly use random k-fold and leak era
information.

Your approach is best practice.

------------------------------------------------------------------------

# 6. Draft Capital Integration

Your Phase I framework is one of the **most innovative elements of the
model**.

Phase I model removes draft capital.

Full model includes draft capital.

Capital delta measures the difference.

This allows identification of:

-   market overvaluation
-   market undervaluation

This concept is extremely powerful for dynasty evaluation.

------------------------------------------------------------------------

## Improvement Suggestion

Draft capital is highly nonlinear.

Difference between pick 10 and pick 30 is much larger than between pick
150 and 170.

Test transformations:

capital = log(overall_pick)

or

capital = sqrt(pick)

------------------------------------------------------------------------

# 7. Sample Size Limitations

Training sample sizes:

WR \~286\
RB \~201\
TE \~127

The TE dataset is particularly small.

------------------------------------------------------------------------

## Suggested Fix

Expand training window to include earlier classes (2011+).

Instead of median-imputing missing PFF splits, allow them to remain
missing or exclude them for older eras.

Median imputation artificially compresses variance.

------------------------------------------------------------------------

# 8. Evaluation Metrics

Current focus appears to be R².

For dynasty modeling, ranking metrics are more relevant.

Recommended evaluation metrics:

Spearman rank correlation\
Top‑12 hit rate\
Top‑24 hit rate\
ROC‑AUC for breakout classification

Example classification target:

Top‑12 fantasy season probability

------------------------------------------------------------------------

# 9. Prediction Uncertainty

Current model outputs point estimates only.

Dynasty forecasting benefits from probability outputs.

Recommended outputs:

P(elite)\
P(starter)\
P(bust)

Possible methods:

quantile regression\
bootstrap ensembles

------------------------------------------------------------------------

# 10. Context Modeling (Future Improvement)

Current model predicts **player ability only**.

NFL outcomes depend heavily on:

-   team offensive environment
-   QB quality
-   target competition
-   coaching tendencies

Future architecture could use a two-stage model:

Stage 1 → player talent model\
Stage 2 → landing spot adjustment

Example contextual features:

team pass rate\
QB EPA\
target competition\
coach pass tendency

------------------------------------------------------------------------

# Final Assessment

Overall model quality: **very strong**.

The methodology, transparency, and statistical care exceed most public
dynasty models.

Biggest improvement opportunities:

1.  Target share + career YPRR features
2.  Gradient boosting ensemble
3.  Probability outputs
4.  Weighted career production instead of best-season-only
5.  Dual target labels (B2S + Peak5)

With these upgrades, the model could become **one of the most robust
public dynasty prospect models available**.
