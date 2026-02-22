# dynasty-prospect-model

College-to-NFL fantasy projection model — regression on B2S outcomes.

Part of the three-repo FF pipeline:
```
cfb-prospect-db   → college features (stats, combine, recruiting, draft capital)
nfl-fantasy-db    → NFL outcomes (B2S labels, big board)
dynasty-prospect-model  ← you are here
```

---

## What This Is

An independent regression model that predicts a college prospect's **B2S score**
(Best Two Seasons — avg PPR PPG of top-2 seasons in first 3 NFL years) from
pre-draft college and combine data. Not a ZAP reproduction — we fit our own
coefficients to actual B2S outcomes.

---

## Pipeline

```
Step 1: Build training data
    python scripts/build_training_set.py

Step 2: EDA (Jupyter notebooks)
    jupyter lab
    → Open notebooks/01_feature_correlations.ipynb
    → Explore feature-B2S correlations by position
    → Identify which features matter, which don't

Step 3: Fit model (after EDA)
    python scripts/fit_model.py --all

Step 4: Score a draft class
    python scripts/score_class.py --year 2026
```

---

## Setup

1. Prerequisites: cfb-prospect-db and nfl-fantasy-db must be populated.

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create `.env`:
   ```
   cp .env.example .env
   # Edit CFB_DB_PATH and NFL_DB_PATH to point to your SQLite files
   ```

4. Build training CSVs:
   ```
   python scripts/build_training_set.py
   ```
   Produces: `data/training_WR.csv`, `data/training_RB.csv`, `data/training_TE.csv`

---

## Training Window

- Draft classes: 2011–2022 (≥3 NFL seasons complete by end of 2024)
- ~700 labeled players across WR/RB/TE
- Feature coverage note: consensus big board (nflmockdraftdatabase.com) available
  from 2016 onward; recruiting (CFBD) available from 2018 onward

---

## Feature Set (pre-draft, no NFL data leakage)

### WR
| Feature | Description |
|---------|-------------|
| best_rec_rate | rec_yards / team_pass_att in best qualifying season |
| best_dominator | rec_yards / team total rec_yards (best season) |
| best_age | age at best season (breakout age proxy) |
| career_rush_yards | dual-threat signal |
| early_declare | entered draft with eligibility remaining |
| weight_lbs | size |
| speed_score | (weight × 200) / (40_time⁴) |
| draft_capital_score | normalized draft value (0–100) |
| recruit_rating | 247Sports composite |
| consensus_rank | overall pre-draft consensus board rank |
| teammate_score | sum of draft capital for co-draftees from same school |

### RB
| Feature | Description |
|---------|-------------|
| best_rec_rate | receiving ability (key differentiator for RBs) |
| best_reception_share | target share in best season |
| best_age | breakout age |
| weight_lbs | |
| speed_score | |
| draft_capital_score | |
| teammate_score | |
| consensus_rank | |

### TE
| Feature | Description |
|---------|-------------|
| draft_capital_score | dominant predictor for TEs |
| speed_score | |
| career_rec_per_target | rec_yards / targets (YPRR proxy — PFF YPRR unavailable) |
| best_age | |
| weight_lbs | |
| consensus_rank | |

---

## Target Variable: B2S Score

- **WR/RB**: average PPR points/game of best 2 seasons in first 3 NFL years (≥8 games each)
- **TE**: best single season PPG (first 3 NFL years, ≥8 games)
- Players with zero qualifying seasons: B2S = 0 (busts)

---

## Model Approach (planned)

1. OLS linear regression — baseline, interpretable, fast
2. Leave-one-year-out cross-validation — avoids data leakage
3. LightGBM with monotonic constraints — main model
4. Calibrate to 0–100 scale using historical B2S distribution
5. Validation: Spearman rank vs ZAP published scores as directional check

---

## Known Data Gaps

| Gap | Impact | Notes |
|-----|--------|-------|
| Consensus big board 2011–2015 | Missing for older classes | nflmockdraftdatabase.com has no data pre-2016 |
| Recruiting data 2007–2017 | Missing (CFBD API quota) | Retry when quota resets |
| College YPRR (routes run) | TE model only | PFF+ required; proxy = rec_yards/targets |
