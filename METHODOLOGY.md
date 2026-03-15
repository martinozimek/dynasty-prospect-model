# Dynasty Prospect Model — Complete Methodology Documentation

**Purpose of this document:** Exhaustive documentation of every design decision, assumption, formula, and
shortcut made in building this model. Written so that an independent reviewer — human or AI — can vet the
methodology without needing to read the source code.

**What this model does:** Predicts how productive a skill-position NFL draft prospect (WR, RB, TE) will be
in dynasty fantasy football, expressed as a percentile score (0–100) called the ZAP Score. It also computes
a "Phase I" score that strips out all draft capital signal, and a capital delta risk flag.

---

## Table of Contents

1. [Training Target: B2S Score](#1-training-target-b2s-score)
2. [Data Sources](#2-data-sources)
3. [Player Identity Matching](#3-player-identity-matching)
4. [Training Window](#4-training-window)
5. [Sample Sizes](#5-sample-sizes)
6. [Feature Engineering: Raw Features](#6-feature-engineering-raw-features)
7. [Feature Engineering: Derived Features and Interaction Terms](#7-feature-engineering-derived-features-and-interaction-terms)
8. [Missing Data Handling](#8-missing-data-handling)
9. [Model Architecture: Capital Model](#9-model-architecture-capital-model)
10. [Model Architecture: Phase I (No-Capital) Model](#10-model-architecture-phase-i-no-capital-model)
11. [Cross-Validation Strategy](#11-cross-validation-strategy)
12. [ZAP Score Construction](#12-zap-score-construction)
13. [Capital Delta and Risk Classification](#13-capital-delta-and-risk-classification)
14. [Scoring Future Draft Classes (Inference)](#14-scoring-future-draft-classes-inference)
15. [Known Limitations and Unresolved Assumptions](#15-known-limitations-and-unresolved-assumptions)
16. [Assumption Inventory for Vetting](#16-assumption-inventory-for-vetting)

---

## 1. Training Target: B2S Score

### What it is

**B2S = Best 2 of First 3 NFL Seasons** PPR points-per-game.

Operationally: for each player's first three NFL seasons, compute their PPR PPG in each season. Take the
two highest values and average them. That average is the B2S score.

### Qualifying threshold

A season qualifies (is eligible to be one of the "best 2") only if the player appeared in **≥ 8 games**
in that season. Seasons with fewer than 8 games are discarded entirely before selecting the best 2.

**Source:** Confirmed from page 8 of JJ Zachariason's 2025 Prospect Guide. The "≥ 8 games" threshold is
explicitly stated. The "≥ 7 games" alternative is NOT used.

### What "first 3 seasons" means

"First 3 seasons" = the 3 NFL seasons beginning with the player's draft year. Season 1 = draft year
(regardless of how many games they played as a rookie). Season 3 = draft year + 2.

**Assumption:** Players injured in year 1 (IR'd in week 1) still have year 1 count as a consumed season.
The model does not search for "first 3 seasons with ≥ 1 game." This is an unverified assumption about
JJ's implementation.

### PPR scoring

PPR = 1.0 point per reception + 0.1 points per receiving yard + 6.0 points per receiving TD + 0.1 points
per rushing yard + 6.0 points per rushing TD. This is standard PPR, which is the most common dynasty
scoring format. No additional bonuses (no TE premium, no bonus for 100+ yard games).

**Source:** Computed from nflverse weekly game logs (player_game table in nfl-fantasy-db).

### Final-week exclusion

Whether week 18 (regular-season finale) games are excluded from PPG computation is **UNCONFIRMED**. Some
dynasty analysts exclude the final week because starters are rested. Our implementation includes all
regular-season weeks. This may inflate PPG slightly for starters on playoff teams.

### Why B2S instead of Year 1 or Career PPG

- Year 1 is too noisy (rookie learning curves, snap count development, coaching system fit)
- Career PPG rewards longevity over peak talent — a 12-year average includes age-decline seasons
- B2S captures peak sustainable performance in the early dynasty window (when dynasty value is highest)
- "Best 2 of 3" is robust to one outlier year (injury, suspension, bad team)

**Assumption embedded here:** Dynasty value is primarily determined by peak performance in years 1–3, not
career longevity. This is a reasonable assumption for dynasty formats but may not apply equally across
scoring settings.

---

## 2. Data Sources

### 2.1 College Football Database (CFBD)

**API:** collegefootballdata.com (REST API, no authentication for most endpoints).

**What we pull:**
- Player season stats: `rec_yards`, `targets`, `receptions`, `rec_tds`, `rush_yards`, `rush_attempts`,
  `rush_tds`, `games_played` — for all FBS players, 2007–2025
- Player usage rates: `usage_overall`, `usage_pass`, `usage_rush` — per player per season
- Team season data: `pass_attempts`, `sp_plus_rating` — per team per season

**Coverage:** 150k+ player-season rows for 2007–2025. `games_played` is 100% populated (computed from
week-by-week game logs due to a CFBD quirk where the `/stats/season` endpoint does not return games played).

**Important CFBD quirk:** `dominator_rating` in CFBD is computed as:
```
dominator_rating = (player_rec_yards / team_rec_yards + player_rec_tds / team_rec_tds) / 2
```
This is a RECEIVING dominator only. It does not include rushing yards or rushing TDs. For WR and TE, this
is the appropriate dominator. For RB, this means CFBD dominator captures receiving contribution only —
not rushing. A dominant rushing RB with no receiving role will have a low dominator despite being the
workhorse. We do not correct for this in the RB model; instead `best_total_yards_rate` (which includes
rushing yards) is the primary RB production input.

### 2.2 PFF College Data

**Source:** Pro Football Focus (PFF+) subscription data, exported as CSV files and ingested into our
database.

**Coverage:** 2008–2025 season data; 17,763 player-season rows. PFF data first becomes comprehensive for
FBS skill players around 2014. Pre-2014 coverage exists but is sparser.

**What PFF provides that CFBD does not:**
- `yprr`: Yards per route run — the most predictive per-play receiving metric per Scott Barrett
- `routes_run`, `routes_per_game`: Route volume
- `receiving_grade`: PFF's composite receiving grade (0–100)
- `drop_rate`: Drops / catchable targets
- `contested_catch_rate`: Contested catches / contested targets
- `target_separation`: Average yards of separation at time of target
- **Depth splits** (`depth_data` JSON): `deep_yprr`, `deep_target_rate`, `behind_los_rate`
- **Concept splits** (`concept_data` JSON): `slot_yprr`, `slot_target_rate`, `screen_rate`
- **Scheme splits** (`scheme_data` JSON): `man_yprr`, `zone_yprr`, `man_zone_delta`

**Critical implementation note:** ALL PFF JSON values for the split columns are stored as Python strings
in the database, not as floats. Every value must be cast with `float()` before arithmetic. A missing
value appears as `None` or `""` (empty string), not `NaN`. Helper function `_f(v)` handles this.

**PFF as source of truth for counting stats:** Where PFF data exists for a (player_id, season_year) pair,
PFF's counting stats (`rec_yards`, `targets`, `receptions`, etc.) **override** the CFBD counting stats.
PFF's play-by-play tracking is considered more accurate than the CFBD box-score data. After overriding,
`rec_yards_per_team_pass_att` is recomputed using PFF rec_yards / CFBD team_pass_att.

**PFF split coverage rates (training set, 2014–2022):**
- Depth/concept splits (deep_yprr, slot_yprr): ~64–65% of training rows have data
- Scheme splits (man_yprr, zone_yprr): ~35–41% of training rows have data (scheme data is less
  comprehensive in older PFF exports)

### 2.3 NFL Combine and Draft Data

**Source:** nflverse (R package, exported to CSV). Covers NFL combines 2000–2026 and NFL drafts 2000–2025.

**What we use:**
- Combine: `forty_time`, `weight_lbs`, `height_inches`, `vertical_jump`, `broad_jump`, `three_cone`,
  `shuttle`, `bench_press`
- Draft: `overall_pick`, `draft_round`, `position_drafted`, `nfl_team`, `draft_year`
- Draft capital score: pre-computed in cfb-prospect-db using the exponential decay formula (see §6.6)

**Coverage:** 2011–2025 draft classes covered for WR/RB/TE. 2026 combine data ingested as of 2026-03-13.
Match rate between nflverse name + position + year and CFBD player records: ~76–79%.

### 2.4 247Sports Recruiting Data

**Source:** CFBD API's `/recruiting/players` endpoint, which aggregates 247Sports composite ratings.

**What we use:**
- `recruit_rating`: 247Sports composite score, scale 0.0–1.0 (e.g., 5-star ≈ 0.98+, 3-star ≈ 0.85)
- `recruit_stars`: Star rating 1–5
- `recruit_rank_national`: National ranking position
- `recruit_year`: Year recruited out of high school

**Coverage:** 2007–2025 recruit classes; 35,583 rows; good FBS coverage.

**Key assumption:** For players without a DOB in CFBD (common for recent recruits), age is estimated as:
```
age_at_season_start = 18.5 + (season_year - recruit_year)
```
This assumes the average recruit enters college at 18.5 years old (i.e., from the fall of their senior
year of high school). This estimate has ±0.5 year accuracy for most players. Players who redshirted or
enrolled early will have systematically wrong ages. This affects `best_age`, `breakout_score`,
`total_yards_rate`, and any interaction term involving age.

### 2.5 Pre-Draft Big Board (Consensus Rank)

**Source:** nfl-fantasy-db's `NFLBigBoard` table, populated from aggregated consensus rankings.

**Coverage:** 2016+ draft classes only. Pre-2016 training rows have `consensus_rank = NULL` and
`position_rank = NULL`.

**What it represents:** The overall positional or cross-positional consensus ranking of a player before
the NFL draft, as aggregated from industry big boards. This is a pre-draft market signal that
incorporates NFL team evaluations, analyst consensus, and (circularly) draft capital expectations.

---

## 3. Player Identity Matching

### The problem

We have NFL players (identified by NFL name, position, draft year in nfl-fantasy-db) and college players
(identified by CFBD player_id in cfb-prospect-db). These two databases have no shared key. Matching is
done by fuzzy name matching.

### Matching architecture

The `CFBLink` table in nfl-fantasy-db bridges NFL players to CFBD player IDs:
- `nfl_player_name` + `position` → `cfb_player_id` + `cfb_full_name` + `match_score`
- `match_score` is a 0–100 fuzzy match confidence score from the `rapidfuzz` library

### Match score filter

Training rows with `match_score < 80` are **dropped entirely**. The threshold of 80 is a judgment call:
links below this threshold are more likely to represent wrong-player matches (different player with a
similar name) than genuine low-confidence correct matches.

**Assumption:** The 80-threshold is approximately right. A systematic audit showed that most mismatches
below 80 represent genuinely different players. However, the threshold was not optimized against a
ground-truth label set — it was chosen by manual inspection of audit output.

### Deduplication

If the same `cfb_player_id` appears twice (e.g., because a player's NFL name has two spellings and both
were matched), only the row with the **higher match_score** is kept. Tie-break: higher b2s_score.

**Known false-match patterns:**
- Initials (K.C. Concepcion matched to K.C. Ossai)
- Same name, different position (CJ Donaldson RB matched to CJ Donaldson LB)
- Shortened nicknames (Jam Miller matched to James Miller LB instead of Jamarion Miller RB)

---

## 4. Training Window

### Year range

**Default: 2014–2022** (9 draft classes).

### Why 2014 as the start year

PFF college data becomes comprehensive for FBS skill players starting with the 2013 college season
(draft class of 2014). Before 2014, PFF data is sparse. When PFF data is missing, the training pipeline
imputes the median value for that feature. Training on 2011–2013 classes with median-imputed PFF values
creates a systematic era bias: players from an era where PFF data didn't exist get scored as if they
were at the median of the PFF distribution, making the PFF features appear less predictive than they
actually are (their true value is obscured by imputation). The 2014 cutoff is a practical compromise —
it sacrifices 3 years of data (~60–90 players) to avoid this imputation artifact.

**Alternative tested:** Using 2011+ (with --start-year 2011 flag). This slightly increases N but reduces
Ridge LOYO R² across all positions because PFF leakage from median imputation adds noise.

### Why 2022 as the end year

B2S requires at least 3 NFL seasons to be complete. The 2022 draft class had their third NFL season in
2024. Using 2023+ classes risks right-censored labels — a 2023 player who had great years 1–2 gets
scored poorly if we only have 2 seasons. The 2022 cutoff ensures all training labels are fully realized.

**Known upcoming update:** 2023 class B2S labels will be complete after the 2025 NFL season (end of
2025). Adding 2023 data will increase training N by ~60–90 players across positions.

---

## 5. Sample Sizes

As of 2026-03-13 (2014–2022 training window):

| Position | N (labeled rows) | Notes |
|----------|-----------------|-------|
| WR | 286 | After dedup + link-score filter |
| RB | 201 | After dedup + link-score filter |
| TE | 127 | After dedup + link-score filter |

TE has the smallest N. This is a material limitation for model stability and is why TE uses a wider
Ridge alpha range (1–1000 vs. 0.1–100 for WR/RB) and why TE LightGBM is retired when its LOYO R² is
more than 0.10 below Ridge LOYO R².

---

## 6. Feature Engineering: Raw Features

All features are computed **before the NFL draft** (no NFL outcome data). This is the no-leakage
guarantee.

### 6.1 Best Season Selection

Rather than using career averages, most features are computed from a player's **single best season**
(`best_season`), defined as the qualifying season (≥ 6 games played) with the highest
`rec_yards_per_team_pass_att`. The minimum 6-game threshold is configurable; the default is 6 games.

**Assumption:** The best qualifying season is more predictive than career average because: (a) it
captures peak college performance, (b) it is less diluted by development seasons or seasons behind a
more experienced player, and (c) it better reflects the player's true capability rather than their
opportunity level.

**What "best" means:** Highest `rec_yards / team_pass_att`. For WRs and TEs, this is receiving
efficiency relative to team context. For RBs, this metric captures the player's receiving role relative
to the team's passing game — it does NOT capture rushing dominance. RB rushing contribution is captured
separately via `best_total_yards_rate`.

**The denominator:** `team_pass_att` = total team pass attempts in that season, from CFBD team season
data. This normalizes for the team's offensive philosophy. A WR who catches 800 yards in 250 team pass
attempts is more efficient than one who catches 800 yards in 550 team pass attempts.

### 6.2 Breakout Score (WR primary production input)

Formula confirmed from JJ Zachariason's Late Round Fantasy podcast (Episode 1083, February 2026):

```
breakout_score = rec_yards_per_team_pass_att × SOS_mult × age_mult
```

Where:
- `rec_yards_per_team_pass_att` = player receiving yards / team pass attempts in that season
- `SOS_mult = max(0.70, min(1.30, 1.0 + (sp_plus - 5.0) / 100.0))`
  - SP+ ≈ 5 is average FBS → mult = 1.0
  - SP+ ≈ 30 is elite (Alabama, Georgia level) → mult ≈ 1.25
  - SP+ ≈ −15 is weak FBS → mult ≈ 0.80
  - SP+ unavailable → mult = 1.0 (no adjustment)
- `age_mult = max(0.0, 26.0 - age_at_season_start)`
  - A 20-year-old gets a mult of 6.0; a 22-year-old gets 4.0; a 26-year-old gets 0.0

**Taken as maximum across all qualifying seasons.** A player who posts a high breakout score in their
sophomore year is rewarded more than one who posts the same raw score in their senior year (younger =
higher age_mult). This is intentional: early age of dominance is a JJ-confirmed positive signal.

**Option-offense filter:** Seasons where `team_pass_att < 200` are excluded before computing breakout
score. FBS average is ~380 pass attempts; Navy/Army-type option offenses have ~100. Without this filter,
a player who catches 300 yards on 100 team pass attempts gets a denominator 3.8x smaller than a
comparable player on a normal team, inflating their breakout score by 3.8x.

**Assumption:** The 200 pass-attempt threshold correctly identifies option offenses. This may
inappropriately exclude some run-heavy but not pure-option teams (e.g., old school SEC teams with 220
pass attempts). Validated manually against known option programs.

### 6.3 Total Yards Rate (RB primary production input)

Formula from JJ Zachariason (Episode 1081, February 2026):

```
total_yards_rate = (rec_yards + rush_yards) / team_pass_att × SOS_mult × age_mult
```

**Important approximation:** The denominator uses `team_pass_att` rather than total team plays. Total
team plays (pass + rush) are not stored in our database. This is a known imprecision: a run-heavy team
might have 250 pass attempts but 600 total plays. Using only pass attempts as the denominator inflates
the rate for players on run-heavy teams (where the denominator is small relative to total opportunity).

**Assumption:** `team_pass_att` is an adequate proxy for team offensive context. This is defensible
because JJ's stated denominator is "team pass attempts" (not total plays), and pass attempts correlate
with modern offensive philosophy more cleanly than total plays.

Same option-offense filter (min 200 team pass attempts) and age_mult as breakout score.

### 6.4 Dominator Rating

`best_dominator` = the player's `dominator_rating` in their best qualifying season, taken directly from
CFBD. As noted in §2.1, this is a receiving-only dominator (rec_yards share + rec_tds share / 2).

**JJ's own position on dominator:** Described as "outdated" in 2026 podcast episodes. JJ prefers
`rec_rate` (our breakout score) over raw dominator. We include dominator in the candidate feature pool
for Lasso to adjudicate, but it has been largely superseded by breakout_score_x_capital as the selected
WR interaction term.

### 6.5 College Fantasy PPG

```
college_fantasy_ppg = (receptions × 1.0 + rec_yards / 10 + rec_tds × 6 + rush_yards / 10 + rush_tds × 6) / games_played
```

This is PPR scoring applied to the player's best college season, normalized per game. It is a general
production metric that captures multi-dimensional contribution (catches, yards, TDs) in one number.

**Assumption:** PPR scoring (1 point per reception) applies to college players the same way it does in
NFL fantasy contexts. College receptions in a spread offense where every screen counts as a reception
are treated the same as NFL catches.

### 6.6 Draft Capital Score

```
draft_capital_score = 100 × exp(−0.023 × (overall_pick − 1))
```

This is an exponential decay formula that maps draft pick number to a 0–100 score:
- Pick 1 → 100.0
- Pick 10 → ~80.4
- Pick 32 → ~51.2
- Pick 100 → ~10.7
- Pick 200 → ~1.1
- Pick 262 (last pick) → ~0.18

**Source:** Stored in the NFLDraftPick table in cfb-prospect-db. Pre-computed at ingestion time.

**Assumption:** The exponential decay functional form is appropriate for capturing the non-linear
relationship between draft position and expected player quality. The decay constant −0.023 was chosen
to map picks 1–262 onto the 0–100 range with a reasonable shape. It was NOT estimated from the data —
this is a structural assumption.

**Alternative decay constants used in the literature:** Massey-Thaler (2013) use a different functional
form (surplus value curve). Our formula is simpler. The key property is monotone decreasing, which is
correct. The precise rate of decay is assumed, not estimated.

### 6.7 Speed Score

```
speed_score = (weight_lbs × 200) / (forty_time^4)
```

**Source:** Bill Barnwell's speed score formula, widely used in sports analytics. It rewards players who
run fast for their weight. A 250-lb player running a 4.45 forty scores higher than a 180-lb player
running the same time, because the heavier player is more athletically exceptional.

**Data source for weight:** Combine weight, falling back to the Player record's listed weight. The
Player record weight may reflect college weight (pre-combine), while combine weight is the definitive
measurement. When available, combine weight is used.

### 6.8 Early Declare

```
early_declare = 1 if career_seasons <= 3 else 0
```

`career_seasons` = total number of college seasons recorded for the player, regardless of redshirt years.

**Assumption:** A player with ≤ 3 seasons of college data (true freshmen who left early, or players who
played only 2–3 seasons before the draft) "declared early." This is a proxy for whether a player was
drafted before exhausting college eligibility. It does NOT distinguish between genuine early declarees
(high upside, NFL-ready) and players who were forced to leave (poor grades, off-field issues). Lasso
did not select this feature in any position's final model — it carries marginal information beyond
draft capital.

### 6.9 Power-4 Conference Flag

```
power4_conf = 1 if conference in {SEC, Big Ten, ACC, Big 12} else 0
```

Based on the conference of the player's best season.

**Assumption:** Power-4 conference competition is meaningfully stronger than Group of 5 or independent.
This does NOT account for cross-conference variation (a MAC team that plays a tough non-conference
schedule, etc.). The SP+ adjustment in breakout_score partially addresses schedule strength, so power4_conf
provides an additional binary tier signal.

### 6.10 Agility Score

```
agility_score = (three_cone + shuttle) / 2
```

Both in seconds; lower = more agile. This is stored as a raw time average, so the monotone direction
for the model is negative (lower agility_score = better).

**Coverage:** Only players who ran both three-cone and shuttle at the combine. Many players skip one or
both. Coverage is ~40–60% of training rows. When either component is missing, agility_score is NaN and
gets median-imputed in the pipeline.

### 6.11 Teammate Score

```
teammate_score = sum of draft_capital_score of other WR/RB/TE draftees from the same school, within ±2 years of the target player's draft year
```

**Rationale:** A player from a school that produced multiple drafted skill players (e.g., Alabama, Ohio
State) was playing in a richer offensive environment with better surrounding talent. This is a proxy for
the quality of the college offense.

**Known weakness:** This rewards players from powerhouse programs for reasons unrelated to their own
talent. An average player at Alabama gets a high teammate_score simply by association. Lasso sometimes
selects this, sometimes not. It is in the candidate pool but its contribution is modest.

### 6.12 Recruiting Rating

`recruit_rating` = 247Sports composite score (0.0–1.0). This is a pre-college-career signal about
how highly scouts rated the player coming out of high school. It predates the NFL draft and is
**independent of draft capital**, making it valid for use in the Phase I (no-capital) model.

**Coverage:** ~85–90% of 2014+ training rows. Players from smaller programs or JUCO transfers may
lack recruiting profiles.

---

## 7. Feature Engineering: Derived Features and Interaction Terms

All of these are computed in `scripts/analyze.py::engineer_features()`.

### 7.1 Log Transforms

```python
log_draft_capital = log(draft_capital_score + 0.1 + 1)
log_rec_rate      = log(best_rec_rate + 0.001)
log_dominator     = log(best_dominator + 0.001)
```

Log transforms compress the right tail of distributions. Draft capital has a fat right tail (pick 1
is 100 but pick 262 is ~0.18 — a 500x range). Log compresses this. Log transforms are always applied
to positive quantities; the small constant prevents log(0).

### 7.2 Interaction Terms

**Capital × production interactions (the dominant WR and RB features):**
```python
capital_x_dominator   = draft_capital_score × best_dominator
capital_x_age         = draft_capital_score × best_age
rec_rate_x_capital    = best_rec_rate × draft_capital_score
breakout_score_x_capital = best_breakout_score × log_draft_capital
total_yards_rate_x_capital = best_total_yards_rate × log_draft_capital
```

**Motivation:** The hypothesis is that draft capital and production are not independent — their joint
value is greater than the sum of their parts. A player with both high production AND high capital is
more likely to succeed than the additive prediction. This is consistent with the "NFL teams are
approximately rational" assumption: they assign high capital to players who have both the college
production evidence AND the physical traits they cannot observe from public data.

**Assumption embedded:** Multiplicative interaction is the correct functional form. An additive model
(capital + production) may be equally valid. Linear multiplicative interactions were chosen because
they are: (a) interpretable, (b) consistent with the "both signals together" hypothesis, and
(c) well-handled by Ridge regression without overfitting.

**PFF × capital interactions:**
```python
yprr_x_capital          = best_yprr × draft_capital_score
deep_target_x_capital   = best_deep_target_rate × draft_capital_score
deep_yprr_x_capital     = best_deep_yprr × draft_capital_score
man_delta_x_capital     = best_man_zone_delta × draft_capital_score
slot_rate_x_capital     = best_slot_target_rate × draft_capital_score
```

Same rationale: these capture players where both NFL teams and PFF metrics agree on a specific
efficiency dimension.

### 7.3 Draft Tier Encoding

```python
draft_tier = 4  if pick <= 16    # top-half first round
           = 3  if 17 <= pick <= 50   # late first + early second
           = 2  if 51 <= pick <= 100  # second + early third
           = 1  if pick > 100         # Day 3
```

**Motivation:** JJ Zachariason (Episode 1086) argues that draft capital operates in tiers, not as a
precise rank. A pick-17 and pick-32 player are in the same tier; a pick-1 and pick-17 are in different
tiers. The ordinal tier encoding captures this step-function structure that a continuous capital score
or log-transform cannot.

### 7.4 is_top16_rb Flag

```python
is_top16_rb = 1 if (position == "RB") and (overall_pick <= 16) else 0
```

**Motivation:** JJ (Episode 1086) observes that top-half first-round RBs have LOYO R² of ~0.526 —
nearly double the overall RB model R². This is so strong that it essentially creates a two-regime model:
top-16 RBs are near-automatic successes regardless of college production. The binary flag lets the
model explicitly capture this regime.

### 7.5 Draft Premium

```python
draft_premium = (100 - consensus_rank) / 100
```

Where `consensus_rank` is clipped at 300 before computing. This is a 0–1 scale where 0 = rank 100
(or higher number = lower ranked) and 1 = rank 0 (nonsensical, but the formula is defined this way
so rank 1 gives premium = 0.99). **VIF analysis found this collinear with position_rank (r ≈ −0.953)**
and it was removed from the TE candidate pool. It remains in the WR candidate pool.

### 7.6 Combined Athleticism Composite

```python
combined_ath = speed_score × (100 - overall_pick) / 100
```

This is speed weighted by inverse draft position. A fast player drafted early scores higher than a
fast player drafted late. The interpretation is: "athletically gifted players whose NFL teams saw fit
to draft them early." It combines two signals but in a somewhat circular way (speed × capital), which
is why the pure athleticism signals (speed_score, broad_jump, etc.) are retained separately in the
candidate pool.

---

## 8. Missing Data Handling

### Strategy: Median Imputation

All missing values in the training and scoring pipelines are handled by scikit-learn's
`SimpleImputer(strategy="median")`. The imputer is fit on the training data and then applied to
the test (or scoring) data using training medians.

**Implication:** Missing combine data (e.g., a player who didn't run the 40, or who declined the
combine) gets the median 40 time for the training set. Missing PFF data (pre-2014 players, or players
who didn't have enough route volume to qualify for PFF splits) gets the training set median for all
PFF metrics.

**Critical assumption:** Missing data is missing at random (MAR), or at minimum, the imputed median
value is an appropriate expectation for the missing case. This assumption likely fails for:
- Players who **declined to run the 40** at the combine: these are often players who either (a) had
  a slow projected 40 and chose to preserve draft stock, or (b) were such high picks they didn't need
  to test. Both scenarios mean the "true" 40 time is systematically different from the median.
- Players with **no PFF data**: pre-2014 classes, and some position groups that PFF didn't fully track.
  These players are not "average" on YPRR; they simply weren't tracked.

**No special handling for these cases.** This is a known limitation.

### Age fallback for missing DOB

When a player lacks a date of birth in CFBD (common for recent recruits), age is estimated from
recruit_year (see §2.4). This affects approximately 30–40% of the 2023–2025 declared class.

---

## 9. Model Architecture: Capital Model

### Pipeline

For each position, the final deployed model is a scikit-learn `Pipeline`:

```
Step 1: SimpleImputer(strategy="median")     — impute NaN with training medians
Step 2: StandardScaler()                      — zero-mean, unit-variance scaling
Step 3: RidgeCV(alphas=alpha_range, cv=5)    — Ridge regression with CV alpha tuning
```

The pipeline is trained on the Lasso-selected feature subset. The pipeline cannot be changed without
retraining — it encodes the training medians and scale parameters alongside the model coefficients.

### Feature Selection: LassoCV

Before fitting the Ridge model, LassoCV (`sklearn.linear_model.LassoCV`) is run on the full candidate
feature pool to select which features to retain:

```python
alphas = np.logspace(-4, 2, 60)     # 60 alpha values from 0.0001 to 100
lasso  = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
```

Features with non-zero Lasso coefficients are passed to the final Ridge model. If no features are
selected (rare), all candidates are used.

**Why Lasso for selection, then Ridge for prediction?**
- Lasso is L1-regularized and drives irrelevant feature coefficients to exactly zero — good for
  selection
- Lasso with small N and correlated features can be unstable (randomly zero-out correlated features)
- Ridge (L2) is more stable with correlated features and small N — better for prediction
- "Lasso selects, Ridge predicts" is a well-established approach for this regime

### Ridge Alpha Range (per-position)

```
WR: alpha in [0.1, 100]   (50 values, log-spaced)
RB: alpha in [0.1, 100]   (50 values, log-spaced)
TE: alpha in [1, 1000]    (50 values, log-spaced)
```

TE uses a wider range with stronger regularization because N=127 is small relative to the number of
candidate features (~40). Stronger regularization is appropriate to prevent overfitting.

**Selection method:** 5-fold cross-validation within RidgeCV on the training set. The alpha that
minimizes held-out MSE is selected.

### LightGBM (secondary model)

A LightGBM gradient boosted tree model is also trained per position, using the same Lasso-selected
features. Per-position hyperparameters:

```
WR: n_estimators=150, max_depth=3, min_child_samples=10, lr=0.05, subsample=0.8, colsample=0.8
RB: n_estimators=200, max_depth=4, min_child_samples=8,  lr=0.05, subsample=0.8, colsample=0.8
TE: n_estimators=100, max_depth=2, min_child_samples=15, lr=0.05, subsample=0.7, colsample=0.7
```

**Monotone constraints:** All LGBM models use monotone constraints to enforce domain-consistent
directionality (e.g., higher draft capital → higher predicted B2S; lower forty time → higher B2S).
This prevents the tree from learning counter-intuitive artifacts from noisy training data.

**TE LGBM retirement rule:** If LGBM LOYO R² < Ridge LOYO R² − 0.10, the TE LGBM model is not saved.
The gap threshold of 0.10 indicates that LGBM is overfit and not generalizing. With N=127 and ~10
selected features, LightGBM often memorizes training examples without generalizing.

**LGBM is NOT used for Phase I (no-capital) models.** Phase I has even fewer reliable signal features
and the already-small N combined with no capital signal creates severe overfitting risk for tree models.

### VIF Pruning

Before constructing the candidate feature pool, a Variance Inflation Factor (VIF) analysis identified
highly collinear features. These were **removed from the candidate pool** (not just Lasso-dropped):

- WR: `overall_pick` removed (r = −0.873 with `draft_tier`)
- RB: `overall_pick`, `capital_x_dominator`, `draft_capital_score` removed (r > 0.875 with other capital features)
- TE: `combined_ath_x_capital`, `draft_premium` removed (r > 0.953 with other features)

**Rationale:** Lasso can handle correlated features by selecting one and zeroing the other, but the
chosen feature may be arbitrary (dependent on random seed and data order). VIF pruning removes the less
interpretable duplicate, ensuring Lasso selects among the most meaningful representatives.

**Assumption:** Our judgment about which correlated feature to remove (e.g., `overall_pick` rather than
`draft_tier`) is correct. We chose to remove the feature that is a less stable signal (pick number
has more noise than tier membership) or less interpretable (complex interaction terms over simpler base
features).

---

## 10. Model Architecture: Phase I (No-Capital) Model

### Purpose

The Phase I model scores prospects on **pure college production, PFF efficiency, and athleticism** —
zero draft capital signal. It answers: "Where would this player rank if the NFL had never revealed their
opinion?"

### What is excluded from Phase I

ALL of the following are excluded from Phase I candidate feature pools:
- `draft_capital_score`, `overall_pick`, `draft_round`, `log_draft_capital`, `draft_tier`, `is_top16_rb`
- `consensus_rank`, `position_rank`, `draft_premium` (these are pre-draft market signals informed by
  NFL team interest — circular with capital)
- ALL capital interaction terms: `capital_x_dominator`, `capital_x_age`, `rec_rate_x_capital`,
  `breakout_score_x_capital`, `total_yards_rate_x_capital`, `yprr_x_capital`, `deep_target_x_capital`,
  `deep_yprr_x_capital`, `man_delta_x_capital`, `slot_rate_x_capital`

### What is included in Phase I

- All college production rates: `best_breakout_score`, `best_rec_rate`, `best_dominator`, `best_age`,
  `early_declare`, `college_fantasy_ppg`, `power4_conf`, `recruit_rating`
- All PFF base metrics: `best_yprr`, `best_receiving_grade`, `best_routes_per_game`, `best_man_yprr`,
  `best_zone_yprr`, `best_man_zone_delta`, `best_slot_yprr`, `best_slot_target_rate`, `best_deep_yprr`,
  `best_deep_target_rate`, `best_drop_rate`, `best_target_sep`
- All athleticism: `speed_score`, `forty_time`, `broad_jump`, `vertical_jump`, `agility_score`,
  `weight_lbs`, `height_inches`

### Phase I training

Same LassoCV → Ridge pipeline as the capital model, using the position-specific `_CANDIDATE_FEATURES_NOCAP`
pool. The same training data (2014–2022) is used. The same nested LOYO-CV is used for evaluation.

**Artifacts:** Saved as `{POS}_ridge_nocap.pkl`, `{POS}_features_nocap.json`, `metadata_nocap.json`.

### Phase I LOYO R² (expected lower)

The Phase I model is expected to have substantially lower LOYO R² than the capital model, because draft
capital is the single most predictive feature. If Phase I LOYO R² > capital model LOYO R², that would
indicate a problem (either the capital model is overfitting, or Phase I is picking up a proxy for capital).

---

## 11. Cross-Validation Strategy

### Primary: Leave-One-Year-Out (LOYO)

The CV strategy is **temporal holdout**: for each draft year in the training set, train on all other
years and test on that year. This mimics the real use case (predicting a future draft class from
historical data).

**LOYO R²** is the primary reported evaluation metric. It is computed by concatenating all held-out
predictions across all years and computing R² against actual B2S scores.

**Per-year breakdown:** LOYO R² per held-out year is also reported. Years with fewer than 2 labeled
test observations are skipped.

### Nested Lasso in LOYO

Inside each LOYO fold, LassoCV is **re-run on that fold's training data** before fitting the Ridge
model. This is "nested CV" or "double CV" — it removes the selection leakage that would occur if
Lasso were run once on the full dataset and then LOYO were run with those features.

**Implementation:** The outer Lasso selects features for the final production model. The nested inner
Lasso (one per fold) selects features for that fold's Ridge. The two Lasso runs may select different
features — that is expected and correct.

**Assumption:** Using the outer-selected features as the candidate pool for nested inner Lasso (rather
than the full 40–50 candidate pool) is an acceptable simplification. This choice was made because using
the full candidate pool for nested Lasso created excessive fold-to-fold variability (selecting 4–11
features per fold depending on the held-out year), which injected noise without adding meaningful
information. Diagnostics confirmed the selection bias removed by nested Lasso is ~0.002 R² — small
but honest.

### RidgeCV Alpha Tuning Inside LOYO

Each fold's Ridge model uses RidgeCV with 5-fold cross-validation to select alpha from the
per-position alpha range. This means alpha is tuned on each fold's training data, not fixed at a
global value.

**Implication:** The LOYO R² reported is the CV estimate of the deployed model's performance, where
the deployed model uses RidgeCV alpha tuning on the full training set. The two alpha selections
(per-fold RidgeCV and final full-dataset RidgeCV) will often produce similar but not identical alphas.

### Rolling-Window CV (TE, optional for others)

For TE (automatically) and other positions when requested:
- Train on 2014–2018 (5 years, ~46 TE rows)
- Test on 2019–2022 (4 years, ~51 TE rows)

This is a "forward-looking" simulation: predict 4 recent years from 5 earlier years. More realistic
than LOYO but provides only one train-test split.

### K-Fold CV (TE, optional for others)

5-fold shuffle-split CV on the full labeled dataset. Better stability than LOYO when some holdout
years have very few observations (TE years with n < 10 produce unreliable per-year R²).

### Which R² to trust?

For WR/RB: LOYO R² is primary (N is large enough for temporal holdout to be meaningful).
For TE: All three should be considered together. LOYO can be noisy for years with n < 10 TEs. The
rolling window and k-fold provide additional perspectives.

---

## 12. ZAP Score Construction

### What ZAP represents

ZAP = percentile of a prospect's Ridge-predicted B2S relative to the distribution of Ridge predictions
on the **training set**.

### Formula

```python
zap_score = percentileofscore(train_preds_arr, prospect_pred, kind="weak")
```

Where:
- `train_preds_arr` = all in-sample Ridge predictions on the training set (stored in `metadata.json`)
- `prospect_pred` = the Ridge pipeline's raw prediction for the scoring prospect

`kind="weak"` means: fraction of training predictions that are **≤** prospect_pred, times 100.

**Bounded at [0, 100]:** A prospect whose predicted B2S exceeds the maximum training prediction gets
ZAP = 100.

### Reference distribution

The ZAP percentile is relative to the **training set predictions**, not the training set actual B2S
values. This is an important distinction:

- Training set actual B2S values include noise, injury, bad teams, etc.
- Training set predictions are the model's "smooth" version of outcomes
- The percentile of a prospect's prediction relative to training predictions tells us: "What fraction
  of historical draftees did this model predict worse than this prospect?"

**Assumption:** The training set prediction distribution is a reasonable reference for "what the model
knows." A ZAP of 80 means: "This model predicted 80% of 2014–2022 draftees at this position to be
worse than this prospect." It does NOT mean "This prospect will be at the 80th percentile of NFL
outcomes" — the LOYO R² of ~0.39–0.41 implies significant uncertainty.

### Why percentile instead of raw predicted B2S

Raw predicted B2S depends on the absolute scale of the training labels. After retraining with new data
or different features, the absolute values shift. A percentile score is scale-invariant and more
interpretable to end users ("Top 10% prospect" vs "predicted 14.3 PPG B2S").

---

## 13. Capital Delta and Risk Classification

### Capital Delta

```
capital_delta = ZAP (capital model) − Phase1_ZAP (no-capital model)
```

A positive delta means: the full model (including capital) ranks the player higher than the no-capital
model. The player is receiving credit from the NFL for something that the college/combine data does not
independently support.

A negative delta means: the full model ranks the player lower than the no-capital model. The player
has college production/efficiency data that the NFL has undervalued.

### Risk Classification

```
capital_delta ≥ +20  → "High Risk"    (capital far exceeds independent evidence)
capital_delta ≤ −15  → "Low Risk"     (model sees more than capital alone)
otherwise            → "Neutral"
```

**Assumption:** The thresholds +20 and −15 are not data-derived. They were chosen by judgment to
produce meaningful classification frequencies in the 2026 draft class. A vetter should examine whether
these thresholds produce stable classifications across historical draft classes (i.e., do 2019 High Risk
players actually underperform relative to their capital?).

### Relationship to JJ's Draft Capital Delta

JJ Zachariason's delta is computed as:
```
JJ delta = ZAP score − (capital component score only)
```

Where "capital component score" = a 0–100 percentile of pick number alone.

Our delta is different: we subtract the full no-capital model score (which includes production, PFF
efficiency, and athleticism) from the full capital model score. Our delta is more comprehensive —
it isolates the capital premium above ALL independent evidence, not just above the capital percentile.

**JJ's own guidance (2025 Prospect Guide):** "Draft Capital Delta can help us find gems, but just
remember that it's not the key metric we should be working off of. ZAP score will always be the top
priority."

---

## 14. Scoring Future Draft Classes (Inference)

### Draft capital at scoring time

When scoring the 2026 draft class **before the draft**, `draft_capital_score` is projected from
the player's consensus big board rank using the same exponential decay formula:

```
projected_pick = consensus_rank   (treating big board rank as pick number)
draft_capital_score = 100 × exp(−0.023 × (projected_pick − 1))
```

**Assumption:** Big board consensus rank is a good proxy for actual draft position. In practice,
players often go earlier or later than their big board rank. The capital score is re-computed after
the draft using actual pick numbers (via `--post-draft` flag).

**For players without a big board rank:** There is no fallback median imputation for pre-draft capital.
The player is excluded from scoring if consensus_rank is unknown. This avoids assigning a "median
capital" to an undrafted player who will not be drafted.

### Age at scoring time

For 2026 prospects, age at each season start is computed using the same DOB or recruit_year fallback
as the training set (§2.4). Breakout score and total yards rate are computed using the same formulas
applied to all available college seasons up to 2025.

---

## 15. Known Limitations and Unresolved Assumptions

**This section is the most important for a reviewer/vetter to focus on.**

### 15.1 Small N for TE

N=127 TEs. With ~10 selected features, this is a 12:1 observations-to-parameters ratio — acceptable
but not robust. Adding 2023 class will bring N to ~145. The TE model's LOYO R² variance is high
(range −0.26 to +0.70 across years).

### 15.2 Leakage from B2S label construction

B2S labels are constructed in nfl-fantasy-db using nflverse game logs. If any column in the training
data was accidentally derived from NFL data (e.g., a feature that correlates with the player's actual
NFL career), there is label leakage. This is guarded against by:
- Using only pre-draft features
- The code review confirming no nfl-fantasy-db data enters the training feature set

**However:** `consensus_rank` and `position_rank` (big board) are correlated with draft capital, which
is correlated with NFL team evaluations, which are informed by NFL team scouts who observe practices,
interviews, and medical exams. These signals carry information not in public college statistics. They
are not label leakage in the strict sense, but they do partially encode information that independent
college/combine metrics cannot.

### 15.3 Non-stationarity of college football

College football has evolved significantly from 2014 to 2022: the transfer portal, spread offenses,
increased passing rates. A breakout score computed from 2014 data may not be directly comparable to
2024 data. SP+ adjustments help at the team level but not at the era level.

### 15.4 PFF metrics are not stable across eras

PFF's grading methodology and route-running tracking have evolved. A "best_yprr" of 2.5 in 2014
may not be equivalent to 2.5 in 2022 because the player population being graded has changed.

### 15.5 Position-agnostic "best season" selection

Best season is selected by `rec_yards_per_team_pass_att` for all positions, including RBs. This
implicitly favors RBs who were used as receivers — a team's best rushing back who contributed little
in the passing game may have their "best season" selected as the year they caught passes, not the
year they dominated as a rusher. `best_total_yards_rate` partially corrects for this but uses
the same best-season selection metric.

### 15.6 Extrapolation in Phase I for elite athletic outliers

The Phase I model is trained on 2014–2022 players. When scoring 2026 prospects whose athletic profiles
(speed, vertical, broad jump) exceed the training maximum, the Ridge model extrapolates linearly.
Ridge's linear extrapolation is not bounded by the training range. Prospects who are athletic outliers
(e.g., a TE with the fastest forty time in training history AND an elite broad jump) will receive
Phase I scores that are pure extrapolation. Flag these cases: if a prospect's athleticism metrics are
near or beyond the training min/max, the Phase I score is an out-of-sample extrapolation.

### 15.7 Capital delta thresholds are arbitrary

The +20 / −15 thresholds for High Risk / Low Risk are not estimated from data. A proper validation
would compare post-draft outcomes for "High Risk" vs "Neutral" vs "Low Risk" classified prospects
from historical draft classes. This has not been done.

### 15.8 Breakout score formula not independently verified

The breakout score formula (rec_rate × SOS_mult × age_mult) is attributed to JJ Zachariason's
podcast transcript (Episode 1083). The SOS_mult formula specifically — `max(0.70, min(1.30, 1 + (sp_plus - 5) / 100))` — is our reconstruction from JJ's verbal description. JJ did not publish the
algebraic formula. If JJ's actual formula differs (different SP+ reference point, different multiplier
range, different age threshold), our breakout score may be systematically different from his.

### 15.9 Team pass attempts as RB denominator

Total yards rate uses team pass attempts as the denominator (§6.3). Total team plays would be more
appropriate for measuring RB contribution relative to offensive opportunity. Pass attempts are used
because total team plays are not stored in our database. On run-heavy teams (250 pass attempts,
600 total plays), this denominator is ~2.4x smaller than the "correct" denominator, inflating
total_yards_rate for players on those teams.

---

## 16. Assumption Inventory for Vetting

The following is a numbered checklist of every structural assumption embedded in this model.
An independent reviewer should evaluate each:

| # | Assumption | Where | Testable? |
|---|------------|-------|-----------|
| A1 | B2S = best 2 of first 3 NFL seasons, min 8 games per season | Target definition | Yes — compare against alternative thresholds (7 games, 10 games, 3-of-4 seasons) |
| A2 | Season 1 = draft year, even if player appeared in 0 games | Target definition | Unclear — JJ's implementation not confirmed |
| A3 | All regular-season weeks included in PPG (no week 18 exclusion) | Target definition | Weak — week 18 resting is real but effect is small |
| A4 | Training window 2014–2022 eliminates PFF era bias | Training window | Yes — compare LOYO R² with 2011+ start |
| A5 | Match score ≥ 80 threshold correctly separates right links from wrong links | Identity matching | Yes — audit sample below 80 |
| A6 | Median imputation is appropriate for missing combine data | Missing data | Partial — "declined to run" players are not MAR |
| A7 | Median imputation is appropriate for missing PFF data | Missing data | No — pre-2014 players are not missing at random |
| A8 | Age = 18.5 + (season_year − recruit_year) for players without DOB | Feature engineering | Yes — compare against known DOBs in sample |
| A9 | Best season = season with highest rec_yards_per_team_pass_att | Feature engineering | Yes — compare against alternatives (most recent season, career average) |
| A10 | Option-offense filter threshold = 200 team pass attempts | Feature engineering | Yes — test 150, 250 thresholds |
| A11 | Breakout score formula matches JJ's actual formula | Feature engineering | Partially — sourced from podcast transcript reconstruction |
| A12 | Total yards rate denominator = team pass attempts (not total plays) | Feature engineering | Yes — quantify bias for run-heavy teams |
| A13 | Draft capital score = 100 × exp(−0.023 × (pick − 1)) | Feature engineering | Yes — compare against alternative curves (Massey-Thaler, linear) |
| A14 | Multiplicative interaction terms are the right functional form | Feature engineering | Partial — compare additive-only model |
| A15 | Draft tiers = {1–16, 17–50, 51–100, 100+} | Feature engineering | Yes — try different breakpoints |
| A16 | Top-16 RBs are a qualitatively different regime | Feature engineering | Yes — JJ cites R²=0.526; verify in our data |
| A17 | Lasso selects, Ridge predicts is appropriate | Model architecture | Partial — compare against Elastic Net, LASSO-predict |
| A18 | TE Ridge alpha range [1–1000] is appropriate for N=127 | Model architecture | Yes — try [0.1–100] for TE and compare LOYO R² |
| A19 | Nested Lasso on outer-selected features (not full pool) is honest | CV methodology | Yes — compare against full nested pool |
| A20 | LOYO R² is the primary evaluation metric | CV methodology | Partial — compare rolling window and k-fold |
| A21 | Training prediction distribution is the right ZAP reference | ZAP construction | Yes — compare against actual B2S percentile distribution |
| A22 | Capital delta thresholds: +20 = High Risk, −15 = Low Risk | Risk classification | Yes — validate against historical outcomes |
| A23 | Phase I includes recruit_rating (independent of capital) | Phase I design | Yes — recruiting rankings are partially informed by early NFL interest |
| A24 | consensus_rank as proxy for draft pick in pre-draft scoring | Inference | Yes — check distribution of consensus_rank vs actual pick for historical classes |
| A25 | PFF split values stored as strings (cast to float before arithmetic) | Data pipeline | Yes — auditable from raw database values |
| A26 | Power-4 conference binary is meaningful after SP+ adjustment | Feature engineering | Yes — if SP+ correctly adjusts for schedule, power4_conf should add no information |

---

*End of methodology documentation.*

*Document reflects model state as of 2026-03-14. Key scripts: `scripts/build_training_set.py`,
`scripts/analyze.py::engineer_features()`, `scripts/fit_model.py`, `scripts/score_class.py`.*
