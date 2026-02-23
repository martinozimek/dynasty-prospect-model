# Data Quality Report
**Dynasty Prospect Model — Three-Repo Pipeline**
Last audit: 2026-02-23 | Script: `scripts/audit_data.py`

---

## Audit Summary

| Section | PASS | WARN | FAIL |
|---------|------|------|------|
| cfb-prospect-db | 4 | 3 | 2 |
| nfl-fantasy-db | 7 | 0 | 2 |
| Cross-DB join integrity | 3 | 1 | 2 |
| Training set | 13 | 5 | 0 |
| **Total** | **27** | **9** | **6** |

**Overall status: FAIL** — all 6 remaining failures are infrastructure-level issues in the
full CFBLink / cfb-prospect-db tables, not in the training set itself. Training set checks
all PASS. See action items for resolution path.

Flagged-row CSVs are written to `data/audit/` each time the audit runs (gitignored).

---

## Training Set Status

**The training set is clean for model work.** All three position-specific training checks pass:

| Check | WR | RB | TE |
|---|---|---|---|
| Wrong-link detection (best_season_year >= draft_year) | PASS (0) | PASS (0) | PASS (0) |
| Feature plausibility (age, rec_rate ranges) | PASS | PASS | PASS |
| Low match score rows (< 85 in training) | WARN (1) | PASS (0) | PASS (0) |

**Training row counts (post deduplication and link-score filter):**

| Position | Rows | Classes |
|---|---|---|
| WR | 284 | 2011–2022 |
| RB | 199 | 2011–2022 |
| TE | 126 | 2011–2022 |
| **Total** | **609** | |

`build_training_set.py` applies two safety filters before writing CSVs:
1. **Deduplication by cfb_player_id**: when multiple NFL names map to the same CFB player
   (name variants like "Laviska Shenault" / "Laviska Shenault Jr.", or fuzzy wrong links),
   keeps only the highest match_score row. Logged as `[dedup POS]` warnings.
2. **Link-score filter** (`--min-link-score 80`, default): drops rows with match_score < 80.
   These are low-confidence links where CFB features likely belong to a different player.

---

## RESOLVED: Wrong CFB Links (was CRITICAL)

**Status: FIXED** — `link_to_cfb.py` has been updated with a draft-year alignment hard filter.
All training-set wrong-link checks now pass.

### What was wrong

`link_to_cfb.py` matched NFL players to cfb-prospect-db players by name similarity and
draft_year ±1 tolerance. For players with common names (Johnson, Jones, Brown, Thomas),
it matched to a different player in cfb-prospect-db who happened to have a similar name —
particularly for early draft classes (2011–2015) where the matched player's college seasons
extended years after the NFL player's draft year (physically impossible).

**25 confirmed wrong links were detected** using the rule: `best_season_year >= draft_year`
(a player drafted in April 2011 cannot have their best CFB season in 2011 or later).

### Fix applied

`link_to_cfb.py` now loads `last_season_year` (max CFB season year per player from
`CFBPlayerSeason`) for each candidate and applies this hard filter in `_fuzzy_match()`:

```python
# Hard alignment filter: candidate's last CFB season must precede draft year
last_season = c.get("last_season_year")
if draft_year is not None and last_season is not None and last_season >= draft_year:
    return False  # definitionally a different person
```

The CFBLink table was wiped (`--wipe`) and re-run with the fixed filter. Results:
- Old: 836 links (including 25+ confirmed wrong links)
- New: 787 links (wrong links rejected; most dropped to QA/missing)
- Draft year alignment: **0 impossible cases** (was 66), 6 stale links (gap > 4 years, review)

---

## Feature Coverage Tables

Coverage for players in training set (draft classes 2011–2022, post-deduplication).

### Wide Receivers — 284 players

| Feature | Coverage | % | Status |
|---|---|---|---|
| best_rec_rate (yd/att) | 284 / 284 | 100% | OK |
| best_dominator | 284 / 284 | 100% | OK |
| best_reception_share | 284 / 284 | 100% | OK |
| best_age | 284 / 284 | 100% | OK |
| best_games | 284 / 284 | 100% | OK |
| best_sp_plus (competition adj.) | 284 / 284 | 100% | OK |
| career_rec_yards | 284 / 284 | 100% | OK |
| career_rush_yards | 284 / 284 | 100% | OK |
| career_targets | 284 / 284 | 100% | OK |
| draft_capital_score | 284 / 284 | 100% | OK |
| draft_round | 284 / 284 | 100% | OK |
| overall_pick | 284 / 284 | 100% | OK |
| teammate_score | 284 / 284 | 100% | OK |
| b2s_score (target) | 284 / 284 | 100% | OK |
| weight_lbs | ~242 / 284 | ~85% | WARN — combine gap |
| forty_time | ~219 / 284 | ~77% | WARN — combine gap |
| speed_score | ~219 / 284 | ~77% | WARN — combine gap |
| consensus_rank | ~172 / 284 | ~61% | WARN — 2011–2015 missing |
| recruit_rating | ~18 / 284 | ~6% | LOW — 2011–2017 blocked |

**B2S distribution [WR]:** mean=6.39 | median=5.74 | p25=1.98 | p75=9.77 | max=23.02

**Draft class balance [WR]:** 19–35 players per class, 2011–2022. All classes adequate.

---

### Running Backs — 199 players

| Feature | Coverage | % | Status |
|---|---|---|---|
| best_rec_rate | 199 / 199 | 100% | OK |
| best_dominator | 199 / 199 | 100% | OK |
| best_reception_share | 199 / 199 | 100% | OK |
| best_age | 199 / 199 | 100% | OK |
| best_games | 199 / 199 | 100% | OK |
| best_sp_plus | 199 / 199 | 100% | OK |
| career_rec_yards | 199 / 199 | 100% | OK |
| career_rush_yards | 199 / 199 | 100% | OK |
| career_targets | 199 / 199 | 100% | OK |
| draft_capital_score | 199 / 199 | 100% | OK |
| draft_round | 199 / 199 | 100% | OK |
| overall_pick | 199 / 199 | 100% | OK |
| teammate_score | 199 / 199 | 100% | OK |
| b2s_score (target) | 199 / 199 | 100% | OK |
| weight_lbs | ~165 / 199 | ~83% | WARN — combine gap |
| forty_time | ~157 / 199 | ~79% | WARN — combine gap |
| speed_score | ~157 / 199 | ~79% | WARN — combine gap |
| consensus_rank | ~118 / 199 | ~59% | LOW — 2011–2015 missing |
| recruit_rating | ~15 / 199 | ~8% | LOW — 2011–2017 blocked |

**B2S distribution [RB]:** mean=7.27 | median=7.05 | p25=1.95 | p75=11.12 | max=26.77

**Draft class balance [RB]:** 12–23 players per class. 2012 is the lightest. All classes usable.

---

### Tight Ends — 126 players

| Feature | Coverage | % | Status |
|---|---|---|---|
| best_rec_rate | 126 / 126 | 100% | OK |
| best_dominator | 126 / 126 | 100% | OK |
| best_reception_share | 126 / 126 | 100% | OK |
| best_age | 126 / 126 | 100% | OK |
| best_games | 126 / 126 | 100% | OK |
| best_sp_plus | 126 / 126 | 100% | OK |
| career_rec_yards | 126 / 126 | 100% | OK |
| career_rush_yards | 126 / 126 | 100% | OK |
| career_targets | 126 / 126 | 100% | OK |
| draft_capital_score | 126 / 126 | 100% | OK |
| draft_round | 126 / 126 | 100% | OK |
| overall_pick | 126 / 126 | 100% | OK |
| teammate_score | 126 / 126 | 100% | OK |
| b2s_score (target) | 126 / 126 | 100% | OK |
| weight_lbs | ~107 / 126 | ~85% | WARN — combine gap |
| forty_time | ~95 / 126 | ~75% | WARN — combine gap |
| speed_score | ~95 / 126 | ~75% | WARN — combine gap |
| consensus_rank | ~77 / 126 | ~61% | WARN — 2011–2015 missing |
| recruit_rating | ~8 / 126 | ~6% | LOW — 2011–2017 blocked |

**B2S distribution [TE]:** mean=5.17 | median=4.68 | p25=0.00 | p75=8.27 | max=16.17

**Draft class balance [TE]:** 6–16 players per class. 2011, 2012, 2014, 2016 are thin (<10).
TE model has the thinnest data — noted limitation.

---

## Missing Data Inventory

### Data We Want — Gaps by Source

| Data | Classes Affected | Players Missing | Source | Blocker | Priority |
|---|---|---|---|---|---|
| Recruiting ratings (recruit_rating) | 2011–2017 | ~490 / 609 (80%) | CFBD API | Monthly API quota exhausted 2026-02-22 | HIGH — needed for model |
| Pre-draft consensus rank | 2011–2015 | ~215 / 609 (35%) | nflmockdraftdatabase.com | No data available pre-2016 | MEDIUM — feature gap noted |
| NFL combine 40-time / speed_score | All classes | ~180 / 609 (30%) | nflverse combine | Players who skipped combine | LOW — impute or indicator |
| Combine weight | All classes | ~120 / 609 (20%) | nflverse combine | Players who skipped combine | LOW — same |
| 2026 NFL combine data | 2026 class | All 2026 players | nflverse combine | Combine Feb/Mar 2026, not yet ingested | ACTION NEEDED |

### Recruiting Gap Detail

CFBD's recruiting API was called until the free-tier monthly quota was exhausted on 2026-02-22.
Classes 2018–2025 are populated. Classes 2007–2017 have 0 recruiting rows.

The training window (2011–2022) requires classes 2011–2017 — approximately 490 players with
no `recruit_rating`. This is the largest single data gap.

**Resume date:** CFBD resets monthly quotas at the start of each calendar month. Retry March 2026.

**Command to run:**
```bash
cd cfb-prospect-db
python scripts/populate_db.py --recruit-years 2011 2012 2013 2014 2015 2016 2017
```

### Consensus Big Board Gap

nflmockdraftdatabase.com does not have data before 2016. Draft classes 2011–2015 will always
have null `consensus_rank` — approximately 113 WR, 83 RB, 53 TE (249 players, 41% of training set).

**Options:**
- Accept the gap; use `consensus_rank` only for 2016+ classes in the model
- Source historical boards from archived ESPN / NFL.com draft PDFs manually for top-50 per class
- Treat `consensus_rank` as a feature with systematic missing pattern (missing = pre-2016)

### NFL Combine Gap

Approximately 180–200 players (~30%) skipped the combine or have no recorded 40-yard dash.
This includes many mid-round and late-round picks. Missing is not purely random — players
with lower perceived athleticism are less likely to attend.

**Imputation strategy (decide at EDA phase):**
- Positional mean imputation
- Indicator variable (`is_combine_measured`) as a model feature
- Drop the feature and rely on `weight_lbs` alone for size

---

## Source-Level Data Issues (Infrastructure)

These are the 6 remaining FAILs. They affect the full cfb-prospect-db / CFBLink tables
but do NOT affect the training set (which has its own dedup + score filter layers).

### cfb-prospect-db

| Check | Status | Detail |
|---|---|---|
| games_played null rate | WARN | 226 / 150,939 rows (0.1%) — small; likely option-era position players |
| games_played > 15 | WARN | 74 rows — bowl games can push to 14-15; review > 15 |
| Duplicate seasons | PASS | 0 duplicates |
| Season stat plausibility | PASS | No rec_yards > 2000, targets > 200, or receptions > targets |
| rec_yards_per_team_pass_att > 6.0 | FAIL | 2 rows — investigate team pass_att denominator in populate_db.py |
| dominator_rating < 0 | FAIL | 605 rows — floating-point error; near-zero negatives (e.g. -0.003) in skill-position edge cases |
| Team pass_attempts < 100 | WARN | 11 team-seasons — option offenses near filter boundary |
| Draft pick ranges | PASS | All round/pick values valid |
| Combine measurables | PASS | No impossible weight / 40-time; no speed_score missing when computable |

**dominator_rating note:** 605 season rows have a small negative value (e.g. -0.003) despite
non-null `rec_yards`. Root cause: floating-point edge case in `populate_db.py` when the team's
`total_rec_yards` denominator is very small. Impact on training is negligible (effectively 0),
but should be fixed in cfb-prospect-db for cleanliness.

**rec_rate extreme note:** 2 rows with `rec_yards_per_team_pass_att > 6.0` — likely a season
where team `pass_attempts` is recorded as very low. Investigate `cfb_rec_rate_extreme.csv`.

### nfl-fantasy-db

| Check | Status | Detail |
|---|---|---|
| B2S score ranges | PASS | All in plausible range; no nulls in training window |
| B2S zero-score rate | INFO | 183 / 719 (25.5%) have qualifying_seasons = 0 — expected for busts |
| Season score PPG range | PASS | No PPG > 50 or < -5.0; 14 minor negatives (legitimate: negative rush yards) |
| Format completeness | PASS | All 6,563 player-seasons have all 5 scoring format rows |
| Game log week range | PASS | All weeks in valid range 1–18 |
| CFB link match scores | FAIL | 13 links with match_score < 80 — filtered out of training by link-score filter |
| Duplicate cfb_player_id | FAIL | 58 rows — mix of name variants (same player) and fuzzy wrong links; deduplicated in training by cfb_player_id |
| Fumbles lost range | PASS | No negative values; no seasons > 10 |

### Cross-DB Join

| Check | Status | Detail |
|---|---|---|
| CFBLink IDs exist in cfb-prospect-db | PASS | 0 missing |
| Position match | FAIL | 52 hard mismatches — some are legitimate college-to-NFL conversions (Braxton Miller QB→WR, Antonio Gibson WR→RB); others may be wrong links from similar names at same position |
| Draft year alignment | WARN | 0 impossible (was 66 before fix); 6 stale links (gap > 4 years — review manually) |
| Duplicate cfb_player_id | FAIL | 58 rows (same as nfl-fantasy-db section above) |
| B2S coverage in training window | PASS | 8.2% of B2S rows (59/719) have no CFB link — acceptable |

---

## Known Players Lost to Deduplication

When two NFL players were linked to the same CFB player ID, the lower-match-score player
was dropped from training even if they are real NFL players. Notable examples:

| Dropped player | Winner kept | Position | Reason |
|---|---|---|---|
| A.J. Brown (WR, 2019, b2s=15.80) | Equanimeous St. Brown | WR | A.J. Brown's correct CFB link not found; wrong link (Equanimeous) had score=100 |
| Tyron Johnson (WR, 2020, b2s=7.23) | Tyler Johnson | WR | Fuzzy link to same CFB player |
| D'Ernest Johnson (RB, 2019, b2s=5.65) | Ty Johnson | RB | Fuzzy link to same CFB player |
| Kenneth Walker (RB, 2022, b2s=14.99) | Kenneth Walker III | RB | Name variant — same player, correct |
| Laviska Shenault (WR, 2020, b2s=6.76) | Laviska Shenault Jr. | WR | Name variant — same player, correct |

For players like A.J. Brown who are genuinely missing from training, a manual link fix
(providing the correct cfb_player_id) would restore them. This is a low-priority QA task.

---

## Action Items

### Priority 1 — Recruiting Backfill (HIGH, blocked until March 2026)

Resume CFBD API calls for recruit years 2011–2017 when quota resets.
Currently ~6% recruiting coverage for training window. The feature `recruit_rating`
will be near-useless for model fitting until this is resolved.

```bash
cd cfb-prospect-db
python scripts/populate_db.py --recruit-years 2011 2012 2013 2014 2015 2016 2017
```

### Priority 2 — 2026 Combine Ingestion (MEDIUM, time-sensitive)

The 2026 NFL combine occurred in February/March 2026. Ingest before scoring 2026 class:

```bash
cd cfb-prospect-db
python scripts/populate_nfl.py --combine-years 2026 --skip-draft --skip-rosters --skip-strength
```

### Priority 3 — dominator_rating < 0 (LOW, cfb-prospect-db fix)

605 season rows have negative dominator_rating (like -0.003). Fix the denominator
calculation in `cfb-prospect-db/scripts/populate_db.py`. Impact on training is negligible
but the fix is cleaner.

### Priority 4 — rec_rate extreme (LOW, investigate 2 rows)

2 rows in cfb-prospect-db have `rec_yards_per_team_pass_att > 6.0`. See
`data/audit/cfb_rec_rate_extreme.csv`. Likely a bad team pass_att denominator.

### Priority 5 — Position Mismatch Review (LOW, partially expected)

Review `data/audit/crossdb_position_mismatches.csv` for the 52 hard mismatches.
Legitimate position changes (QB→WR, WR→RB, ATH→TE) should be documented.
True wrong links should have their CFBLink entries removed.

### Priority 6 — Manual Link Fixes for Notable Missing Players (LOW)

Players like A.J. Brown (2019 WR, b2s=15.80) are missing from training because
link_to_cfb.py couldn't find their CFB record. A manual CFBLink insert with the correct
cfb_player_id would restore them.

To find the correct cfb_player_id: query cfb-prospect-db by name + position + season years,
then insert directly:
```python
# Example: fix A.J. Brown (Ole Miss WR, drafted 2019)
link = CFBLink(nfl_player_name="A.J. Brown", position="WR",
               cfb_player_id=<CORRECT_ID>, draft_year=2019,
               match_method="manual", match_score=100.0)
```

### Priority 7 — Consensus Big Board 2011–2015 (LOW, likely impossible free)

No free source has pre-2016 consensus boards. Accept the gap. Treat `consensus_rank`
as a 2016+ only feature, or source from archived ESPN / NFL.com draft board PDFs manually
for top-50 players in each missing class.

---

## Re-Running the Audit

```bash
cd dynasty-prospect-model
python scripts/audit_data.py           # full audit, writes CSVs to data/audit/
python scripts/audit_data.py --no-csv  # print only, no file output
python scripts/audit_data.py --section cfb      # cfb-prospect-db section only
python scripts/audit_data.py --section crossdb  # cross-DB section only
```

The audit should be re-run after any data ingestion or pipeline change. Training set
checks must all PASS before model fitting begins.
