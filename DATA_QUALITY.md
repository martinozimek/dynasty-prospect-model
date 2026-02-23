# Data Quality Report
**Dynasty Prospect Model — Three-Repo Pipeline**
Last audit: 2026-02-23 | Script: `scripts/audit_data.py`

---

## Audit Summary

| Section | PASS | WARN | FAIL |
|---------|------|------|------|
| cfb-prospect-db | 4 | 3 | 2 |
| nfl-fantasy-db | 7 | 0 | 2 |
| Cross-DB join integrity | 2 | 0 | 3 |
| Training set | 6 | 7 | 6 |
| **Total** | **19** | **10** | **13** |

**Overall status: FAIL** — model work must not begin until CRITICAL items below are resolved.

Flagged-row CSVs are written to `data/audit/` each time the audit runs.

---

## CRITICAL: Wrong CFB Links

**This is the most important finding. ~8-10% of training rows use the wrong college player's stats.**

### What happened

`scripts/link_to_cfb.py` matches NFL players to cfb-prospect-db players by name similarity.
For players with common names (Johnson, Jones, Brown, Thomas, etc.), it matched to a different
player in cfb-prospect-db who happened to have the same or similar name — especially for early
draft classes (2011–2015) where the matched player's college seasons extend years after the NFL
player's draft year (which is impossible).

### Detection method

If a training row's `best_season_year >= draft_year`, the CFB player in the link cannot be the
correct person. A player drafted in April 2022 cannot have their "best" college season in 2022.

### Confirmed wrong links in training set (25 rows)

**Wide Receivers — 14 wrong links**

| NFL Name | Incorrectly linked to | Draft Year | CFB Best Season | Match Score |
|---|---|---|---|---|
| A.J. Green | A.J. Green (wrong player) | 2011 | 2017 | 100.0 |
| Cecil Shorts | Cecil Doggette Jr. | 2011 | 2024 | 85.5 |
| Stephen Burton | CJ Burton | 2011 | 2023 | 85.5 |
| Marvin Jones | Marvin Jones Jr. | 2012 | 2023 | 95.0 |
| T.J. Houshmandzadeh | T.J. Graham | 2012 | 2019 | 85.5 |
| Charles D. Johnson | Carlos Johnson | 2013 | 2021 | 81.3 |
| Darius Johnson | Carlos Johnson | 2013 | 2021 | 78.6 |
| Damaris Johnson | Carlos Johnson | 2013 | 2021 | 75.9 |
| John Brown | John Brown III | 2014 | 2017 | 95.0 |
| Jeff Janis | Jeff Undercuffler | 2014 | 2023 | 85.5 |
| Will Fuller | Will Anderson Jr. | 2016 | 2021 | 85.5 |
| David Moore | David Moore (wrong player) | 2017 | 2019 | 100.0 |
| Mike Strachan | Mike Yoan Sandjo-Njiki | 2021 | 2023 | 85.5 |
| Mike Woods | Mike Yoan Sandjo-Njiki | 2021 | 2023 | 85.5 |

**Running Backs — 6 wrong links**

| NFL Name | Incorrectly linked to | Draft Year | CFB Best Season | Match Score |
|---|---|---|---|---|
| Daniel Thomas | Daniel Thomas (wrong player) | 2011 | 2016 | 100.0 |
| Chris Thompson | Chris Thompson (wrong player) | 2013 | 2016 | 100.0 |
| James White | James White (wrong player) | 2014 | 2015 | 100.0 |
| Marcus Murphy | Marcus Murphy (wrong player) | 2015 | 2018 | 100.0 |
| Brandon Wilds | Brandon Wilson | 2017 | 2017 | 88.9 |
| Trey Ragas | Trey Eberhart III | 2021 | 2025 | 85.5 |

**Tight Ends — 5 wrong links**

| NFL Name | Incorrectly linked to | Draft Year | CFB Best Season | Match Score |
|---|---|---|---|---|
| Julius Thomas | Thomas Bertrand-Hudon | 2011 | 2022 | 85.5 |
| David Thomas | Thomas Bertrand-Hudon | 2011 | 2022 | 85.5 |
| Adam Shaheen | Adam Vinatieri Jr. | 2017 | 2025 | 85.5 |
| Eric Saubert | Eric Singleton Jr. | 2017 | 2024 | 85.5 |
| James Mitchell | James Mitchell (same name, wrong player?) | 2022 | 2022 | 100.0 |

### Additional flagged links (not yet confirmed wrong)

Beyond the 25 confirmed above, the cross-DB audit flagged:
- **66 links** where the linked CFB player's last season is >= the NFL player's draft year
- **33 unique cfb_player_ids** shared across 69 NFL players (one CFB player linked to multiple NFL names)
- **15 links** with match_score < 80 (clearly low-confidence)

See `data/audit/crossdb_draft_year_alignment.csv` and `data/audit/nfl_link_dup_cfb_ids.csv`.

### Fix required

Re-run `link_to_cfb.py` with a draft-year constraint: when matching, require the linked
CFB player's last season year < NFL player's draft_year. Use this as a hard filter, not just
a tiebreaker. For confirmed wrong links, null out the cfb_player_id and flag for manual review.

---

## Feature Coverage Tables

Coverage for players in training window (draft classes 2011–2022).

### Wide Receivers — 309 players

| Feature | Coverage | % | Status |
|---|---|---|---|
| best_rec_rate (yd/att) | 309 / 309 | 100% | OK |
| best_dominator | 309 / 309 | 100% | OK |
| best_reception_share | 309 / 309 | 100% | OK |
| best_age | 309 / 309 | 100% | OK |
| best_games | 309 / 309 | 100% | OK |
| best_sp_plus (competition adj.) | 309 / 309 | 100% | OK |
| career_rec_yards | 309 / 309 | 100% | OK |
| career_rush_yards | 309 / 309 | 100% | OK |
| career_targets | 309 / 309 | 100% | OK |
| draft_capital_score | 309 / 309 | 100% | OK |
| draft_round | 309 / 309 | 100% | OK |
| overall_pick | 309 / 309 | 100% | OK |
| teammate_score | 309 / 309 | 100% | OK |
| b2s_score (target) | 309 / 309 | 100% | OK |
| weight_lbs | 265 / 309 | 85.8% | WARN — combine gap |
| forty_time | 236 / 309 | 76.4% | WARN — combine gap |
| speed_score | 236 / 309 | 76.4% | WARN — combine gap |
| consensus_rank | 182 / 309 | 58.9% | LOW — 2011–2015 missing |
| recruit_rating | 26 / 309 | 8.4% | LOW — 2011–2017 blocked |

**B2S distribution [WR]:** mean=6.38 | median=5.74 | p25=1.98 | p75=9.77 | max=23.02

**Draft class balance [WR]:** 21–35 players per class, 2011–2022. All classes adequate.

---

### Running Backs — 211 players

| Feature | Coverage | % | Status |
|---|---|---|---|
| best_rec_rate | 210 / 211 | 99.5% | OK |
| best_dominator | 210 / 211 | 99.5% | OK |
| best_reception_share | 210 / 211 | 99.5% | OK |
| best_age | 211 / 211 | 100% | OK |
| best_games | 211 / 211 | 100% | OK |
| best_sp_plus | 210 / 211 | 99.5% | OK |
| career_rec_yards | 211 / 211 | 100% | OK |
| career_rush_yards | 211 / 211 | 100% | OK |
| career_targets | 211 / 211 | 100% | OK |
| draft_capital_score | 211 / 211 | 100% | OK |
| draft_round | 211 / 211 | 100% | OK |
| overall_pick | 211 / 211 | 100% | OK |
| teammate_score | 211 / 211 | 100% | OK |
| b2s_score (target) | 211 / 211 | 100% | OK |
| weight_lbs | 175 / 211 | 82.9% | WARN — combine gap |
| forty_time | 167 / 211 | 79.1% | WARN — combine gap |
| speed_score | 167 / 211 | 79.1% | WARN — combine gap |
| consensus_rank | 123 / 211 | 58.3% | LOW — 2011–2015 missing |
| recruit_rating | 16 / 211 | 7.6% | LOW — 2011–2017 blocked |

**B2S distribution [RB]:** mean=7.18 | median=6.99 | p25=1.78 | p75=11.07 | max=26.77

**Draft class balance [RB]:** 12–24 players per class. 2012 is the lightest (12). All classes usable.

---

### Tight Ends — 135 players

| Feature | Coverage | % | Status |
|---|---|---|---|
| best_rec_rate | 132 / 135 | 97.8% | OK |
| best_dominator | 132 / 135 | 97.8% | OK |
| best_reception_share | 132 / 135 | 97.8% | OK |
| best_age | 135 / 135 | 100% | OK |
| best_games | 135 / 135 | 100% | OK |
| best_sp_plus | 132 / 135 | 97.8% | OK |
| career_rec_yards | 135 / 135 | 100% | OK |
| career_rush_yards | 135 / 135 | 100% | OK |
| career_targets | 135 / 135 | 100% | OK |
| draft_capital_score | 135 / 135 | 100% | OK |
| draft_round | 135 / 135 | 100% | OK |
| overall_pick | 135 / 135 | 100% | OK |
| teammate_score | 135 / 135 | 100% | OK |
| b2s_score (target) | 135 / 135 | 100% | OK |
| weight_lbs | 115 / 135 | 85.2% | WARN — combine gap |
| forty_time | 102 / 135 | 75.6% | WARN — combine gap |
| speed_score | 102 / 135 | 75.6% | WARN — combine gap |
| consensus_rank | 82 / 135 | 60.7% | WARN — 2011–2015 missing |
| recruit_rating | 11 / 135 | 8.1% | LOW — 2011–2017 blocked |

**B2S distribution [TE]:** mean=5.12 | median=4.65 | p25=0.00 | p75=8.27 | max=16.17

**Draft class balance [TE]:** 7–17 players per class. 2014 (7) and 2016 (7) are thin.

---

## Missing Data Inventory

### Data We Want — Gaps by Source

| Data | Classes Affected | Players Missing | Source | Blocker | Priority |
|---|---|---|---|---|---|
| Recruiting ratings (recruit_rating) | 2011–2017 | ~490 / 655 (75%) | CFBD API | Monthly API quota exhausted 2026-02-22 | HIGH — needed for model |
| Pre-draft consensus rank | 2011–2015 | ~215 / 655 (33%) | nflmockdraftdatabase.com | No data available pre-2016 | MEDIUM — feature gap noted |
| NFL combine times (forty_time) | All classes | ~180 / 655 (27%) | nflverse combine | Players who did not attend combine | LOW — missing at random, handle with imputation |
| Combine weight | All classes | ~120 / 655 (18%) | nflverse combine | Players who did not attend combine | LOW — same |
| 2026 NFL combine data | 2026 class | All 2026 players | nflverse combine | Combine occurred Feb/Mar 2026, not yet ingested | ACTION NEEDED |

### Recruiting Gap Detail

CFBD's recruiting API was called until the free-tier monthly quota was exhausted on 2026-02-22.
Classes 2018–2025 are populated. Classes 2007–2017 have 0 recruiting rows.

The training window (2011–2022) requires classes 2011–2017 — approximately 490 players with
no recruit_rating. This is the largest single data gap.

**Resume date:** CFBD resets monthly quotas at the start of each calendar month. Retry in March 2026.

**Command to run:**
```
cd cfb-prospect-db
python scripts/populate_db.py --recruit-years 2011 2012 2013 2014 2015 2016 2017
```

### Consensus Big Board Gap

nflmockdraftdatabase.com does not have data before 2016. Draft classes 2011–2015 will always
have null consensus_rank — approximately 113 WR, 83 RB, 53 TE (249 players, 38% of training set).

**Options:**
- Accept the gap and use consensus_rank only for 2016+ classes
- Source a historical big board from an archived publication (e.g., ESPN draft board archives)
- Treat consensus_rank as a feature only applicable to recent classes in the model

### NFL Combine Gap

Approximately 180-200 players (~27%) skipped the combine or have no recorded 40-yard dash.
This includes many mid-round and late-round picks. Missing is not random — players with
lower perceived athleticism are less likely to have combine data.

**Imputation strategy (decide at EDA phase):**
- Positional mean imputation
- Indicator variable (is_combine_measured) as a model feature
- Drop the feature and rely on weight_lbs alone for size

---

## Source-Level Data Issues

### cfb-prospect-db

| Check | Status | Detail |
|---|---|---|
| games_played null rate | WARN | 226 / 150,939 rows (0.1%) — small, likely option-era position players |
| games_played > 15 | WARN | 74 rows — bowl games can push to 14-15; review if > 15 |
| Duplicate seasons | PASS | 0 duplicates |
| Season stat plausibility | PASS | No rec_yards > 2000, targets > 200, or receptions > targets |
| rec_yards_per_team_pass_att | FAIL | 2 extreme rows (> 6.0) — investigate team pass_att denominator |
| dominator_rating < 0 | FAIL | 605 rows — floating-point error on rows with near-zero numerator; investigate in populate_db.py |
| Team pass_attempts < 100 | WARN | 11 team-seasons — option offenses that passed through MIN_TEAM_PASS_ATT filter |
| Draft pick ranges | PASS | All round/pick values in valid range |
| Combine measurables | PASS | No impossible weight/40-time values; no speed_score missing when computable |

**Dominator_rating note:** 605 season rows have a small negative dominator_rating (values like
-0.003) despite having recorded receiving yards. Root cause is likely a floating-point edge case
in `populate_db.py` when the team's total_rec_yards denominator is very small. These rows affect
non-training skill-position fringe players; confirm whether any appear in training set.

### nfl-fantasy-db

| Check | Status | Detail |
|---|---|---|
| B2S score ranges | PASS | All in plausible range per position; no nulls in training window |
| B2S zero-score rate | INFO | 183 / 719 (25.5%) have qualifying_seasons = 0 — expected for busts and late-round picks |
| Season score PPG range | PASS | No PPG > 50 or < -5.0; minor negatives (14 rows) are legitimate negative rushing yards |
| Format completeness | PASS | All 6,563 player-seasons have all 5 scoring format rows |
| Game log week range | PASS | All weeks in valid range 1–18 |
| CFB link match scores | FAIL | 15 links with match_score < 80 — almost certainly wrong players |
| Duplicate cfb_player_id | FAIL | 33 cfb_player_ids shared across 69 NFL links — multiple NFL players linked to same CFB player |
| Fumbles lost range | PASS | No negative values; no seasons > 10 |

### Cross-DB Join

| Check | Status | Detail |
|---|---|---|
| CFBLink IDs all exist in cfb-prospect-db | PASS | 0 missing |
| Position match | FAIL | 103 hard mismatches — most are wrong links (see Critical section above); some are legitimate college-to-NFL position changes (Braxton Miller QB->WR, Antonio Gibson WR->RB) |
| Draft year alignment | FAIL | 66 links with gap < 1 (last CFB season >= draft year) — confirms wrong-link pattern |
| Duplicate cfb_player_id | FAIL | 69 rows (same as 2.7 above) |
| B2S coverage in training window | PASS | 100% of B2S rows in training window have a link |

---

## Action Items

Ordered by priority (must fix before model work begins):

### Priority 1 — Fix Wrong CFB Links (BLOCKING)

1. Update `link_to_cfb.py` to require `last_cfb_season < draft_year` as a hard filter
   when selecting among candidate matches.
2. For the 25 confirmed wrong-link rows (see tables above), null out the `cfb_player_id`
   in `cfb_link` and flag `match_method = 'failed'`.
3. Re-run linking for all classes 2011–2022. Verify the 63 flagged alignment cases resolve.
4. Re-run `build_training_set.py` and re-run audit to confirm wrong-link FAILs are gone.

**Expected outcome:** Training set shrinks slightly (formerly wrong-linked players will have no
link) but remaining rows will be trustworthy.

### Priority 2 — Recruiting Backfill (HIGH, blocked until March 2026)

Resume CFBD API calls for recruit years 2011–2017 when quota resets.
Currently 0% recruiting coverage for the majority of the training window.

```bash
cd cfb-prospect-db
python scripts/populate_db.py --recruit-years 2011 2012 2013 2014 2015 2016 2017
```

### Priority 3 — Investigate dominator_rating < 0 (MEDIUM)

605 season rows have negative dominator_rating despite having non-null receiving yards.
Root cause likely in `populate_db.py` team denominator calculation. Investigate and fix;
confirm no affected rows appear in training set for WR/RB/TE.

### Priority 4 — 2026 Combine Ingestion (MEDIUM, time-sensitive)

The 2026 NFL combine occurred in February/March 2026. Ingest before any 2026 scoring:

```bash
cd cfb-prospect-db
python scripts/populate_nfl.py --combine-years 2026 --skip-draft --skip-rosters --skip-strength
```

### Priority 5 — Position Mismatch Review (LOW, partially expected)

Review `data/audit/crossdb_position_mismatches.csv` for the 103 position mismatches.
Some are confirmed wrong links (will be resolved by Priority 1). Others are legitimate:
- Braxton Miller: QB at Ohio State, converted to WR in NFL
- Antonio Gibson: WR at Memphis, played RB in NFL
- Players listed as "?" or "ATH" in cfb-prospect-db

Decide whether to clean up Player.position in cfb-prospect-db or to tolerate
known position-change cases in the model.

### Priority 6 — Consensus Big Board 2011–2015 (LOW, likely impossible free)

No free source has pre-2016 consensus boards. Accept the gap. Treat consensus_rank
as a 2016+ only feature, or source from archived ESPN / NFL.com draft board PDFs manually
for the top-50 players in each missing class.

---

## Re-Running the Audit

```bash
cd dynasty-prospect-model
python scripts/audit_data.py           # full audit, writes CSVs to data/audit/
python scripts/audit_data.py --no-csv  # print only
python scripts/audit_data.py --section cfb      # one section
python scripts/audit_data.py --section crossdb  # cross-DB only
```

The audit must be re-run and pass (or all FAILs documented and accepted) before
any model fitting begins.
