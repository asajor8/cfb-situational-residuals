# Situational success-rate residuals — CFB offenses, 2019 + 2021–2025

Data source: CollegeFootballData.com (CFBD) API — `/plays` (regular season, classification=fbs), `/teams/fbs`, `/ratings/sp`.
Generated 2026-11-06. All numbers below come from the files listed at the bottom; full tables in `RESULTS_tables.md` and `team_season_summary.csv`.

## 1. Download completeness
- Seasons: 2019, 2021, 2022, 2023, 2024, 2025 (2020 skipped on purpose; 2026 not pulled). Every week returned data until two empty weeks in a row. No week failed and no 429/5xx retries were needed.
- Weeks with plays: 2019 w1–15; 2021–2023 w1–15; 2024–2025 w1–16. The last week is small every year (conference-championship / Army–Navy week, e.g. 2024 w16 = 145 plays).
- Raw plays: 2019 152,647 · 2021 150,770 · 2022 151,023 · 2023 151,485 · 2024 154,296 · 2025 157,074 → **917,295 total**. No duplicate play ids.
- 2024 week 1 rebuilt from the cached file matches the earlier lesson: 12,787 offense snaps, 42.1% success.

## 2. Filter funnel
| Year | Raw plays | Offense snaps* | FBS-offense snaps | Modeling snaps |
|---|---|---|---|---|
| 2019 | 152,647 | 116,584 | 109,327 | 79,096 |
| 2021 | 150,770 | 114,082 | 107,027 | 77,338 |
| 2022 | 151,023 | 113,851 | 106,495 | 79,292 |
| 2023 | 151,485 | 113,204 | 106,126 | 75,504 |
| 2024 | 154,296 | 113,551 | 106,300 | 76,257 |
| 2025 | 157,074 | 115,284 | 107,981 | 76,316 |
| **All** | **917,295** | **686,556** | **643,256** | **463,803** |

*Offense snaps = the 13 scrimmage playTypes, minus kneels/spikes, with down in 1–4 (88 plays dropped for a bad down).
- The kneel regex dropped 0 (2019), 0 (2021), 383, 797, 882 and 927 snaps. The spike regex dropped 67, all in 2025.
- **Caveat:** 2019 and 2021 play text uses an older format with no "kneel" wording. Kneels show up as "TEAM run for a loss…" (1,058 such lines in 2019, 392 in 2021). Those kneels stay in as failed rushes, which pulls those two seasons down a little. The regex also misses the plural "kneels" (7 plays total).

Prior-year SP+ 25th-percentile cutoffs for "low-end" defenses (FBS teams only; the nationalAverages row removed):
2019 games ← 2018 SP+ < **−8.80** (130 teams) · 2021 ← 2020 SP+ < **−6.93** (130) · 2022 ← 2021 < **−7.18** (130) · 2023 ← 2022 < **−10.10** (131) · 2024 ← 2023 < **−9.40** (133) · 2025 ← 2024 < **−8.18** (134).
FBS newcomers with no prior-year SP+ count as low-end, as specified.

## 3. Schedule inflation is real, and it comes mostly from FCS games
FBS offenses, all six seasons:
| Opponent bucket | Snaps | Success | Model-expected | Residual | Mean PPA |
|---|---|---|---|---|---|
| FCS / non-FBS | 48,020 | **51.3%** | 46.2% | **+5.1 pts** | 0.334 |
| Low-end non-power FBS | 131,433 | 44.1% | 43.1% | +1.0 | 0.174 |
| Other FBS | 163,310 | 41.8% | 41.8% | 0.0 | 0.112 |
| Power 4/5 + Notre Dame | 300,493 | 41.7% | 41.7% | 0.0 | 0.104 |

The FCS bump shows up every year: FCS-game success runs 49.8–52.8% against 40.9–42.5% vs power defenses. Easier down-and-distance explains only about 4.5 pts of that ~9.6-pt gap (expected 46.2% vs 41.7%). The other **+5.1 pts is pure opponent weakness**, so those snaps stay out of the model. Week 1 2024 showed the same thing (51.7% vs FCS).

## 4. Expected-success model
- Logistic regression on the 463,803 modeling snaps: `success ~ C(down) + distance + yardsToGoal + score_diff`, where score_diff = offenseScore − defenseScore.
- Coefficients (log-odds): intercept 1.080; 2nd down −0.422; 3rd −0.638; 4th −0.419; distance **−0.155 per yard**; yardsToGoal +0.0033; score_diff +0.0084. All SEs ≤ 0.02. Pseudo-R² 0.045, Brier 0.2289. Overall success 41.7%, mean PPA 0.107.
- **Pooled and in-sample (descriptive, not predictive).** No team, opponent or conference terms; those belong in the residual. Opponent quality uses prior-year SP+ only.
- Calibration is exact by down (42.8 / 40.9 / 39.3 / 52.7%). By situation, actual vs predicted:
  - 1st & 10: 42.7 vs 42.6 (n=184,231)
  - 2nd & 7+: 32.7 vs 32.7
  - 3rd & 7+: 25.1 vs 26.6
  - 3rd & 4–6: 41.7 vs 45.4
  - 3rd & 1: **73.9 vs 60.3** (n=9,822)
  - 4th & 1: 69.8 vs 64.6
  - 1st & 11+: 27.4 vs 21.1
  - Red zone (1–10 yards to goal): 47.1 vs 52.0

  The straight-line distance term under-predicts short yardage.
- **Interaction check:** adding down × distance lowers AIC by 609 and weighted situation error from 1.29 to 0.87 pts. It narrows the 3rd & 1 miss only to 10.2 pts. A log-distance version (sensitivity only) gets the 3rd & 1 miss down to 2.7 pts.
- **Team-season residuals barely change across the three specs:** correlation ≥ 0.999, max difference 0.5 pts. So I **kept the additive model**. Short-yardage miscalibration is a known limitation: it slightly flatters heavy short-yardage offenses.
- **PPA check (not refit):** at the team-season level, residual vs mean PPA correlates at **r = 0.82** (0.80–0.85 each year). Raw success vs PPA is r = 0.91.

## 5. Team-season table — the rankings change
- Qualifiers need ≥300 modeling snaps. That leaves 124–131 teams per year and 765 team-seasons; it mainly drops transition-year schools and service academies that load up on cupcakes.
- Raw-success rank vs residual rank: Spearman 0.84–0.90 per year. The median team moves **10 spots**, and 51% move 10 or more. Each year, 6–9 of the raw top 25 fall out of the residual top 25.

**High raw success, but cupcakes or easy situations → ordinary residual**
| Team-season | Raw (rank) | Cupcake share | Non-cupcake success | Expected | Residual (rank) |
|---|---|---|---|---|---|
| 2024 Army | 50.9% (#5) | 50.3% | 46.5% | 45.4% | +1.1 (#42) |
| 2025 Navy | 49.9% (#10) | 57.6% | 43.7% | 42.3% | +1.4 (#43) |
| 2025 Ole Miss | 48.6% (#14) | 18.1% | 45.0% | 44.2% | +0.9 (#47) |
| 2023 Penn State | 47.1% (#19) | 17.6% | 43.6% | 44.5% | −0.9 (#65) |
| 2025 Texas Tech | 45.9% (#35) | 14.4% | 43.1% | 44.0% | −1.0 (#75) |

**Lower raw success, but a harder slate or harder situations → better residual**
| Team-season | Raw (rank) | Cupcake share | Non-cupcake success | Expected | Residual (rank) | Opp prior SP+ |
|---|---|---|---|---|---|---|
| 2024 App State | 44.0% (#58) | 26.8% | 44.7% | 39.3% | +5.4 (#8) | 0.2 |
| 2025 Arkansas | 48.4% (#15) | 16.8% | 46.6% | 40.5% | +6.1 (#8) | 16.2 |
| 2025 Georgia Tech | 49.9% (#9) | 10.4% | 50.2% | 43.1% | **+7.1 (#1)** | 6.3 |
| 2019 UCLA | 43.2% (#56) | 0.0% | 43.2% | 41.4% | +1.8 (#25) | 6.3 |

How to read it: raw success mixes three things — offense quality, cupcake opponents, and friendly situations (short distances, playing with a lead). Army 2024 lost 4.4 pts of success once half its snaps (the cupcake ones) came out. Arkansas 2025 succeeded *less* than Oregon (48.4% vs 51.3%), but it faced the toughest defenses among the top residual teams (mean prior-year SP+ 16.2) and harder downs (expected 40.5% vs 45.5%). Its residual (+6.1) beats Oregon's (+4.5).

## 6. Oregon case study
| Year | Raw (rank) | Cupcake share | Non-cupcake success | Expected | Residual (rank) | Mean PPA (rank) | n |
|---|---|---|---|---|---|---|---|
| 2019 | 46.1% (#22) | 9.0% | 45.3% | 44.0% | +1.3 (#37) | 0.258 (#7) | 823 |
| 2021 | 49.2% (#12) | 8.0% | 48.9% | 42.8% | +6.1 (#8) | 0.212 (#31) | 799 |
| 2022 | 53.5% (#3) | 9.7% | 52.6% | 44.5% | +8.1 (#3) | 0.274 (#8) | 789 |
| 2023 | 56.6% (#1) | 15.5% | 55.7% | 45.0% | +10.7 (#2) | 0.409 (#2) | 734 |
| 2024 | 51.6% (#3) | 9.8% | 52.0% | 45.6% | +6.4 (#3) | 0.339 (#1) | 781 |
| 2025 | 51.3% (#5) | 8.3% | 50.0% | 45.5% | +4.5 (#11 of 125) | 0.330 (#2) | 692 |

- **Cupcakes don't explain Oregon's season numbers.** Only 8–16% of its snaps came against FCS or low-end teams, and its non-cupcake success is within 1 pt of raw. The 48.2% against Idaho in week 1 of 2024 was a cupcake game; the full 2024 season is still strong (52.0% non-cupcake, residual +6.4, #3; PPA #1).
- **Situation is the bigger factor.** Oregon's expected success (44.0–45.6%) sits among the highest in FBS because it plays from ahead and in manageable distances. In 2025 its raw edge over the average qualifier (43.2%) was +8.0 pts; after the situation adjustment the residual is +4.5, and its rank slips from #5 to #11.
- **The "lost to a lower-raw, tougher-schedule team" question, 2023 Washington:**
  - Washington's raw success was 53.0% vs Oregon's 56.6%, with **0%** cupcake snaps (Oregon 15.5%).
  - Residuals narrow the gap from 3.7 pts to 2.1 (+8.6 vs +10.7).
  - In both head-to-heads, Washington's offense out-succeeded Oregon's: 50.8% vs 50.0% in week 7, and 53.8% vs 46.3% in week 14.
- **2025 Indiana:** Indiana had *higher* raw success (54.4%) and a higher residual (+5.8, #9). In their week-7 game, Oregon's offense hit just 36.1% (residual −6.2) while Indiana's hit 48.4% (+5.5).
- **What this means:** a season-level offensive number, even adjusted, doesn't predict one game. It also ignores defense — 2022 Georgia had a lower offensive residual than Oregon (+5.7 vs +8.1) and still dominated the game (Georgia offense 67.7% success vs Oregon's 48.5%).

## 7. 2025 residual leaders and laggards (≥300 modeling snaps, 125 qualifiers)
- **Top:** Georgia Tech +7.1 (raw #9) · North Texas +7.1 (raw #4; 44.7% cupcake share, so a softer slate) · Vanderbilt +6.8 (raw #3) · Ohio State +6.8 (raw #1) · Tennessee +6.5 · USC +6.5. **Oregon +4.5, #11.**
- **Bottom:** Ball State −6.5 · Kent State −6.5 · Colorado −6.3 · Akron −6.3 (raw #101 but 62.5% cupcake share; non-cupcake success 32.0%) · Oklahoma State −6.2.

## 8. Caveats
- **In-sample and descriptive.** One pooled model across six seasons.
- **No opponent strength inside the modeling sample.** The residual adjusts for situation; opponents are handled only by the cupcake filter. Power and other-FBS defenses allowed the same average success (41.7% vs 41.8%), but a fuller version should add an opponent-defense adjustment.
- **Situation terms partly reflect offense quality.** Distance and score_diff partly come from how good the offense is (good offenses create short yardage and leads), so residuals are conservative for elite offenses.
- **Score timing.** CFBD offense/defense scores are taken as pre-snap; on scoring plays they may already include the score.
- **Kneels in 2019/2021 are under-detected** (see §2). Short-yardage calibration is weak (see §4).
- **Special teams excluded on purpose** (planned later slice). Defense is covered in `RESULTS_defense.md` (same snaps, mirrored).
- **PPA is CFBD's own metric,** used only as an independent check. 1,398 snaps (0.2%) have no PPA.

## Files
- Code: `src/download.py` (cached, retry/backoff; reads `CFBD_API_KEY` from the environment and never writes it), `src/clean.py`, `src/model.py`, `src/sensitivity.py`, `src/report.py`, `src/run_all.sh`
- Raw (not in this repo; download with your own API key): `data/raw/plays_{year}_w{week}.json`, `teams_fbs_{2018,2019,2021–2025}.json`, `sp_{2018–2024}.json`, `download_log.json`
- Clean intermediates (rebuilt by the pipeline; parquets not committed): `data/processed/offense_snaps.parquet` (686,556 snaps with flags) and `data/processed/modeling_snaps.parquet` (adds expected_success and residual; the model is also applied to cupcake snaps so bucket comparisons work)
- `data/derived/team_season_summary.csv` (all 794 team-seasons; `qualified` = ≥300 modeling snaps)
- `data/processed/` also holds `funnel.csv`, `sp_cutoffs.json`, `model_summary.json`, the `cal_*.csv` calibration tables, and `opp_bucket*.csv` (included). `model_stdout.txt` and `sensitivity_stdout.txt` are regenerated by the pipeline and are not committed.
- `docs/RESULTS_tables.md` (auto-generated tables) and `figures/raw_vs_residual_2025.png`
