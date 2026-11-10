# Defense add-on — defensive PPA and success residual, 2019 + 2021–2025

Data source: CollegeFootballData.com (CFBD) play-by-play. These are the same kept scrimmage play types as the offense work (no special teams), and no plays were re-downloaded. Generated 2026-11-06. Full tables are in `RESULTS_defense_tables.md`, `team_season_defense.csv` and `team_season_combined.csv`.

## Method (no new model)
- **Defensive PPA = −CFBD `ppa`** on each snap (offense-perspective Predicted Points Added, which CFBD already adjusts for situation). Positive means the defense prevented points versus expectation. 767 defensive-sample snaps with null PPA stay null.
- **Defensive success** = 1 − success.
- **Defensive residual** = expected_success − success. It reuses the existing pooled additive logistic model (`success ~ C(down) + distance + yardsToGoal + score_diff`), with nothing refit. That makes it the exact negative of the offense residual on each snap.
- **PPA is the primary stat;** the success residual is the companion check.
- **Pooled, in-sample, descriptive.** Opponent quality uses prior-year SP+ only.
- **Defensive PPA is usually negative.** Offenses average positive PPA on scrimmage snaps; the 2025 qualifier mean is −0.143. Read ranks, or distance from that mean.
- **Cupcake filter, flipped:** a defense is rated on a snap only if both teams are FBS that season and the *offense* is not low-end non-power. Low-end uses the same rule as before: not in a power conference or Notre Dame, and either no prior-year SP+ or prior-year SP+ below that year's 25th percentile (cutoffs −8.80, −6.93, −7.18, −10.10, −9.40, −8.18). New flags written to `modeling_snaps.parquet`: `is_power_offense`, `off_prior_sp`, `is_low_end_offense`, `use_in_def_model`, `off_bucket`, `defensive_ppa`, `defensive_success`, `defensive_residual`.

## Sample sizes
| | FBS-side snaps | Modeling snaps |
|---|---|---|
| Offense (FBS offense, non-cupcake defense) | 643,256 | 463,803 |
| Defense (FBS defense, non-cupcake offense) | 638,536 | **466,120** |

Qualifiers need ≥300 defensive modeling snaps: 129 (2019), 128 (2021), 127 (2022), 124 (2023), 130 (2024) and 129 (2025) teams.

## Weak offenses inflate defensive numbers
FBS defenses, all years, grouped by the offense they faced:
| Offense faced | Snaps | Success allowed | Expected | Def residual | Mean def PPA |
|---|---|---|---|---|---|
| FCS / non-FBS | 43,300 | **32.9%** | 38.0% | **+5.1 pts** | **+0.076** |
| Low-end non-power FBS | 129,116 | 39.5% | 40.8% | +1.4 | −0.059 |
| Other FBS | 163,828 | 42.6% | 42.4% | −0.3 | −0.135 |
| Power + ND | 302,292 | 43.3% | 42.4% | −0.9 | −0.142 |

- Against FCS offenses, success allowed was 31.1–34.3% every year, versus 42.4–43.7% against power offenses.
- Defensive PPA ran +0.024 to +0.109 against FCS offenses versus −0.126 to −0.161 against power offenses.
- This mirrors the offense finding (+5.1 pts on both sides).

## Checks
- Team-season defensive residual vs defensive PPA: r = **0.76** (0.73–0.80 by year), a bit looser than on offense (0.82).
- **CFBD sanity check:** the mean PPA allowed matches CFBD's `/stats/season/advanced` defense PPA at r = **0.988** (767 qualifier team-seasons, 0.984–0.992 by year). That check is on all FBS-defense snaps, not the filtered set.
- Raw-to-filtered rank stability:
  - Defensive PPA: Spearman 0.90–0.93.
  - Raw success allowed vs residual: Spearman 0.85–0.88. The median defense moves 10 spots, and 53% move 10 or more.

## Ranking changes
**Looked great raw because of weak offenses**
| Team-season | Raw SR allowed (rank) | Cupcake-offense share | Non-cupcake SR allowed (rank) | Def residual (rank) | Def PPA raw → filtered (rank) |
|---|---|---|---|---|---|
| 2024 Louisiana Tech | 35.8% (#10) | 55.7% | 42.6% (#62) | −0.1 (#56) | −0.012 (#21) → −0.147 (#63) |
| 2022 Toledo | 35.2% (#8) | 51.5% | 39.9% (#34) | +1.1 (#40) | −0.027 (#25) → −0.117 (#50) |
| 2024 Tulane | 36.0% (#13) | 40.4% | 40.3% (#35) | +1.6 (#31) | +0.034 (#11) → −0.178 (#85) |
| 2019 UAB | 33.4% (#4) | 59.3% | 37.1% (#18) | +6.5 (#3) | +0.065 (#5) → −0.057 (#38) |

**Graded better after the filter or the residual**
| Team-season | Raw SR allowed (rank) | Cupcake-offense share | Non-cupcake SR allowed (rank) | Def residual (rank) | Def PPA raw → filtered (rank) |
|---|---|---|---|---|---|
| 2019 Air Force | 39.9% (#46) | 55.6% | 34.4% (#4) | +6.6 (#2) | −0.047 (#42) → −0.063 (#39) |
| 2023 Oklahoma State | 40.2% (#42) | 7.2% | 39.6% (#29) | +3.2 (#15) | −0.143 (#76) → −0.130 (#59) |
| 2024 Colorado | 38.0% (#28) | 8.4% | 37.1% (#10) | +3.2 (#10) | −0.004 (#18) → +0.027 (#7) |
| 2025 Buffalo | 38.8% (#32) | 47.1% | 40.1% (#31) | +2.3 (#19) | −0.064 (#48) → −0.017 (#18) |

## Oregon defense by year (qualifier ranks)
| Year | Raw SR allowed (rank) | Non-cupcake SR allowed (rank) | Expected faced | Def residual (rank) | Def PPA (rank) |
|---|---|---|---|---|---|
| 2019 | 37.2% (#26) | 38.2% (#27) | 38.8% | +0.6 (#51) | +0.039 (#5 of 129) |
| 2021 | 44.1% (#89) | 44.6% (#79) | 41.6% | −3.1 (#94) | −0.120 (#43 of 128) |
| 2022 | 45.6% (#105) | 46.6% (#102) | 41.2% | −5.4 (#118) | −0.111 (#46 of 127) |
| 2023 | 38.0% (#25) | 39.7% (#30) | 39.7% | −0.0 (#50) | −0.018 (#16 of 124) |
| 2024 | 37.7% (#27) | 38.2% (#17) | 38.7% | +0.5 (#51) | −0.030 (#17 of 130) |
| 2025 | 36.7% (#15) | 36.2% (#10) | 39.8% | +3.6 (#10) | +0.040 (#8 of 129) |

Only 6–14% of the snaps Oregon's defense faced came from cupcake offenses, so the filter barely moves it. The defense was a weakness in 2021–2022 and climbed to top 10 by both measures in 2025.

## Both sides of the earlier matchups
Simple net = offense mean PPA + defensive PPA, and offense residual + defensive residual, per snap; ranks are among teams that qualify on both sides.

**2022 — Georgia's defense was the missing piece**
| 2022 | Off residual (rank) | Off PPA (rank) | Def residual (rank) | Def PPA (rank) | Net PPA (rank) |
|---|---|---|---|---|---|
| Oregon | +8.1 (#3) | 0.274 (#8) | −5.4 (#118) | −0.111 (#46) | 0.162 (#13) |
| Georgia | +5.7 (#9) | 0.304 (#5) | +3.7 (#11) | +0.093 (#2) | **0.397 (#1)** |

- Oregon's offense out-graded Georgia's on success residual, but Georgia's defense ranked #2 in defensive PPA versus Oregon's #46.
- In the week-1 game:
  - Oregon's offense: 48.5% success (expected 40.1%), but **−0.073 PPA/snap**, −4.7 total over 68 snaps.
  - Georgia's offense: 67.7% success, **+0.772 PPA/snap**, +40.9 total over 62 snaps.

**2023 — Washington vs Oregon: defense does *not* explain the losses**
| 2023 | Off residual (rank) | Off PPA (rank) | Def residual (rank) | Def PPA (rank) | Net PPA (rank) |
|---|---|---|---|---|---|
| Oregon | +10.7 (#2) | 0.409 (#2) | −0.0 (#50) | −0.018 (#16) | **0.391 (#2)** |
| Washington | +8.6 (#3) | 0.343 (#3) | −3.6 (#101) | −0.096 (#41) | 0.248 (#10) |

- On season numbers, Oregon was better on both sides of the ball.
- In both games Washington's offense piled up more total PPA:
  - Week 7: 19.4 vs 15.2 (per snap 0.319 vs 0.181).
  - Week 14: 19.2 vs 15.9 (per snap 0.246 vs 0.295, but 78 snaps vs 54).
- Takeaway: those losses were single-game outcomes, which season-level efficiency doesn't capture, and a schedule story doesn't explain them.

## 2025 leaders (129 qualifiers)
- **Defensive PPA, top 8:** Texas Tech +0.211 · San Diego State +0.105 · James Madison +0.091 · Ohio State +0.080 · Toledo +0.078 · Notre Dame +0.059 · Miami +0.054 · **Oregon +0.040 (#8)**.
- **Defensive PPA, bottom 8:** Air Force −0.424 · UAB −0.389 · Rutgers −0.363 · South Alabama −0.336 · Florida Atlantic −0.336 · Arkansas −0.327 · Delaware −0.327 · Northern Illinois −0.325.
- **Defensive residual, top 8:** James Madison +10.1 · Toledo +6.7 · Texas Tech +6.5 · Oklahoma +6.5 · Missouri +5.4 · Louisville +4.6 · Ohio State +4.4 · Indiana +4.1. Oregon is +3.6 (#10).
- **Defensive residual, bottom 8:** UAB −8.6 · Air Force −8.2 · Georgia Southern −7.9 · Louisiana −6.8 · Arkansas −6.5 · Tennessee −6.0 · Rutgers −5.7 · Jacksonville State −5.7.
- **Where the two measures disagree:**
  - Oklahoma: residual #4 but PPA #13.
  - Missouri: residual #5 but PPA #22.
  - Tennessee: residual #124 but PPA #70.

  The residual counts stops; PPA also weights explosive plays and turnovers.
- 2025 Arkansas: its offense residual was #8, but its defense ranked #124 in PPA and #125 in residual.

## Caveats
- In-sample and descriptive. The success model was fit on the offense sample and reused as-is, so the mean defensive residual among qualifiers isn't exactly zero (2025: −0.8 pts).
- No opponent-strength term inside the sample beyond the cupcake filter.
- PPA is CFBD's metric, used as given.
- Kneels are under-detected in 2019/2021 text; they count as defensive "successes."
- Scores on scoring plays may already include the points (CFBD convention).
- Small-snap teams such as service academies and transition schools sit near the 300-snap floor.
- Net figures are simple per-snap sums. They aren't weighted by plays per game, and special teams are excluded.

## Files
- Code: `src/defense.py` (flags, team-season defense table, combined join, CFBD check), `src/defense_report.py` (tables and chart)
- Data: `data/derived/team_season_defense.csv`, `data/derived/team_season_combined.csv` (offense + defense columns; defense columns prefixed `d_`), `data/processed/modeling_snaps.parquet` (updated with defensive columns; not committed)
- Summaries: `data/processed/defense_summary.json`, `def_opp_bucket.csv`, `def_ppa_by_offbucket_year.csv`, `def_sr_allowed_by_offbucket_year.csv`; cached `data/raw/adv_season_{year}.json` (raw cache not committed)
- `docs/RESULTS_defense_tables.md`, `figures/defense_ppa_vs_residual_2025.png`
