# Persistence — do situational residuals stick year to year?

Generated 2026-11-06 from `team_season_summary.csv` and `team_season_combined.csv`. Qualifiers need ≥300 modeling snaps in both seasons of a pair. Adjacent pairs are 2021→2022, 2022→2023, 2023→2024, 2024→2025 (n=488 team-pairs pooled). 2019→2021 is a two-year gap (COVID-skipped 2020) and is reported separately, not in the pooled figure.

## Punchline
Offense situational residuals **persist a little, but less than raw success or CFBD PPA**. Pooled adjacent Pearson correlations:
- residual **0.39** (Spearman 0.36)
- raw success **0.47** (0.46)
- mean PPA **0.46** (0.45)

So adjusting for down/distance/score removes some of the sticky part of the signal (personnel and scheme that also show up as friendly situations). Residuals are still predictive of next year (top-quartile teams stay above average), just not as sticky as the unadjusted numbers.

Defense: residual **0.32**, defensive PPA **0.44**. Same pattern — PPA sticks more than the success residual.

## Offense residual by year pair
| Pair | n | Pearson | Spearman | Top-quartile next residual | Bottom-quartile next residual |
|---|---|---|---|---|---|
| 2019→2021 (gap) | 127 | 0.30 | 0.29 | +0.020 | −0.014 |
| 2021→2022 | 125 | 0.46 | 0.44 | +0.024 | −0.026 |
| 2022→2023 | 119 | 0.45 | 0.37 | +0.021 | −0.021 |
| 2023→2024 | 122 | 0.32 | 0.34 | +0.011 | −0.014 |
| 2024→2025 | 122 | 0.31 | 0.26 | +0.013 | −0.010 |

Top-quintile retention (chance a top-20% residual team is still top-20% next year; random ≈20%):
2021→2022 48% · 2022→2023 46% · 2023→2024 38% · 2024→2025 32%.

## Defense by year pair
| Pair | n | Residual r | Def PPA r |
|---|---|---|---|
| 2019→2021 | 127 | 0.28 | 0.47 |
| 2021→2022 | 124 | 0.31 | 0.50 |
| 2022→2023 | 118 | 0.38 | 0.49 |
| 2023→2024 | 122 | 0.31 | 0.42 |
| 2024→2025 | 124 | 0.31 | 0.39 |

## Oregon (offense residual path)
2019 +1.3 (#37) → 2021 +6.1 (#8) → 2022 +8.1 (#3) → 2023 +10.7 (#2) → 2024 +6.4 (#3) → 2025 +4.5 (#11).
A multi-year elite stretch, then a soft landing — consistent with moderate persistence, not lock-in.

## Big flips (adjacent years)
Largest residual drops: Virginia 2021→2022 (−11.6 pts, #2→#106), UCLA 2022→2023 (−11.6, #1→#69), Western Kentucky 2021→2022 (−10.7, #1→#42), LSU 2023→2024 (−9.8, #1→#38), Oklahoma 2023→2024 (−9.6, #7→#120).
Largest rises: Massachusetts 2022→2023 (+10.8), Arkansas 2023→2024 (+10.2), Utah 2024→2025 (+9.6), Miami 2023→2024 (+9.6, #89→#1), Washington 2021→2022 (+9.5, #78→#2).

## What this means for the portfolio question
1. **Yes, over/underperformance partly persists** after situation control — enough that a one-year residual is useful prior for next year, especially the top quartile.
2. **It does not persist as strongly as raw success or PPA**, so the residual is not a magic sticky talent measure by itself. Pair it with PPA (already-adjusted check) rather than replacing it.
3. **Single-season leaders are fragile.** Several #1 residual offenses fell into the middle or bottom the next year (UCLA, LSU, Western Kentucky).

## Files
- `data/processed/persist_off_residual.csv`, `persist_off_raw.csv`, `persist_off_ppa.csv`
- `data/processed/persist_def_residual.csv`, `persist_def_ppa.csv`
