# College football situational success residuals

**Question.** After controlling for down, distance, field position, and score, which FBS offenses (and defenses) systematically over- or underperform classic success rate, and does that stick year to year?

**Answer in one line.** Yes on both sides, with clear schedule inflation from FCS/cupcake snaps. Residuals persist moderately (offense r≈0.39, defense r≈0.32) but less than raw success or CFBD PPA, so pair the residual with PPA rather than replacing it.

## Data and definition
- Source: CollegeFootballData play-by-play, seasons 2019 + 2021–2025 (2020 skipped). 917,295 raw plays → 463,803 offense modeling snaps after filters.
- Success: classic 50/70/100. Special teams (punts, kickoffs, field goals) excluded. Kneels/spikes dropped when text allows.
- Cupcake filter: drop snaps vs FCS and vs low-end non-P4 (prior-year SP+ below that year's 25th percentile among FBS). Same filter flipped for defense (drop snaps vs weak offenses).
- Model: pooled logistic `success ~ C(down) + distance + yardsToGoal + score_diff` on non-cupcake snaps. Residual = actual − expected. PPA is CFBD's own metric, used only as an independent check (not refit).

## Offense findings
- FCS snaps run ~51% success vs ~42% vs power defenses; about +5.1 pts of that gap is pure opponent weakness after situation control. Those snaps stay out of the model.
- Team-season residuals reshuffle rankings: median team moves ~10 spots vs raw success; 6–9 of each year's raw top 25 fall out of the residual top 25.
- Examples: 2024 Army raw #5 → residual #42 (50% cupcake share). 2025 Georgia Tech raw #9 → residual #1 (+7.1). 2025 Arkansas raw #15 → residual #8 on a hard slate.
- Residual vs mean PPA: r = 0.82 at team-season level.

## Oregon (offense case study)
| Year | Raw rank | Residual | Residual rank | Mean PPA rank |
|------|----------|----------|---------------|---------------|
| 2019 | #22 | +1.3 | #37 | #7 |
| 2021 | #12 | +6.1 | #8 | #31 |
| 2022 | #3 | +8.1 | #3 | #8 |
| 2023 | #1 | +10.7 | #2 | #2 |
| 2024 | #3 | +6.4 | #3 | #1 |
| 2025 | #5 | +4.5 | #11 | #2 |

Cupcakes do not explain Oregon's numbers (8–16% of snaps). Friendly situations do: expected success sits ~44–46%, so the residual is smaller than the raw edge. Season-level efficiency still fails single games (2023 Washington head-to-heads; 2025 Indiana).

## Defense add-on
- Defensive PPA = −offense PPA; defensive residual = expected − success (reuses the offense model; no new fit).
- Weak offenses inflate defensive numbers the same way (+5.1 pts vs FCS). Filter applies.
- 2025 Oregon defense: residual +3.6 (#10), def PPA +0.040 (#8). 2021–2022 defense was a weakness; 2025 is a strength.
- Net view matters: 2022 Georgia beat Oregon despite a lower offensive residual because Georgia's defense was elite (net PPA #1 vs Oregon #13).

## Persistence
Adjacent-year Pearson (offense): residual 0.39, raw success 0.47, mean PPA 0.46. Defense: residual 0.32, def PPA 0.44.
Top-quintile residual retention falls from ~48% (2021→2022) to ~32% (2024→2025). Single-season #1s often crash (UCLA, LSU, Western Kentucky).
Oregon path: +1.3 → +6.1 → +8.1 → +10.7 → +6.4 → +4.5 (elite stretch, then soft landing).

## Caveats
- In-sample and descriptive. No opponent-strength term inside the modeling sample beyond the cupcake filter.
- Short-yardage (esp. 3rd & 1) is under-predicted; additive model kept after sensitivity checks (team-season residuals barely move).
- Kneels under-detected in 2019/2021 text. Score fields on scoring plays may already include the points.
- Special teams left for a later slice.

## Source artifacts
`docs/RESULTS.md`, `docs/RESULTS_defense.md`, `docs/RESULTS_persistence.md`, `data/derived/team_season_*.csv`, `figures/`, and `src/`.
