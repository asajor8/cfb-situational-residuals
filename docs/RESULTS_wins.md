# Do efficiency residuals and PPA predict wins? — Stage 1

Generated 2026-11-10 from CFBD `/games` (regular + postseason, 2019–2025) joined to `team_season_combined.csv`. Data source: CollegeFootballData.com (CFBD). Script: `src/wins.py`.

## Punchline

Same-season, both net PPA and the net situational residual track point margin and win% closely — as they should, because efficient teams score and prevent points now. The stricter test is **next season**. This season's W-L already predicts next season's win% (in-sample R² 0.215; leave-one-season-out 0.211). Adding **net PPA** raises that to 0.241 (ΔR² +0.025; LOSO 0.235). Adding **net residual** instead raises it to 0.243 (ΔR² +0.028; LOSO 0.239). Both increments hold out of sample, so efficiency adds signal beyond the win-loss record.

A one-SD bump in this season's net PPA (0.182), holding win% fixed, is associated with +6.7 percentage points of next-season win% (SE 1.5). The same for net residual (SD 0.045) is +5.1 (SE 1.1).

The residual's increment is as large as — slightly larger than — net PPA's, even though the residual is less sticky year to year and less tightly tied to this year's record (same-season r with win% is 0.65 vs 0.84 for PPA). That is the point of the "beyond W-L" test: more of PPA already shows up in this year's wins, so less unique signal is left once you condition on the record.

## 1. Data and sample

- **Games:** CFBD `/games?classification=fbs` for 2019 regular 848 / post 40, 2020 regular 542 / post 26, 2021 regular 849 / post 38, 2022 regular 854 / post 42, 2023 regular 868 / post 42, 2024 regular 874 / post 46, 2025 regular 888 / post 46. Cached as `data/raw/games_{year}_{regular|postseason}.json`. Only completed games with both scores present. 11,254 team-game rows among FBS teams; **10,504 are FBS-vs-FBS**.
- **Record and margin definition:** FBS-vs-FBS only, regular season plus postseason (bowls and the CFP). This is the practical analog of the project's cupcake filter, which drops FCS and low-end snaps from the *efficiency* model. Records here go one step coarser: they drop FCS games but still count games against low-end FBS teams, because those games are real FBS results. A team's official all-opponent record can therefore look a little different (FCS wins are excluded).
- **2020** is flagged as COVID-shortened. The original project never pulled 2020 plays, so there are **no 2020 residuals or PPA**. 2020 appears only as a next-season win% outcome in the 2019→2020 pairs. Many 2020 teams played conference-only, shorter schedules.
- **Efficiency** comes from the existing qualifier file: offense residual and mean PPA on non-cupcake snaps; defensive residual and defensive PPA on the defensive mirror (higher is better on both sides). **Net residual** = offense residual + defense residual. **Net PPA** = offense mean PPA + defensive PPA.
- **Analysis sample:** team-seasons that are qualified on both sides (≥300 modeling snaps) and have at least 6 completed FBS-vs-FBS games. That is **760 team-seasons** in 2019 and 2021–2025 (2020 has no efficiency to join).
- **Luck** (stored for later stages) = actual FBS-vs-FBS win% minus Pythagorean expected win% from points scored and allowed, with exponent 2.37 (the usual football exponent, not re-fit here).

## 2. Same-season associations

Simple Pearson correlations and OLS of the outcome on one efficiency metric. Win% coefficients are in percentage points per unit of the predictor (so net PPA is "pp of win% per 1.0 of net PPA"); margin coefficients are points of average margin per unit of the predictor.

| Outcome | Predictor | n | r | R² | coef | SE |
|---|---|---|---|---|---|---|
| Point margin | Net PPA | 760 | 0.92 | 0.840 | 54.59 | 0.87 |
| Point margin | Net residual | 760 | 0.75 | 0.559 | 179.35 | 5.79 |
| Win% | Net PPA | 760 | 0.84 | 0.698 | 103.5 | 2.5 |
| Win% | Net residual | 760 | 0.65 | 0.428 | 326.3 | 13.7 |

Same-season r(net PPA, point margin) = **0.92**; r(net residual, point margin) = **0.75**. r(net PPA, win%) = **0.84**; r(net residual, win%) = **0.65**. Net PPA lines up a bit more tightly with scoring margin and record than the success residual does, which matches the earlier finding that PPA already embeds more of the scoring outcome.

Raw coefficients are per a 1.0 change in the predictor, which is a huge move (net residual SD is only 0.045). In more natural units, a one-SD higher net PPA goes with about +9.9 points of average margin and +18.9 points of win%; a one-SD higher net residual goes with about +8.1 points of margin and +14.7 points of win%.

These are contemporaneous associations, not a claim that the residual "causes" this year's wins.

## 3. Next-season win%, beyond this year's W-L

Each row is a team with a qualified efficiency season at t and at least 6 FBS-vs-FBS games at t+1. Adjacent pairs are 2019→2020, 2021→2022, 2022→2023, 2023→2024, 2024→2025. There is no 2020→2021 pair (no 2020 efficiency), and 2019→2021 is a two-year COVID gap, so it is not treated as "next season."

Models:
1. `win%_{t+1} ~ win%_t`
2. `win%_{t+1} ~ win%_t + net PPA_t`
3. `win%_{t+1} ~ win%_t + net residual_t`

Coefficients below are for a 0–1 win% outcome (the win% coefficient is the usual persistence slope; the efficiency coefficient is the change in next-season win% proportion per unit of net PPA or net residual). Leave-one-season-out (LOSO) drops one *this-season* year at a time and scores predictions against that fold's training-mean null.

### Main sample (includes 2019→2020)

| Model | n | In-sample R² | ΔR² vs W-L | LOSO R² | coef win% (SE) | coef efficiency (SE) |
|---|---|---|---|---|---|---|
| This-season win% only | 618 | 0.215 | — | 0.211 | 0.472 (0.036) | — |
| + net PPA | 618 | 0.241 | +0.025 | 0.235 | 0.224 (0.065) | 0.369 (0.081) |
| + net residual | 618 | 0.243 | +0.028 | 0.239 | 0.324 (0.047) | 1.137 (0.238) |

LOSO R² by this-season year:

| This season | n | W-L only | + net PPA | + net residual |
|---|---|---|---|---|
| 2019 | 111 | 0.216 | 0.244 | 0.247 |
| 2021 | 128 | 0.212 | 0.255 | 0.256 |
| 2022 | 127 | 0.283 | 0.315 | 0.308 |
| 2023 | 123 | 0.091 | 0.098 | 0.120 |
| 2024 | 129 | 0.242 | 0.253 | 0.253 |

### Sensitivity: drop pairs involving 2020

n = 507 (2019→2020 removed). In-sample R²: W-L only 0.215, + net PPA 0.239 (Δ +0.025), + net residual 0.242 (Δ +0.027). LOSO R²: W-L 0.209, + PPA 0.231, + residual 0.235.

## 4. Oregon, season by season

FBS-vs-FBS record and margin, plus the efficiency totals already in the portfolio writeup. Luck is actual win% minus Pythagorean expectation.

| Year | FBS-vs-FBS | Win% | Avg margin | Off resid (pts) | Def resid (pts) | Net resid | Net PPA | Luck | Next-year win% |
|---|---|---|---|---|---|---|---|---|---|
| 2019 | 11-2 | 84.6% | +17.8 | +1.3 | +0.6 | +1.9 | +0.297 | +0.5% | 57.1% |
| 2021 | 9-4 | 69.2% | +1.6 | +6.1 | -3.1 | +3.0 | +0.091 | +16.0% | 75.0% |
| 2022 | 9-3 | 75.0% | +7.7 | +8.1 | -5.4 | +2.7 | +0.162 | +11.2% | 84.6% |
| 2023 | 11-2 | 84.6% | +24.2 | +10.7 | -0.0 | +10.7 | +0.391 | -4.2% | 92.3% |
| 2024 | 12-1 | 92.3% | +15.8 | +6.4 | +0.5 | +6.9 | +0.309 | +12.2% | 85.7% |
| 2025 | 12-2 | 85.7% | +17.1 | +4.5 | +3.6 | +8.1 | +0.369 | +2.9% | — |

- **2019:** Oregon went 11-2 in FBS-vs-FBS games (84.6%, average margin +17.8). Offense residual +1.3 pts, defense residual +0.6 pts (net +1.9), net PPA +0.297. Actual win% was right on Pythagorean luck (+0.5%). The next season they won 57.1% of FBS-vs-FBS games (4-3).
- **2021:** Oregon went 9-4 in FBS-vs-FBS games (69.2%, average margin +1.6). Offense residual +6.1 pts, defense residual -3.1 pts (net +3.0), net PPA +0.091. Actual win% was ahead of Pythagorean luck (+16.0%). The next season they won 75.0% of FBS-vs-FBS games (9-3).
- **2022:** Oregon went 9-3 in FBS-vs-FBS games (75.0%, average margin +7.7). Offense residual +8.1 pts, defense residual -5.4 pts (net +2.7), net PPA +0.162. Actual win% was ahead of Pythagorean luck (+11.2%). The next season they won 84.6% of FBS-vs-FBS games (11-2).
- **2023:** Oregon went 11-2 in FBS-vs-FBS games (84.6%, average margin +24.2). Offense residual +10.7 pts, defense residual -0.0 pts (net +10.7), net PPA +0.391. Actual win% was behind Pythagorean luck (-4.2%). The next season they won 92.3% of FBS-vs-FBS games (12-1).
- **2024:** Oregon went 12-1 in FBS-vs-FBS games (92.3%, average margin +15.8). Offense residual +6.4 pts, defense residual +0.5 pts (net +6.9), net PPA +0.309. Actual win% was ahead of Pythagorean luck (+12.2%). The next season they won 85.7% of FBS-vs-FBS games (12-2).
- **2025:** Oregon went 12-2 in FBS-vs-FBS games (85.7%, average margin +17.1). Offense residual +4.5 pts, defense residual +3.6 pts (net +8.1), net PPA +0.369. Actual win% was ahead of Pythagorean luck (+2.9%). There is no next season in this window.

The 2021 team is the luck example: 9-4 with only a +1.6 average margin (luck +16 points) and a modest net PPA (+0.091), because the defense was well below average. The 2023 team is the efficiency peak — net residual +10.7, net PPA +0.391, average margin +24.2 — and they were a little *unlucky* in close games; the next year they went 12-1. 2025 is the first season in this window where the defense residual is clearly positive too (+3.6).

## 5. Caveats

- 2020 is a short, conference-heavy COVID season with no efficiency metrics. The main next-season sample includes 2019→2020 as an outcome year; the sensitivity drops those pairs. Neither sample has a 2020→2021 efficiency-to-wins pair.
- Same-season correlations are descriptive. Efficient offenses and defenses produce points and wins in the same year; that is not the "beyond W-L" test.
- FBS-vs-FBS records ignore FCS games, so they will not always match a team's official record or the one on a Wikipedia page.
- Qualifying on ≥300 non-cupcake snaps drops some transition / cupcake-heavy programs, consistent with the earlier reports.
- The residual and PPA are correlated (offense residual vs mean PPA was r = 0.82). This stage does not put both in one next-season model; a horse race belongs in a later stage if we want it.
- Pythagorean luck uses a fixed 2.37 exponent, not one estimated on this CFB sample. It is stored for the luck-regression stage, not used as a predictor here.
- No adjustment for opponent strength on the win/margin side beyond dropping FCS, and no terms for coaching changes or roster turnover.
- Postseason games are in the record, so a deep playoff run lifts both win% and margin.
- LOSO is a start on out-of-sample checking; it is not a full walk-forward forecasting exercise.

## 6. Options for stage 2

1. **Point-margin target.** Repeat the next-season regressions with average point margin (and maybe raw wins) as the outcome: does this year's efficiency beat this year's margin for predicting next year's margin?
2. **Luck regression.** Do teams that won more games than Pythagorean (or a margin logistic) expected fall back next year? That was option 2 in the original inference list, and luck is already in the team-season file.
3. **Formal inference on the increment.** Confidence intervals or a bootstrap / nested F-test on the ΔR², and a joint model that asks whether the residual still adds anything once net PPA is already in the regression (or the reverse).

## Files

- `src/wins.py`
- `data/derived/team_season_wins.csv` — team-season record, margin, luck, and efficiency
- `data/processed/wins_next_pairs.csv` — adjacent-season pairs used in section 3
- `data/processed/wins_models.json` — correlations, OLS, and LOSO in machine-readable form
- `figures/wins_same_season_margin.png`, `figures/wins_next_season_r2.png`
- Raw `/games` JSON is not shipped (rebuild with `make wins` and a CFBD key)
