# Does Pythagorean luck regress next season? — Stage 2

Generated 2026-11-10 from the stage-1 files `data/processed/wins_next_pairs.csv` and `team_season_wins.csv`. Luck is actual FBS-vs-FBS win% minus Pythagorean expected win% with exponent 2.37. Data source: CollegeFootballData.com (CFBD). Script: `src/luck.py`.

## Punchline

Luck barely sticks, and the wins mostly come back. Year-to-year luck correlation is **r = 0.08** (Spearman 0.08), close to the "it's noise" expectation of zero. A team that outperformed Pythagorean this year gives almost all of that surplus back the next year: regressing the change in win% on luck yields a coefficient of **-0.98** (SE 0.10, R² 0.145). That is about one game given back next season for each lucky game this season, in 12-game units.

The top luck decile dropped **16.6 percentage points** of win% the next year (about **-2.0 games** per 12-game season), from 66.7% to 50.0%. The bottom (unlucky) decile rose **11.5 percentage points** (about **+1.4 games**), from 35.6% to 47.0%. Both groups landed near .500 — luck mean-reverts toward average, it does not turn good teams into bad ones.

Adding luck to the stage-1 next-season model (`win%_{t+1} ~ win%_t + net PPA_t`) raises in-sample R² from 0.241 to 0.256 (ΔR² +0.015), and it holds out of sample (LOSO 0.235 → 0.250, Δ +0.015). The luck coefficient stays negative after controlling for this year's record and efficiency, so this is not just "good records regress."

## 1. What luck is, and the sample

Luck = FBS-vs-FBS win% − PF^2.37 / (PF^2.37 + PA^2.37), using regular-season plus postseason point totals from stage 1. Mean luck in the pair file is about zero (-0.3%, SD 9.1% ≈ 1.1 games per 12-game season). Luck is almost uncorrelated with net PPA (r = 0.05) and only moderately tied to this year's win% (r = 0.41), which is what you want if luck is mostly close-game and margin-shape noise rather than a second efficiency metric.

The sample is the same 618 adjacent team-pairs as stage 1 (2019→2020, 2021→2022, …, 2024→2025), with a sensitivity that drops the 2019→2020 COVID-outcome pairs (n = 507).

## 2. (a) Does luck persist?  Expect ~0

| | n | r | Spearman | R² | coef (SE) |
|---|---|---|---|---|---|
| Luck_t → luck_{t+1} | 618 | 0.08 | 0.08 | 0.007 | 0.085 (0.042) |

A little bit of "luck" may be unmodeled strength or schedule, but almost none of it carries over. Next year's close-game outcomes are mostly a new draw.

## 3. (b) Change in win% on this year's luck

Outcome is `win%_{t+1} − win%_t`. A coefficient of −1 on luck means the team gives the surplus (or deficit) back one-for-one. Coefficients are for a 0–1 win% scale; the luck coefficient is also the games given back next year per lucky game this year, in 12-game units.

| Model | n | R² | luck coef (SE) | Other coefs (SE) |
|---|---|---|---|---|
| Luck only | 618 | 0.145 | -0.983 (0.096) | — |
| Luck + this-season win% | 618 | 0.291 | -0.538 (0.096) | win% -0.438 (0.039) |
| Luck + net PPA | 618 | 0.248 | -0.945 (0.090) | net PPA -0.417 (0.045) |
| Luck + net residual | 618 | 0.192 | -0.994 (0.094) | net residual -1.133 (0.190) |
| Luck + win% + net PPA | 618 | 0.294 | -0.428 (0.120) | win% -0.559 (0.089); net PPA 0.153 (0.101) |
| Luck + win% + net residual | 618 | 0.299 | -0.417 (0.105) | win% -0.550 (0.057); net residual 0.705 (0.259) |

Without controls, the give-back is essentially complete (-0.98). Controlling for this year's win% cuts the luck coefficient to -0.54 (SE 0.10) — still clearly negative, so lucky teams fade by more than the usual "good records come back to earth." Net PPA or the net residual alone barely moves the luck coefficient, which fits the near-zero correlation between luck and efficiency. In the fullest specification (win% + net PPA) the luck coefficient is **-0.43** (SE 0.12): each lucky game this season still predicts about -0.4 of a game given back next season after you already know the record and the PPA.

## 4. (c) Does luck improve the next-season win% model?

Stage-1 baseline: `win%_{t+1} ~ win%_t + net PPA_t`. Adding luck_t is equivalent to also giving the model this year's Pythagorean expected win% (because luck = win% − Pythagorean). A negative luck coefficient means last year's point totals have information about next year's record that the raw W-L — and even net PPA — do not fully capture.

| Model | n | In-sample R² | ΔR² | LOSO R² | Δ LOSO | luck coef (SE) |
|---|---|---|---|---|---|---|
| win% + net PPA | 618 | 0.241 | — | 0.235 | — | — |
| + luck | 618 | 0.256 | +0.015 | 0.250 | +0.015 | -0.428 (0.120) |

LOSO R² by this-season year:

| This season | n | win% + PPA | + luck | Δ |
|---|---|---|---|---|
| 2019 | 111 | 0.244 | 0.260 | +0.016 |
| 2021 | 128 | 0.255 | 0.275 | +0.020 |
| 2022 | 127 | 0.315 | 0.315 | +0.001 |
| 2023 | 123 | 0.098 | 0.129 | +0.031 |
| 2024 | 129 | 0.253 | 0.264 | +0.010 |

The increment is real but smaller than the stage-1 efficiency increment (that was ΔR² +0.025 for adding net PPA to W-L alone). Once luck is in the model, the net PPA coefficient shrinks (0.369 → 0.153), because Pythagorean and PPA are both strength measures. Luck is not a second efficiency stat; it is the part of the record that the point totals did not endorse.

## 5. (d) Lucky vs unlucky deciles

| Decile | n | Mean luck | This win% | Next win% | Δ win% | Δ games / 12 |
|---|---|---|---|---|---|---|
| 1 | 62 | -16.0% | 35.6% | 47.0% | +11.5% | +1.4 |
| 2 | 62 | -10.0% | 38.5% | 50.4% | +11.9% | +1.4 |
| 3 | 62 | -6.5% | 37.1% | 47.4% | +10.3% | +1.2 |
| 4 | 61 | -3.7% | 47.6% | 50.9% | +3.4% | +0.4 |
| 5 | 62 | -1.4% | 49.9% | 47.8% | -2.1% | -0.3 |
| 6 | 62 | +1.1% | 49.1% | 49.7% | +0.6% | +0.1 |
| 7 | 61 | +3.1% | 55.7% | 51.7% | -4.0% | -0.5 |
| 8 | 62 | +5.8% | 52.5% | 47.2% | -5.3% | -0.6 |
| 9 | 62 | +9.6% | 58.1% | 49.1% | -9.0% | -1.1 |
| 10 | 62 | +15.5% | 66.7% | 50.0% | -16.6% | -2.0 |

Top decile (mean luck +15.5% ≈ +1.9 games): win% falls from 66.7% to 50.0%, **-2.0 games** per 12. Bottom decile (mean luck -16.0% ≈ -1.9 games): win% rises from 35.6% to 47.0%, **+1.4 games** per 12. The next-year records sit near the league middle, not in the opposite tail.

## 6. Oregon, and three famous seasons

- **2021 Oregon, lucky.** 2021 Oregon went 9-4 in FBS-vs-FBS games (69.2%, average margin +1.6, Pythagorean 53.3%). Luck +16.0% is about +1.9 games in a 12-game season. The next year they went 9-3 (75.0%, change +5.8% ≈ +0.7 games per 12). They did *not* fade. The offense residual was already +6.1 and jumped again in 2022–23, and 2022 luck stayed high (+11). Luck is a residual, not a curse: when the underlying team is getting better, the record can keep rising.
- **2023 Oregon, slightly unlucky.** 2023 Oregon went 11-2 in FBS-vs-FBS games (84.6%, average margin +24.2, Pythagorean 88.9%). Luck -4.2% is about -0.5 games in a 12-game season. The next year they went 12-1 (92.3%, change +7.7% ≈ +0.9 games per 12). A mild deficit, an elite net residual (+10.7) and net PPA (+0.391), and a 12-1 encore. Consistent with a good team that left a couple of games on the table and then collected.

- **Lucky — 2021 Michigan State went 10-2 in FBS-vs-FBS games (83.3%, average margin +4.8, Pythagorean 59.7%). Luck +23.6% is about +2.8 games in a 12-game season. The next year they went 5-7 (41.7%, change -41.7% ≈ -5.0 games per 12).**
- **Lucky — 2022 TCU went 12-2 in FBS-vs-FBS games (85.7%, average margin +7.5, Pythagorean 63.0%). Luck +22.7% is about +2.7 games in a 12-game season. The next year they went 4-7 (36.4%, change -49.4% ≈ -5.9 games per 12).**
- **Unlucky — 2021 Nebraska went 2-9 in FBS-vs-FBS games (18.2%, average margin +1.6, Pythagorean 53.9%). Luck -35.7% is about -4.3 games in a 12-game season. The next year they went 3-8 (27.3%, change +9.1% ≈ +1.1 games per 12).**

Michigan State 2021 is the cleanest of the famous lucky seasons (same head coach the next year) and still gave the games back. TCU's 2022 CFP run is the headline close-game team, but 2023 also lost a lot of that roster — treat the drop as luck plus turnover, not luck alone. Nebraska 2021 is the classic unlucky season: they *outscored* FBS opponents and still went 2-9. The bounce was only modest (3-8), a reminder that a brutal luck number can sit on top of a program that really was struggling.

## 7. (e) Sensitivities

**Drop 2019→2020** (n = 507). Luck persistence r = 0.08. Δwin% on luck only: coef -0.98 (SE 0.10, R² 0.147). Luck + win% + net PPA: luck coef -0.42 (SE 0.13). Adding luck to the next-season model: in-sample R² 0.239 → 0.255 (Δ +0.015); LOSO 0.231 → 0.246 (Δ +0.015). Top/bottom decile Δwin% -18.5% / +10.3%. Same story as the main sample.

**Refit Pythagorean exponent.** Least-squares fit on this-season win% vs points scored and allowed (qualified non-COVID team-seasons only, no next-year leakage) gives **x = 2.374**, essentially the usual football 2.37 (SSE is essentially tied). Recomputed luck correlates 1.00 with the stage-1 luck column. Under the refit: luck persistence r = 0.08; Δwin% on luck coef -0.98 (SE 0.10); adding luck to win% + net PPA, ΔR² +0.015 (LOSO Δ +0.015). The conclusions do not change.

## 8. Caveats

- "Luck" is everything Pythagorean on raw point totals does not explain: close-game variance, blowout geometry, unmodeled opponent strength, and some true quality. The near-zero year-to-year correlation says most of it is not sticky skill, but it is not a pure coin flip either (r ≈ 0.08).
- Famous fade/bounce seasons can also be coaching and roster turnover (especially TCU 2022–23 and Washington 2023–24, which is why Washington is not used as a lead example).
- Oregon 2021 shows the other side: a lucky record plus a rising offense does not have to shrink.
- Records and luck are FBS-vs-FBS, regular plus postseason, same as stage 1.
- The ΔR² from adding luck is modest. This stage reports SEs and LOSO but not a formal bootstrap or nested F-test; that remains a later-stage option.
- The refit exponent uses a simple squared-error match of Pythagorean to win%. Other fitting choices would land just as close to 2.37.

## 9. Options for stage 3

1. **Point-margin target.** Repeat the next-season work with average point margin as the outcome: does this year's efficiency beat this year's margin for predicting next year's margin? This was option 1 on the stage-2 list and is still the clean complement to the win% models.
2. **Formal inference, and a horse race.** Confidence intervals or a bootstrap / nested F-test on the ΔR² from stages 1–2, plus a joint model that asks whether the situational residual still adds anything once net PPA (and maybe luck) is already in the regression.
3. **Close-game luck vs turnover.** Split one-score (or sub-8-point) game records out of Pythagorean luck, and split team-pairs by coaching or roster continuity, so famous drops like TCU 2022–23 are not counted as pure luck regression.

## Files

- `src/luck.py`
- `data/processed/luck_models.json`, `data/processed/luck_deciles.csv`
- `figures/luck_regression.png`
- `docs/RESULTS_luck.md`
