# Formal inference and a horse race — Stage 3

Generated 2026-11-10 from `data/processed/wins_next_pairs.csv` and `team_season_combined.csv`. Next-season sample is the same 618 adjacent team-pairs as stages 1–2 (2019→2020 and 2021–2025, 134 teams). Inference uses classical nested F-tests, cluster-robust (by team) standard errors, and a 2,000-replication bootstrap that resamples teams with replacement and keeps each team's seasons together (seed 20261110). Data source: CollegeFootballData.com (CFBD). Script: `src/inference.py`.

## Punchline

The stage-1 and stage-2 increments are real, not just in-sample noise. Adding net PPA to this season's win% raises in-sample R² by **+0.025** (F = 20.5, p < 0.001); adding the net residual instead raises it by **+0.028** (F = 22.8, p < 0.001). Adding luck on top of win% + net PPA raises it by **+0.015** (F = 12.7, p < 0.001). The two efficiency-over-W-L increments have leave-one-season-out 95% CIs that exclude zero. Luck's LOSO CI just includes zero ([-0.001, 0.035], bootstrap p = 0.059), so the F-test is stronger than the out-of-sample interval.

**Does the residual still add anything once PPA is in?** Yes — the net residual still adds something once net PPA is in the model (ΔR² +0.009, nested F p = 0.008; in the joint model the residual is +2.9 pp of next-season win% per SD, cluster p = 0.011). Net PPA itself is no longer distinguishable from zero in that joint model (cluster p = 0.683), but that is largely collinearity with this year's win% (VIF ≈ 6.5), not a claim that PPA is useless: it clearly adds when it is the one extra term on top of W-L (p < 0.001). The unique increment is small (LOSO ΔR² +0.008), and its LOSO bootstrap CI includes zero [-0.003, 0.021]. So the residual's *unique* next-season contribution is statistically detectable in the in-sample tests and the joint-model coefficient, but it is not a large or out-of-sample-decisive increment.

Luck stays negative and significant in the joint model (-3.6 pp per SD, cluster p < 0.001): teams that outperformed Pythagorean this year win less next year, holding record and efficiency fixed. Dropping 2019→2020 leaves the F-tests and the joint model in the same place (the PPA-over-W-L LOSO CI becomes a little wider and just includes zero). Persistence correlations, with a team-clustered bootstrap on the same adjacent pairs as the original note, confirm residual < raw success and residual < PPA: the paired differences exclude zero even though the separate r intervals overlap.

## 1. (a) Nested tests and bootstrap CIs for ΔR²

In-sample R² always rises when a regressor is added, so a bootstrap confidence interval for in-sample ΔR² is a statement about the *size* of the increment, not a test that it is zero. The tests of "does this add signal?" are the nested F, the cluster-robust Wald test on the added coefficient, and the LOSO ΔR² (which can be negative) with its bootstrap CI. Bootstrap p-values below are two-sided, from the LOSO ΔR² draws (a fairer null than in-sample ΔR²).

| Increment | n | R² restricted → full | In-sample ΔR² | Boot 95% CI (in-sample) | F | F p | LOSO ΔR² | Boot 95% CI (LOSO) | LOSO boot p |
|---|---|---|---|---|---|---|---|---|---|
| + net PPA | win% | 618 | 0.215 → 0.241 | +0.025 | [0.008, 0.051] | 20.53 | < 0.001 | +0.024 | [0.002, 0.047] | 0.020 |
| + net residual | win% | 618 | 0.215 → 0.243 | +0.028 | [0.010, 0.052] | 22.83 | < 0.001 | +0.027 | [0.006, 0.050] | 0.008 |
| + luck | win% + net PPA | 618 | 0.241 → 0.256 | +0.015 | [0.004, 0.036] | 12.73 | < 0.001 | +0.015 | [-0.001, 0.035] | 0.059 |
| + net residual | win% + net PPA | 618 | 0.241 → 0.249 | +0.009 | [0.001, 0.022] | 7.10 | 0.008 | +0.008 | [-0.003, 0.021] | 0.244 |
| + net residual | win% + net PPA + luck | 618 | 0.256 → 0.262 | +0.006 | [0.000, 0.018] | 5.18 | 0.023 | +0.006 | [-0.004, 0.016] | 0.414 |

Cluster-robust Wald on the added coefficient (same models): net PPA given win% z = 4.25, p < 0.001; net residual given win% z = 4.76, p < 0.001; luck given win% + PPA z = -3.61, p < 0.001; residual given win% + PPA z = 2.97, p = 0.003.

In plain language: PPA over W-L LOSO ΔR² +0.024, 95% CI [0.002, 0.047], excludes 0 (bootstrap p = 0.020); residual over W-L LOSO ΔR² +0.027, 95% CI [0.006, 0.050], excludes 0 (bootstrap p = 0.008); luck over W-L + PPA LOSO ΔR² +0.015, 95% CI [-0.001, 0.035], includes 0 (bootstrap p = 0.059); residual over W-L + PPA LOSO ΔR² +0.008, 95% CI [-0.003, 0.021], includes 0 (bootstrap p = 0.244). The residual's *unique* increment on top of PPA is smaller (in-sample ΔR² +0.009) but the F-test rejects zero (p = 0.008), and it still rejects after luck is included too (ΔR² +0.006, p = 0.023).

## 2. (b) Joint model and the horse race

`win%_{t+1} ~ win%_t + net PPA_t + net residual_t + luck_t`, with standard errors clustered by team. Standardized columns are ordinary betas (SD of next win% per SD of the predictor) and a more readable scale: percentage points of next-season win% per one-SD bump in the predictor, holding the others fixed. Cluster-robust 95% CIs.

| Predictor | Coef | Cluster SE | 95% CI | Cluster p | Std beta | pp / 1 SD | 95% CI (pp / SD) |
|---|---|---|---|---|---|---|---|
| This-season win% | 0.425 | 0.083 | [0.262, 0.589] | < 0.001 | +0.418 | +9.5 | [5.9, 13.2] |
| Net PPA | 0.040 | 0.099 | [-0.154, 0.234] | 0.683 | +0.032 | +0.7 | [-2.8, 4.2] |
| Net residual | 0.659 | 0.260 | [0.149, 1.169] | 0.011 | +0.129 | +2.9 | [0.7, 5.2] |
| Luck | -0.396 | 0.119 | [-0.629, -0.162] | < 0.001 | -0.158 | -3.6 | [-5.7, -1.5] |

Joint-model R² = 0.262 (n = 618). A one-SD better residual is associated with +2.9 percentage points of next-season win% after you already know the record, net PPA, and luck (95% CI [0.7, 5.2]). A one-SD luck surplus is associated with -3.6 pp (95% CI [-5.7, -1.5]). Net PPA's unique piece in this specification is +0.7 pp and the CI crosses zero.

**Collinearity.** VIFs: this-season win% 6.33, net PPA 6.49, net residual 2.65, luck 1.91. Win% and net PPA share a lot of variance (they correlated at r = 0.84 in stage 1), so their *separate* joint-model coefficients should be read carefully — VIF near 6 is elevated, not catastrophic. The residual (VIF 2.7) and luck (VIF 1.9) are in a healthier range. That is why PPA can look "insignificant" in the kitchen-sink model even though it clearly improves a win%-only regression: most of what PPA knows is already in the record (and, once luck is in, in the point totals). The residual is less tied to this year's W-L, so more of its next-season signal is unique.

### Offense / defense split

Same joint idea, but net PPA and net residual are split into offense and defense pieces.

| Predictor | Coef | Cluster SE | 95% CI | Cluster p | Std beta | pp / 1 SD | 95% CI (pp / SD) |
|---|---|---|---|---|---|---|---|
| This-season win% | 0.426 | 0.083 | [0.263, 0.588] | < 0.001 | +0.418 | +9.5 | [5.9, 13.2] |
| Offense PPA | 0.095 | 0.144 | [-0.188, 0.378] | 0.510 | +0.048 | +1.1 | [-2.1, 4.3] |
| Defense PPA | -0.001 | 0.146 | [-0.287, 0.285] | 0.994 | -0.001 | -0.0 | [-3.1, 3.1] |
| Offense residual | 0.440 | 0.389 | [-0.323, 1.204] | 0.258 | +0.069 | +1.6 | [-1.2, 4.3] |
| Defense residual | 0.848 | 0.345 | [0.171, 1.525] | 0.014 | +0.126 | +2.9 | [0.6, 5.2] |
| Luck | -0.397 | 0.120 | [-0.631, -0.162] | < 0.001 | -0.158 | -3.6 | [-5.7, -1.5] |

Split-model R² = 0.263. In this fuller specification the unique residual signal is clearer on **defense** (cluster p = 0.014, +2.9 pp per SD) than on offense (cluster p = 0.258), where offense residual and offense PPA already correlate at r = 0.82. That split should not be over-read: the simpler one-at-a-time horse races do not reject (offense residual given offense PPA + win%, p = 0.853; defense residual given defense PPA + win%, p = 0.205). A joint F-test that *both* residuals are zero given both PPAs and win% does reject (ΔR² +0.009, F = 3.83, p = 0.022), but it is only marginal once luck is included (p = 0.057). The two PPA terms as a block, given the two residuals and win%, are also only marginal (p = 0.074).

VIFs in the split model stay in the same band (win% 6.33, offense PPA 5.43, defense PPA 4.37, offense residual 3.43, defense residual 2.67, luck 1.91).

## 3. (c) Sensitivity: drop 2019→2020

n = 507 pairs, 134 teams. F-tests and the joint-model residual / luck coefficients are essentially unchanged. The PPA-over-W-L LOSO interval is a little weaker and just includes zero.

| Increment | In-sample ΔR² | F p | LOSO ΔR² | LOSO boot 95% CI | LOSO boot p |
|---|---|---|---|---|---|
| + net PPA | win% | +0.025 | < 0.001 | +0.023 | [-0.003, 0.048] | 0.099 |
| + net residual | win% | +0.027 | < 0.001 | +0.027 | [0.003, 0.052] | 0.025 |
| + luck | win% + PPA | +0.015 | 0.001 | +0.015 | [-0.004, 0.037] | 0.153 |
| + net residual | win% + PPA | +0.008 | 0.020 | +0.008 | [-0.005, 0.021] | 0.375 |

Joint-model cluster p: win% < 0.001, net PPA 0.769, net residual 0.021, luck 0.002. Residual pp / SD +2.8 (95% CI [0.4, 5.1]).

## 4. (d) Persistence correlations, with uncertainty

Rebuilt from `team_season_combined.csv` on the same rule as the original persistence note: adjacent pairs only (2021→2022 through 2024→2025), ≥300 modeling snaps in both seasons of a pair. That recovers n = 488 offense pairs and n = 488 defense pairs, and the same point estimates (offense residual 0.39, raw success 0.47, PPA 0.46; defense residual 0.32, defensive PPA 0.44). 95% CIs are from a team-clustered bootstrap (2,000 reps). Differences use the same draws, so they respect the dependence of two correlations measured on the same teams.

| Series | r | 95% CI |
|---|---|---|
| Offense residual | 0.39 [0.30, 0.47] |
| Offense raw success | 0.47 [0.38, 0.54] |
| Offense mean PPA | 0.46 [0.37, 0.53] |
| Defense residual | 0.32 [0.23, 0.41] |
| Defensive PPA | 0.44 [0.34, 0.52] |

Raw success minus residual: +0.08, 95% CI [0.03, 0.13], bootstrap p = 0.001. Offense PPA minus residual: +0.07, 95% CI [0.01, 0.13], bootstrap p = 0.018. Defensive PPA minus defense residual: +0.12, 95% CI [0.04, 0.19], bootstrap p = 0.001.

The original ranking is supported. Residuals persist, but less than raw success or PPA. The separate r intervals overlap — they usually will — but the paired differences, which are the right comparison, exclude zero: raw success minus residual +0.08 (p = 0.001), offense PPA minus residual +0.07 (p = 0.018), defensive PPA minus defense residual +0.12 (p = 0.001). The defense gap is the clearest of the three.

## 5. Caveats

- In-sample ΔR² cannot be negative, so its bootstrap CI is not a hypothesis test of zero. Use the F / Wald tests and the LOSO ΔR² CI for that.
- Team-clustered SEs and a team-clustered bootstrap allow any one program's seasons to be dependent. They do not fix unmodeled coaching changes, portal turnover, or schedule.
- VIF near 6 for win% and net PPA means the joint-model horse race between those two should be read as "little unique PPA leftover after the record (and luck)," not "PPA does not predict."
- The offense/defense split gives the residual more moving parts and less precision, especially on offense. The defense-residual finding is interesting but is one split of one sample.
- Persistence CIs use four adjacent year-pairs and a COVID-skipped 2019→2021 gap that is still left out of the pooled figure, matching the earlier writeup.
- 2,000 bootstrap replications, seed 20261110. Percentile intervals; no BCa adjustment.

## 6. Options for what to do next

1. **Fold stages 1–3 into the writeup and the GitHub repo.** The canvas and `asajor8/cfb-situational-residuals` still describe the original residual project. This is the natural packaging step: add the wins / luck / inference results, figures, and a short methods note, without expanding the question further.
2. **Point-margin target.** The unused stage-2 option: does this year's efficiency beat this year's margin for predicting next year's *margin* (not just win%)? Formal tests can follow the same F / bootstrap pattern.
3. **Close-game luck vs turnover.** Split one-score games out of Pythagorean luck, and split team-pairs by coaching or roster continuity, so famous drops like TCU 2022–23 are not counted as pure luck regression.

## Files

- `src/inference.py`
- `data/processed/inference_models.json`
- `figures/inference_ci.png`
- `docs/RESULTS_inference.md`
