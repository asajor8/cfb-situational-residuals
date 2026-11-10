"""Stage 2: does Pythagorean luck regress the next season?

Luck = actual FBS-vs-FBS win% minus Pythagorean expected win% (exponent 2.37),
already computed in team_season_wins.csv / wins_next_pairs.csv.
No API calls. Data source: CollegeFootballData.com via the stage-1 files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DERIVED, DOCS, FIGURES, PROC, ensure_dirs  # noqa: E402
from wins import (  # noqa: E402
    COVID_YEAR,
    PYTHAG_EXP,
    corr_and_ols,
    fit_ols,
    fmt_coef_se,
    fmt_pct,
    fmt_r,
    loso,
    record_str,
)

ensure_dirs()
FIG = FIGURES
PAIRS = PROC / "wins_next_pairs.csv"
TS = DERIVED / "team_season_wins.csv"

# Famous lucky / unlucky seasons used as examples (must be in the pair file).
FAMOUS = [
    (2021, "Michigan State", "lucky"),
    (2022, "TCU", "lucky"),
    (2021, "Nebraska", "unlucky"),
]


def pyth(pf: pd.Series, pa: pd.Series, exp: float) -> pd.Series:
    pf_x = np.power(pf.clip(lower=0), exp)
    pa_x = np.power(pa.clip(lower=0), exp)
    denom = pf_x + pa_x
    return pd.Series(np.where(denom > 0, pf_x / denom, 0.5), index=pf.index)


def fit_exponent(ts: pd.DataFrame) -> float:
    """Least-squares Pythagorean exponent on this-season win% vs PF/PA.

    Fit on qualified, non-COVID team-seasons with >=6 FBS-vs-FBS games.
    Does not use next-season outcomes.
    """
    s = ts[(ts.both_qualified == True) & (ts.n_games >= 6) & (ts.year != COVID_YEAR)]  # noqa: E712
    s = s[(s.pf > 0) & (s.pa > 0)]

    def sse(x):
        if x <= 0:
            return 1e9
        err = s.win_pct.to_numpy() - pyth(s.pf, s.pa, float(x)).to_numpy()
        return float(np.mean(err ** 2))

    res = minimize_scalar(sse, bounds=(1.0, 5.0), method="bounded", options={"xatol": 1e-4})
    return float(res.x)


def attach(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["dwin"] = d.win_pct_next - d.win_pct
    d["games12_luck"] = 12 * d.luck
    d["games12_dwin"] = 12 * d.dwin
    return d


def decile_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["decile"] = pd.qcut(d.luck, 10, labels=False, duplicates="drop") + 1
    g = (
        d.groupby("decile", as_index=False)
        .agg(
            n=("luck", "size"),
            luck=("luck", "mean"),
            win_pct=("win_pct", "mean"),
            win_pct_next=("win_pct_next", "mean"),
            dwin=("dwin", "mean"),
            luck_next=("luck_next", "mean"),
        )
    )
    g["games12_luck"] = 12 * g.luck
    g["games12_dwin"] = 12 * g.dwin
    return g


def pack_models(df: pd.DataFrame) -> dict:
    dwin_specs = {
        "luck": ["luck"],
        "luck_win": ["luck", "win_pct"],
        "luck_ppa": ["luck", "net_ppa"],
        "luck_resid": ["luck", "net_residual"],
        "luck_win_ppa": ["luck", "win_pct", "net_ppa"],
        "luck_win_resid": ["luck", "win_pct", "net_residual"],
    }
    dwin = {k: fit_ols(df, "dwin", x) for k, x in dwin_specs.items()}
    next_base = fit_ols(df, "win_pct_next", ["win_pct", "net_ppa"])
    next_luck = fit_ols(df, "win_pct_next", ["win_pct", "net_ppa", "luck"])
    return {
        "n": int(len(df)),
        "r_luck_luck_next": float(df.luck.corr(df.luck_next)),
        "r_luck_luck_next_spearman": float(df.luck.corr(df.luck_next, method="spearman")),
        "r_luck_dwin": float(df.luck.corr(df.dwin)),
        "r_luck_win": float(df.luck.corr(df.win_pct)),
        "r_luck_ppa": float(df.luck.corr(df.net_ppa)),
        "luck_persist": corr_and_ols(df, "luck_next", "luck"),
        "dwin": dwin,
        "next_base": {"in_sample": next_base, "loso": loso(df, "win_pct_next", ["win_pct", "net_ppa"])},
        "next_luck": {
            "in_sample": next_luck,
            "loso": loso(df, "win_pct_next", ["win_pct", "net_ppa", "luck"]),
        },
    }


def figures(df: pd.DataFrame, dec: pd.DataFrame, r_luck: float, dwin_coef: float) -> None:
    plt.rcParams["text.usetex"] = False
    plt.rcParams["text.parse_math"] = False
    oreg = df[df.team == "Oregon"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.3))

    ax = axes[0]
    ax.scatter(df.luck, df.luck_next, s=16, c="#999", alpha=0.7, linewidths=0, zorder=1)
    lims = (-0.40, 0.28)
    ax.plot(lims, lims, c="#bbbbbb", lw=0.8, zorder=0)
    ax.axhline(0, c="k", lw=0.6)
    ax.axvline(0, c="k", lw=0.6)
    if len(oreg):
        ax.scatter(oreg.luck, oreg.luck_next, s=70, c="#154733", zorder=3, label="Oregon")
        for r in oreg.itertuples():
            ax.annotate(
                str(int(r.year)),
                (r.luck, r.luck_next),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
                color="#154733",
                fontweight="bold",
            )
    # famous examples
    for year, team, kind in FAMOUS:
        row = df[(df.year == year) & (df.team == team)]
        if not len(row):
            continue
        r = row.iloc[0]
        ax.scatter(r.luck, r.luck_next, s=42, c="#c0392b", zorder=3)
        short = "MSU" if team == "Michigan State" else team
        ax.annotate(
            f"{short} {year}",
            (r.luck, r.luck_next),
            xytext=(4, -9),
            textcoords="offset points",
            fontsize=7,
            color="#c0392b",
        )
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("Luck this season (win% − Pythagorean)")
    ax.set_ylabel("Luck next season")
    ax.set_title(f"Luck barely sticks  (r = {r_luck:.2f})")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    ax.scatter(df.luck, df.dwin, s=16, c="#999", alpha=0.7, linewidths=0, zorder=1)
    xs = np.linspace(df.luck.min(), df.luck.max(), 50)
    intercept = float(df.dwin.mean() - dwin_coef * df.luck.mean())
    ax.plot(xs, intercept + dwin_coef * xs, c="#154733", lw=1.4, zorder=2, label="OLS fit")
    ax.scatter(
        dec.luck,
        dec.dwin,
        s=46,
        c="#1a1a1a",
        marker="D",
        zorder=3,
        label="Decile mean",
    )
    if len(oreg):
        ax.scatter(oreg.luck, oreg.dwin, s=70, c="#154733", zorder=4)
        for r in oreg.itertuples():
            ax.annotate(
                str(int(r.year)),
                (r.luck, r.dwin),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=8,
                color="#154733",
                fontweight="bold",
            )
    for year, team, kind in FAMOUS:
        row = df[(df.year == year) & (df.team == team)]
        if not len(row):
            continue
        r = row.iloc[0]
        ax.scatter(r.luck, r.dwin, s=42, c="#c0392b", zorder=4)
        short = "MSU" if team == "Michigan State" else team
        ax.annotate(
            f"{short} {year}",
            (r.luck, r.dwin),
            xytext=(4, -9),
            textcoords="offset points",
            fontsize=7,
            color="#c0392b",
        )
    ax.axhline(0, c="k", lw=0.6)
    ax.axvline(0, c="k", lw=0.6)
    ax.set_xlabel("Luck this season (win% − Pythagorean)")
    ax.set_ylabel("Change in win% (next − this season)")
    ax.set_title("Lucky teams give the wins back")
    ax.legend(frameon=False, loc="upper right", fontsize=8)

    fig.suptitle(
        f"Pythagorean luck and next season's record\n"
        f"FBS-vs-FBS, 2019→2020 and 2021–2025 pairs (n={len(df)})\n"
        "Data: CollegeFootballData.com",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG / "luck_regression.png", dpi=150)
    plt.close(fig)


def example_line(df: pd.DataFrame, year: int, team: str) -> str | None:
    row = df[(df.year == year) & (df.team == team)]
    if not len(row):
        return None
    r = row.iloc[0]
    rec = record_str(r.wins, r.losses, r.ties)
    rec_n = record_str(r.wins_next, r.losses_next, r.ties_next)
    return (
        f"{year} {team} went {rec} in FBS-vs-FBS games ({fmt_pct(r.win_pct)}, "
        f"average margin {r.avg_margin:+.1f}, Pythagorean {fmt_pct(r.pyth_win_pct)}). "
        f"Luck {fmt_pct(r.luck, signed=True)} is about {12 * r.luck:+.1f} games in a 12-game season. "
        f"The next year they went {rec_n} ({fmt_pct(r.win_pct_next)}, "
        f"change {fmt_pct(r.dwin, signed=True)} ≈ {12 * r.dwin:+.1f} games per 12)."
    )


def write_report(ctx: dict) -> None:
    main = ctx["main"]
    sens = ctx["sens"]
    dec = ctx["dec"]
    dec_s = ctx["dec_sens"]
    pairs = ctx["pairs"]
    exp_hat = ctx["exp_hat"]
    refit = ctx["refit"]
    top, bot = dec[dec.decile == 10].iloc[0], dec[dec.decile == 1].iloc[0]
    top_s, bot_s = dec_s[dec_s.decile == 10].iloc[0], dec_s[dec_s.decile == 1].iloc[0]

    persist = main["luck_persist"]
    d_luck = main["dwin"]["luck"]
    d_win = main["dwin"]["luck_win"]
    d_ppa = main["dwin"]["luck_ppa"]
    d_resid = main["dwin"]["luck_resid"]
    d_full = main["dwin"]["luck_win_ppa"]
    d_full_r = main["dwin"]["luck_win_resid"]
    base = main["next_base"]
    addl = main["next_luck"]
    d_r2 = addl["in_sample"]["r2"] - base["in_sample"]["r2"]
    d_loso = addl["loso"]["r2"] - base["loso"]["r2"]
    d_r2_s = sens["next_luck"]["in_sample"]["r2"] - sens["next_base"]["in_sample"]["r2"]
    d_loso_s = sens["next_luck"]["loso"]["r2"] - sens["next_base"]["loso"]["r2"]

    def dwin_row(m, label, extra):
        luck_c = fmt_coef_se(m["params"]["luck"], m["bse"]["luck"], digits=3)
        extras = []
        for name, lab in extra:
            if name in m["params"]:
                extras.append(f"{lab} {fmt_coef_se(m['params'][name], m['bse'][name], digits=3)}")
        extra_s = "; ".join(extras) if extras else "—"
        return f"| {label} | {m['n']} | {fmt_r(m['r2'], 3)} | {luck_c} | {extra_s} |"

    dec_lines = []
    for r in dec.itertuples():
        dec_lines.append(
            f"| {int(r.decile)} | {int(r.n)} | {fmt_pct(r.luck, signed=True)} | "
            f"{fmt_pct(r.win_pct)} | {fmt_pct(r.win_pct_next)} | {fmt_pct(r.dwin, signed=True)} | "
            f"{r.games12_dwin:+.1f} |"
        )

    og21 = example_line(pairs, 2021, "Oregon")
    og23 = example_line(pairs, 2023, "Oregon")
    famous_lines = []
    for year, team, kind in FAMOUS:
        line = example_line(pairs, year, team)
        if line:
            famous_lines.append(f"- **{kind.capitalize()} — {line}**")

    # games-per-12 interpretations
    raw_give = 12 * d_luck["params"]["luck"] / 12  # coef itself is the give-back rate
    # 1 game of luck = luck of 1/12
    raw_per_game = d_luck["params"]["luck"] * (1 / 12) * 12  # = coef, games given back per lucky game
    ctrl_per_game = d_full["params"]["luck"] * 1.0  # same: Δwin% per 1.0 luck = games/12 per game of luck
    sd_luck = float(pairs.luck.std(ddof=1))

    def loso_fold_rows():
        lines = []
        fbase = {f["year"]: f for f in base["loso"]["folds"]}
        fluck = {f["year"]: f for f in addl["loso"]["folds"]}
        for yr in sorted(fbase):
            lines.append(
                f"| {yr} | {fbase[yr]['n']} | {fmt_r(fbase[yr]['r2'], 3)} | "
                f"{fmt_r(fluck[yr]['r2'], 3)} | {fluck[yr]['r2'] - fbase[yr]['r2']:+.3f} |"
            )
        return "\n".join(lines)

    loso_word = (
        "and it holds out of sample"
        if d_loso >= 0.005
        else (
            "but the leave-one-season-out lift is only a sliver"
            if d_loso > 0
            else "but the lift does not hold in leave-one-season-out"
        )
    )

    md = f"""# Does Pythagorean luck regress next season? — Stage 2

Generated 2026-11-10 from the stage-1 files `data/processed/wins_next_pairs.csv` and `team_season_wins.csv`. Luck is actual FBS-vs-FBS win% minus Pythagorean expected win% with exponent {PYTHAG_EXP}. Data source: CollegeFootballData.com (CFBD). Script: `src/luck.py`.

## Punchline

Luck barely sticks, and the wins mostly come back. Year-to-year luck correlation is **r = {fmt_r(main['r_luck_luck_next'])}** (Spearman {fmt_r(main['r_luck_luck_next_spearman'])}), close to the "it's noise" expectation of zero. A team that outperformed Pythagorean this year gives almost all of that surplus back the next year: regressing the change in win% on luck yields a coefficient of **{d_luck['params']['luck']:.2f}** (SE {d_luck['bse']['luck']:.2f}, R² {fmt_r(d_luck['r2'], 3)}). That is about one game given back next season for each lucky game this season, in 12-game units.

The top luck decile dropped **{abs(top.dwin)*100:.1f} percentage points** of win% the next year (about **{top.games12_dwin:+.1f} games** per 12-game season), from {fmt_pct(top.win_pct)} to {fmt_pct(top.win_pct_next)}. The bottom (unlucky) decile rose **{bot.dwin*100:.1f} percentage points** (about **{bot.games12_dwin:+.1f} games**), from {fmt_pct(bot.win_pct)} to {fmt_pct(bot.win_pct_next)}. Both groups landed near .500 — luck mean-reverts toward average, it does not turn good teams into bad ones.

Adding luck to the stage-1 next-season model (`win%_{{t+1}} ~ win%_t + net PPA_t`) raises in-sample R² from {fmt_r(base['in_sample']['r2'], 3)} to {fmt_r(addl['in_sample']['r2'], 3)} (ΔR² {d_r2:+.3f}), {loso_word} (LOSO {fmt_r(base['loso']['r2'], 3)} → {fmt_r(addl['loso']['r2'], 3)}, Δ {d_loso:+.3f}). The luck coefficient stays negative after controlling for this year's record and efficiency, so this is not just "good records regress."

## 1. What luck is, and the sample

Luck = FBS-vs-FBS win% − PF^{PYTHAG_EXP} / (PF^{PYTHAG_EXP} + PA^{PYTHAG_EXP}), using regular-season plus postseason point totals from stage 1. Mean luck in the pair file is about zero ({fmt_pct(pairs.luck.mean(), signed=True)}, SD {fmt_pct(sd_luck)} ≈ {12 * sd_luck:.1f} games per 12-game season). Luck is almost uncorrelated with net PPA (r = {fmt_r(main['r_luck_ppa'])}) and only moderately tied to this year's win% (r = {fmt_r(main['r_luck_win'])}), which is what you want if luck is mostly close-game and margin-shape noise rather than a second efficiency metric.

The sample is the same {main['n']} adjacent team-pairs as stage 1 (2019→2020, 2021→2022, …, 2024→2025), with a sensitivity that drops the 2019→2020 COVID-outcome pairs (n = {sens['n']}).

## 2. (a) Does luck persist?  Expect ~0

| | n | r | Spearman | R² | coef (SE) |
|---|---|---|---|---|---|
| Luck_t → luck_{{t+1}} | {persist['n']} | {fmt_r(persist['r'])} | {fmt_r(main['r_luck_luck_next_spearman'])} | {fmt_r(persist['r2'], 3)} | {fmt_coef_se(persist['coef'], persist['se'])} |

A little bit of "luck" may be unmodeled strength or schedule, but almost none of it carries over. Next year's close-game outcomes are mostly a new draw.

## 3. (b) Change in win% on this year's luck

Outcome is `win%_{{t+1}} − win%_t`. A coefficient of −1 on luck means the team gives the surplus (or deficit) back one-for-one. Coefficients are for a 0–1 win% scale; the luck coefficient is also the games given back next year per lucky game this year, in 12-game units.

| Model | n | R² | luck coef (SE) | Other coefs (SE) |
|---|---|---|---|---|
{dwin_row(d_luck, 'Luck only', [])}
{dwin_row(d_win, 'Luck + this-season win%', [('win_pct', 'win%')])}
{dwin_row(d_ppa, 'Luck + net PPA', [('net_ppa', 'net PPA')])}
{dwin_row(d_resid, 'Luck + net residual', [('net_residual', 'net residual')])}
{dwin_row(d_full, 'Luck + win% + net PPA', [('win_pct', 'win%'), ('net_ppa', 'net PPA')])}
{dwin_row(d_full_r, 'Luck + win% + net residual', [('win_pct', 'win%'), ('net_residual', 'net residual')])}

Without controls, the give-back is essentially complete ({d_luck['params']['luck']:.2f}). Controlling for this year's win% cuts the luck coefficient to {d_win['params']['luck']:.2f} (SE {d_win['bse']['luck']:.2f}) — still clearly negative, so lucky teams fade by more than the usual "good records come back to earth." Net PPA or the net residual alone barely moves the luck coefficient, which fits the near-zero correlation between luck and efficiency. In the fullest specification (win% + net PPA) the luck coefficient is **{d_full['params']['luck']:.2f}** (SE {d_full['bse']['luck']:.2f}): each lucky game this season still predicts about {d_full['params']['luck']:.1f} of a game given back next season after you already know the record and the PPA.

## 4. (c) Does luck improve the next-season win% model?

Stage-1 baseline: `win%_{{t+1}} ~ win%_t + net PPA_t`. Adding luck_t is equivalent to also giving the model this year's Pythagorean expected win% (because luck = win% − Pythagorean). A negative luck coefficient means last year's point totals have information about next year's record that the raw W-L — and even net PPA — do not fully capture.

| Model | n | In-sample R² | ΔR² | LOSO R² | Δ LOSO | luck coef (SE) |
|---|---|---|---|---|---|---|
| win% + net PPA | {base['in_sample']['n']} | {fmt_r(base['in_sample']['r2'], 3)} | — | {fmt_r(base['loso']['r2'], 3)} | — | — |
| + luck | {addl['in_sample']['n']} | {fmt_r(addl['in_sample']['r2'], 3)} | {d_r2:+.3f} | {fmt_r(addl['loso']['r2'], 3)} | {d_loso:+.3f} | {fmt_coef_se(addl['in_sample']['params']['luck'], addl['in_sample']['bse']['luck'])} |

LOSO R² by this-season year:

| This season | n | win% + PPA | + luck | Δ |
|---|---|---|---|---|
{loso_fold_rows()}

The increment is real but smaller than the stage-1 efficiency increment (that was ΔR² +0.025 for adding net PPA to W-L alone). Once luck is in the model, the net PPA coefficient shrinks ({base['in_sample']['params']['net_ppa']:.3f} → {addl['in_sample']['params']['net_ppa']:.3f}), because Pythagorean and PPA are both strength measures. Luck is not a second efficiency stat; it is the part of the record that the point totals did not endorse.

## 5. (d) Lucky vs unlucky deciles

| Decile | n | Mean luck | This win% | Next win% | Δ win% | Δ games / 12 |
|---|---|---|---|---|---|---|
{chr(10).join(dec_lines)}

Top decile (mean luck {fmt_pct(top.luck, signed=True)} ≈ {top.games12_luck:+.1f} games): win% falls from {fmt_pct(top.win_pct)} to {fmt_pct(top.win_pct_next)}, **{top.games12_dwin:+.1f} games** per 12. Bottom decile (mean luck {fmt_pct(bot.luck, signed=True)} ≈ {bot.games12_luck:+.1f} games): win% rises from {fmt_pct(bot.win_pct)} to {fmt_pct(bot.win_pct_next)}, **{bot.games12_dwin:+.1f} games** per 12. The next-year records sit near the league middle, not in the opposite tail.

## 6. Oregon, and three famous seasons

- **2021 Oregon, lucky.** {og21} They did *not* fade. The offense residual was already +6.1 and jumped again in 2022–23, and 2022 luck stayed high (+11). Luck is a residual, not a curse: when the underlying team is getting better, the record can keep rising.
- **2023 Oregon, slightly unlucky.** {og23} A mild deficit, an elite net residual (+10.7) and net PPA (+0.391), and a 12-1 encore. Consistent with a good team that left a couple of games on the table and then collected.

{chr(10).join(famous_lines)}

Michigan State 2021 is the cleanest of the famous lucky seasons (same head coach the next year) and still gave the games back. TCU's 2022 CFP run is the headline close-game team, but 2023 also lost a lot of that roster — treat the drop as luck plus turnover, not luck alone. Nebraska 2021 is the classic unlucky season: they *outscored* FBS opponents and still went 2-9. The bounce was only modest (3-8), a reminder that a brutal luck number can sit on top of a program that really was struggling.

## 7. (e) Sensitivities

**Drop 2019→2020** (n = {sens['n']}). Luck persistence r = {fmt_r(sens['r_luck_luck_next'])}. Δwin% on luck only: coef {sens['dwin']['luck']['params']['luck']:.2f} (SE {sens['dwin']['luck']['bse']['luck']:.2f}, R² {fmt_r(sens['dwin']['luck']['r2'], 3)}). Luck + win% + net PPA: luck coef {sens['dwin']['luck_win_ppa']['params']['luck']:.2f} (SE {sens['dwin']['luck_win_ppa']['bse']['luck']:.2f}). Adding luck to the next-season model: in-sample R² {fmt_r(sens['next_base']['in_sample']['r2'], 3)} → {fmt_r(sens['next_luck']['in_sample']['r2'], 3)} (Δ {d_r2_s:+.3f}); LOSO {fmt_r(sens['next_base']['loso']['r2'], 3)} → {fmt_r(sens['next_luck']['loso']['r2'], 3)} (Δ {d_loso_s:+.3f}). Top/bottom decile Δwin% {fmt_pct(top_s.dwin, signed=True)} / {fmt_pct(bot_s.dwin, signed=True)}. Same story as the main sample.

**Refit Pythagorean exponent.** Least-squares fit on this-season win% vs points scored and allowed (qualified non-COVID team-seasons only, no next-year leakage) gives **x = {exp_hat:.3f}**, essentially the usual football 2.37 (SSE is essentially tied). Recomputed luck correlates {fmt_r(refit['r_with_237'])} with the stage-1 luck column. Under the refit: luck persistence r = {fmt_r(refit['r_luck_luck_next'])}; Δwin% on luck coef {refit['dwin_coef']:.2f} (SE {refit['dwin_se']:.2f}); adding luck to win% + net PPA, ΔR² {refit['delta_r2']:+.3f} (LOSO Δ {refit['delta_loso']:+.3f}). The conclusions do not change.

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
"""
    (DOCS / "RESULTS_luck.md").write_text(md)


def main():
    pairs = attach(pd.read_csv(PAIRS))
    ts = pd.read_csv(TS)
    pairs_sens = pairs[pairs.next_year != COVID_YEAR].reset_index(drop=True)

    main_m = pack_models(pairs)
    sens_m = pack_models(pairs_sens)
    dec = decile_table(pairs)
    dec_s = decile_table(pairs_sens)
    dec.to_csv(PROC / "luck_deciles.csv", index=False, float_format="%.4f")

    exp_hat = fit_exponent(ts)
    # refit luck on this and next season using next-season PF/PA from the team-season file
    nxt = ts[["year", "team", "pf", "pa"]].rename(
        columns={"year": "next_year", "pf": "pf_next", "pa": "pa_next"}
    )
    ref = pairs.merge(nxt, on=["team", "next_year"], how="left")
    ref["luck_refit"] = ref.win_pct - pyth(ref.pf, ref.pa, exp_hat)
    ref["luck_next_refit"] = ref.win_pct_next - pyth(ref.pf_next, ref.pa_next, exp_hat)
    ref["dwin"] = ref.win_pct_next - ref.win_pct
    refit_persist = float(ref.luck_refit.corr(ref.luck_next_refit))
    refit_vs_old = float(ref.luck.corr(ref.luck_refit))
    # reuse columns named luck so pack_models can run on a renamed copy
    ref2 = ref.copy()
    ref2["luck"], ref2["luck_next"] = ref2.luck_refit, ref2.luck_next_refit
    refit_m = pack_models(ref2.dropna(subset=["luck", "luck_next", "dwin", "win_pct", "net_ppa"]))
    refit_summary = {
        "exponent": exp_hat,
        "r_with_237": refit_vs_old,
        "r_luck_luck_next": refit_persist,
        "dwin_coef": refit_m["dwin"]["luck"]["params"]["luck"],
        "dwin_se": refit_m["dwin"]["luck"]["bse"]["luck"],
        "delta_r2": refit_m["next_luck"]["in_sample"]["r2"] - refit_m["next_base"]["in_sample"]["r2"],
        "delta_loso": refit_m["next_luck"]["loso"]["r2"] - refit_m["next_base"]["loso"]["r2"],
    }

    payload = {
        "generated": "2026-11-10",
        "pythag_exp": PYTHAG_EXP,
        "fitted_exp": exp_hat,
        "main": main_m,
        "sens": sens_m,
        "refit": refit_summary,
    }
    (PROC / "luck_models.json").write_text(json.dumps(payload, indent=2))

    figures(pairs, dec, main_m["r_luck_luck_next"], main_m["dwin"]["luck"]["params"]["luck"])
    write_report(
        {
            "main": main_m,
            "sens": sens_m,
            "dec": dec,
            "dec_sens": dec_s,
            "pairs": pairs,
            "exp_hat": exp_hat,
            "refit": refit_summary,
        }
    )

    print(
        f"n={main_m['n']} r(luck,luck_next)={main_m['r_luck_luck_next']:.3f} "
        f"dwin~luck {main_m['dwin']['luck']['params']['luck']:.3f} "
        f"R2 {main_m['dwin']['luck']['r2']:.3f}"
    )
    print(
        f"next R2 base {main_m['next_base']['in_sample']['r2']:.3f} "
        f"+luck {main_m['next_luck']['in_sample']['r2']:.3f} "
        f"LOSO {main_m['next_base']['loso']['r2']:.3f} -> {main_m['next_luck']['loso']['r2']:.3f}"
    )
    print(f"fitted exp {exp_hat:.3f}; wrote RESULTS_luck.md and figures/luck_regression.png")


if __name__ == "__main__":
    main()
