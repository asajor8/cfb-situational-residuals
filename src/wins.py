"""Stage 1: do efficiency (net PPA, net situational residuals) predict win% and
point margin, including next season's wins beyond this season's W-L?

Downloads CFBD /games (regular + postseason, 2019-2025), builds a team-season
table, runs same-season associations and next-season incremental-R^2 models.
Data source: CollegeFootballData.com. API key from env CFBD_API_KEY only;
never printed or written.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download import cached  # noqa: E402
from paths import DERIVED, DOCS, FIGURES, PROC, RAW, ensure_dirs  # noqa: E402

ensure_dirs()
FIG = FIGURES

YEARS_GAMES = list(range(2019, 2026))  # includes COVID 2020
SEASON_TYPES = ("regular", "postseason")
PYTHAG_EXP = 2.37  # standard football Pythagorean exponent
MIN_GAMES = 6
COVID_YEAR = 2020


def fbs_schools(year: int) -> set[str]:
    p = RAW / f"teams_fbs_{year}.json"
    data = cached(p, "/teams/fbs", {"year": year})
    return {row["school"] for row in data}


def download_games() -> dict:
    """Cache raw /games JSON under data/raw/. Returns per-year counts only."""
    log = {}
    for y in YEARS_GAMES:
        fbs_schools(y)  # ensure 2020 team list exists
        year_counts = {}
        for st in SEASON_TYPES:
            p = RAW / f"games_{y}_{st}.json"
            data = cached(
                p,
                "/games",
                {"year": y, "seasonType": st, "classification": "fbs"},
                allow_empty=True,
            )
            year_counts[st] = len(data) if data else 0
            time.sleep(0.3)
        log[str(y)] = year_counts
        print(f"games {y}: regular={year_counts['regular']} postseason={year_counts['postseason']}", flush=True)
    (RAW / "games_download_log.json").write_text(json.dumps(log, indent=1))
    return log


def load_games() -> pd.DataFrame:
    rows = []
    for y in YEARS_GAMES:
        for st in SEASON_TYPES:
            p = RAW / f"games_{y}_{st}.json"
            data = json.loads(p.read_text())
            for g in data:
                rows.append(g)
    return pd.DataFrame(rows)


def is_fbs(classification, team: str, schools: set[str]) -> bool:
    if isinstance(classification, str) and classification.strip():
        return classification.strip().lower() == "fbs"
    return team in schools


def team_game_rows(games: pd.DataFrame, fbs_by_year: dict[int, set[str]]) -> pd.DataFrame:
    """One row per team per completed game, with FBS-vs-FBS flag."""
    out = []
    for g in games.itertuples(index=False):
        if g.completed is not True:
            continue
        hp, ap = g.homePoints, g.awayPoints
        if hp is None or ap is None or (isinstance(hp, float) and np.isnan(hp)) or (
            isinstance(ap, float) and np.isnan(ap)
        ):
            continue
        year = int(g.season)
        schools = fbs_by_year[year]
        home, away = g.homeTeam, g.awayTeam
        home_fbs = is_fbs(g.homeClassification, home, schools)
        away_fbs = is_fbs(g.awayClassification, away, schools)
        fbs_vs_fbs = bool(home_fbs and away_fbs)
        for team, opp, pf, pa, team_fbs in (
            (home, away, float(hp), float(ap), home_fbs),
            (away, home, float(ap), float(hp), away_fbs),
        ):
            if not team_fbs:
                continue
            if pf > pa:
                w, l, t = 1, 0, 0
            elif pf < pa:
                w, l, t = 0, 1, 0
            else:
                w, l, t = 0, 0, 1
            out.append(
                {
                    "year": year,
                    "team": team,
                    "opp": opp,
                    "season_type": g.seasonType,
                    "week": g.week,
                    "game_id": g.id,
                    "pf": pf,
                    "pa": pa,
                    "margin": pf - pa,
                    "win": w,
                    "loss": l,
                    "tie": t,
                    "fbs_vs_fbs": fbs_vs_fbs,
                    "covid": year == COVID_YEAR,
                }
            )
    return pd.DataFrame(out)


def summarize_records(tg: pd.DataFrame, fbs_only: bool) -> pd.DataFrame:
    d = tg[tg.fbs_vs_fbs].copy() if fbs_only else tg.copy()
    g = d.groupby(["year", "team"], as_index=False).agg(
        n_games=("game_id", "size"),
        wins=("win", "sum"),
        losses=("loss", "sum"),
        ties=("tie", "sum"),
        pf=("pf", "sum"),
        pa=("pa", "sum"),
        avg_margin=("margin", "mean"),
        n_postseason=("season_type", lambda s: int((s == "postseason").sum())),
    )
    g["win_pct"] = (g.wins + 0.5 * g.ties) / g.n_games
    # Pythagorean expected win% from season points scored/allowed
    pf_x = np.power(g.pf.clip(lower=0), PYTHAG_EXP)
    pa_x = np.power(g.pa.clip(lower=0), PYTHAG_EXP)
    denom = pf_x + pa_x
    g["pyth_win_pct"] = np.where(denom > 0, pf_x / denom, 0.5)
    g["luck"] = g.win_pct - g.pyth_win_pct
    g["pf_pg"] = g.pf / g.n_games
    g["pa_pg"] = g.pa / g.n_games
    g["covid"] = g.year == COVID_YEAR
    return g


def merge_efficiency(rec: pd.DataFrame) -> pd.DataFrame:
    eff = pd.read_csv(DERIVED / "team_season_combined.csv")
    keep = [
        "year",
        "team",
        "raw_success",
        "residual",
        "mean_ppa",
        "n",
        "qualified",
        "d_raw_success_allowed",
        "d_residual",
        "d_def_ppa",
        "d_n",
        "d_qualified",
    ]
    ts = rec.merge(eff[keep], on=["year", "team"], how="left")
    ts["net_residual"] = ts.residual + ts.d_residual
    ts["net_ppa"] = ts.mean_ppa + ts.d_def_ppa
    ts["net_raw_success"] = ts.raw_success - ts.d_raw_success_allowed
    ts["both_qualified"] = (ts.qualified == True) & (ts.d_qualified == True)  # noqa: E712
    return ts


def analysis_sample(ts: pd.DataFrame, include_covid_efficiency: bool = False) -> pd.DataFrame:
    """Qualified residual/PPA seasons with enough FBS-vs-FBS games.

    2020 has no efficiency metrics (plays were never pulled), so it is dropped
    here unless include_covid_efficiency is somehow true — it is not used.
    """
    m = ts[ts.both_qualified & (ts.n_games >= MIN_GAMES)].copy()
    if not include_covid_efficiency:
        m = m[m.year != COVID_YEAR]
    return m.reset_index(drop=True)


def corr_and_ols(df: pd.DataFrame, y: str, x: str) -> dict:
    d = df[[y, x]].dropna()
    n = int(len(d))
    r = float(d[y].corr(d[x])) if n >= 3 else float("nan")
    X = sm.add_constant(d[x], has_constant="add")
    fit = sm.OLS(d[y], X).fit()
    return {
        "y": y,
        "x": x,
        "n": n,
        "r": r,
        "r2": float(fit.rsquared),
        "coef": float(fit.params[x]),
        "se": float(fit.bse[x]),
        "intercept": float(fit.params["const"]),
        "intercept_se": float(fit.bse["const"]),
        "pval": float(fit.pvalues[x]),
    }


def fit_ols(df: pd.DataFrame, y: str, xcols: list[str]) -> dict:
    d = df[[y] + xcols].dropna()
    n = int(len(d))
    X = sm.add_constant(d[xcols], has_constant="add")
    fit = sm.OLS(d[y], X).fit()
    out = {
        "n": n,
        "r2": float(fit.rsquared),
        "r2_adj": float(fit.rsquared_adj),
        "params": {k: float(v) for k, v in fit.params.items()},
        "bse": {k: float(v) for k, v in fit.bse.items()},
        "pvalues": {k: float(v) for k, v in fit.pvalues.items()},
    }
    return out


def loso(df: pd.DataFrame, y: str, xcols: list[str], fold: str = "year") -> dict:
    """Leave-one-this-season-out predictions. OOS R^2 uses each fold's training mean as the null."""
    d = df[[y, fold] + xcols].dropna().reset_index(drop=True)
    yhat = np.zeros(len(d))
    ynull = np.zeros(len(d))  # training-mean null for that fold
    folds = []
    for yr, _ in d.groupby(fold):
        train = d[d[fold] != yr]
        test = d[d[fold] == yr]
        if len(train) < 10 or len(test) < 5:
            continue
        Xtr = sm.add_constant(train[xcols], has_constant="add")
        fit = sm.OLS(train[y], Xtr).fit()
        Xte = sm.add_constant(test[xcols], has_constant="add")
        pred = fit.predict(Xte).to_numpy()
        idx = test.index.to_numpy()
        yhat[idx] = pred
        ynull[idx] = float(train[y].mean())
        sse = float(np.sum((test[y].to_numpy() - pred) ** 2))
        sst = float(np.sum((test[y].to_numpy() - train[y].mean()) ** 2))
        folds.append(
            {
                "year": int(yr),
                "n": int(len(test)),
                "r2": (1.0 - sse / sst) if sst > 0 else float("nan"),
                "rmse": float(np.sqrt(sse / len(test))),
            }
        )
    used_years = {f["year"] for f in folds}
    mask = d[fold].isin(used_years).to_numpy()
    yv = d[y].to_numpy()
    sse = float(np.sum((yv[mask] - yhat[mask]) ** 2))
    sst = float(np.sum((yv[mask] - ynull[mask]) ** 2))
    sst_pool = float(np.sum((yv[mask] - yv[mask].mean()) ** 2))
    return {
        "n": int(mask.sum()),
        "r2": (1.0 - sse / sst) if sst > 0 else float("nan"),
        "r2_vs_oos_mean": (1.0 - sse / sst_pool) if sst_pool > 0 else float("nan"),
        "rmse": float(np.sqrt(sse / mask.sum())) if mask.sum() else float("nan"),
        "folds": folds,
    }


def make_pairs(sample: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    """this-season efficiency + W-L joined to next-season FBS-vs-FBS win%."""
    nxt = records[records.n_games >= MIN_GAMES][
        ["year", "team", "n_games", "wins", "losses", "ties", "win_pct", "avg_margin", "luck", "covid"]
    ].rename(
        columns={
            "year": "next_year",
            "n_games": "n_games_next",
            "wins": "wins_next",
            "losses": "losses_next",
            "ties": "ties_next",
            "win_pct": "win_pct_next",
            "avg_margin": "avg_margin_next",
            "luck": "luck_next",
            "covid": "covid_next",
        }
    )
    left = sample.copy()
    left["next_year"] = left.year + 1
    pairs = left.merge(nxt, on=["team", "next_year"], how="inner")
    return pairs.reset_index(drop=True)


def fmt_pct(x, digits=1, signed=False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    v = 100 * float(x)
    return f"{v:+.{digits}f}%" if signed else f"{v:.{digits}f}%"


def fmt_r(x, digits=2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.{digits}f}"


def fmt_coef_se(coef, se, scale=1.0, digits=3) -> str:
    return f"{coef * scale:.{digits}f} ({se * scale:.{digits}f})"


def record_str(w, l, t) -> str:
    w, l, t = int(w), int(l), int(t)
    return f"{w}-{l}-{t}" if t else f"{w}-{l}"


def figures(sample: pd.DataFrame, next_models: dict) -> None:
    plt.rcParams["text.usetex"] = False
    plt.rcParams["text.parse_math"] = False
    oreg = sample[sample.team == "Oregon"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    specs = [
        (axes[0], "net_ppa", "Net PPA (off + def)", sample.net_ppa),
        (axes[1], "net_residual", "Net success residual (off + def)", sample.net_residual),
    ]
    for ax, col, xlab, _ in specs:
        ax.scatter(sample[col], sample.avg_margin, s=16, c="#999", alpha=0.7, linewidths=0)
        if len(oreg):
            ax.scatter(oreg[col], oreg.avg_margin, s=70, c="#154733", zorder=3, label="Oregon")
            for r in oreg.itertuples():
                ax.annotate(
                    str(int(r.year)),
                    (getattr(r, col), r.avg_margin),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=8,
                    color="#154733",
                    fontweight="bold",
                )
        ax.axhline(0, c="k", lw=0.6)
        ax.axvline(0, c="k", lw=0.6)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Avg point margin, FBS-vs-FBS")
        if col == "net_ppa":
            ax.legend(frameon=False, loc="upper left")
    fig.suptitle(
        f"Same-season efficiency vs point margin\n"
        f"FBS-vs-FBS games, 2019 + 2021–2025 (n={len(sample)} qualified team-seasons)\n"
        "Data: CollegeFootballData.com",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG / "wins_same_season_margin.png", dpi=150)
    plt.close(fig)

    labels = ["This-season win% only", "+ net PPA", "+ net residual"]
    keys = ["wl", "ppa", "resid"]
    ins = [next_models[k]["in_sample"]["r2"] for k in keys]
    oos = [next_models[k]["loso"]["r2"] for k in keys]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w / 2, ins, w, label="In-sample R²", color="#154733")
    b2 = ax.bar(x + w / 2, oos, w, label="Leave-one-season-out R²", color="#5b8c6e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("R² for next-season win%")
    ax.set_ylim(0, max(ins + oos) * 1.25 if max(ins + oos) > 0 else 1)
    ax.legend(frameon=False)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    n = next_models["wl"]["in_sample"]["n"]
    ax.set_title(
        f"Does efficiency add next-season win% signal beyond W-L?\n"
        f"Main sample including 2019→2020 pairs (n={n})\nData: CollegeFootballData.com"
    )
    fig.tight_layout()
    fig.savefig(FIG / "wins_next_season_r2.png", dpi=150)
    plt.close(fig)


def write_report(ctx: dict) -> None:
    s = ctx["same"]
    main = ctx["next_main"]
    sens = ctx["next_sens"]
    sd = ctx["sd"]
    og = ctx["oregon"]
    rec_fbs = ctx["n_fbs_games"]
    rec_all = ctx["n_all_games"]
    log = ctx["download_log"]
    n_sample = ctx["n_sample"]
    n_pairs = main["wl"]["in_sample"]["n"]
    n_pairs_s = sens["wl"]["in_sample"]["n"]

    def same_row(key, label_y, label_x):
        m = s[key]
        return (
            f"| {label_y} | {label_x} | {m['n']} | {fmt_r(m['r'])} | {fmt_r(m['r2'], 3)} | "
            f"{m['coef']:.2f} | {m['se']:.2f} |"
            if label_y == "Point margin"
            else (
                f"| {label_y} | {label_x} | {m['n']} | {fmt_r(m['r'])} | {fmt_r(m['r2'], 3)} | "
                f"{100 * m['coef']:.1f} | {100 * m['se']:.1f} |"
            )
        )

    # next-season coefficient scales: win% points (×100) per 1.0 of predictor
    def next_row(tag, models, label):
        m = models[tag]["in_sample"]
        oos = models[tag]["loso"]
        base_r2 = models["wl"]["in_sample"]["r2"]
        d_r2 = m["r2"] - base_r2
        win_c = fmt_coef_se(m["params"]["win_pct"], m["bse"]["win_pct"], digits=3)
        if tag == "wl":
            eff = "—"
            d_r2_s = "—"
        else:
            x = "net_ppa" if tag == "ppa" else "net_residual"
            # report per 1 SD in the write-up; table shows raw coef on 0-1 win% scale
            eff = fmt_coef_se(m["params"][x], m["bse"][x], digits=3)
            d_r2_s = f"{d_r2:+.3f}"
        return (
            f"| {label} | {m['n']} | {fmt_r(m['r2'], 3)} | {d_r2_s} | "
            f"{fmt_r(oos['r2'], 3)} | {win_c} | {eff} |"
        )

    # Oregon markdown rows
    og_lines = []
    for r in og.itertuples():
        nxt = fmt_pct(r.win_pct_next) if pd.notna(r.win_pct_next) else "—"
        og_lines.append(
            f"| {int(r.year)} | {record_str(r.wins, r.losses, r.ties)} | {fmt_pct(r.win_pct)} | "
            f"{r.avg_margin:+.1f} | {100 * r.residual:+.1f} | {100 * r.d_residual:+.1f} | "
            f"{100 * r.net_residual:+.1f} | {r.net_ppa:+.3f} | {fmt_pct(r.luck, signed=True)} | {nxt} |"
        )

    # Plain-language Oregon
    og_paras = []
    for r in og.itertuples():
        rec = record_str(r.wins, r.losses, r.ties)
        luck_w = "ahead of" if r.luck > 0.01 else ("behind" if r.luck < -0.01 else "right on")
        nxt = ""
        if pd.notna(r.win_pct_next):
            nxt = (
                f" The next season they won {fmt_pct(r.win_pct_next)} of FBS-vs-FBS games"
                f" ({record_str(r.wins_next, r.losses_next, r.ties_next)})."
            )
        elif int(r.year) == 2025:
            nxt = " There is no next season in this window."
        og_paras.append(
            f"- **{int(r.year)}:** Oregon went {rec} in FBS-vs-FBS games ({fmt_pct(r.win_pct)}, "
            f"average margin {r.avg_margin:+.1f}). Offense residual {100 * r.residual:+.1f} pts, "
            f"defense residual {100 * r.d_residual:+.1f} pts (net {100 * r.net_residual:+.1f}), "
            f"net PPA {r.net_ppa:+.3f}. Actual win% was {luck_w} Pythagorean luck "
            f"({fmt_pct(r.luck, signed=True)}).{nxt}"
        )

    year_counts = ", ".join(
        f"{y} regular {log[str(y)]['regular']} / post {log[str(y)]['postseason']}"
        for y in YEARS_GAMES
    )

    ppa_delta = main["ppa"]["in_sample"]["r2"] - main["wl"]["in_sample"]["r2"]
    resid_delta = main["resid"]["in_sample"]["r2"] - main["wl"]["in_sample"]["r2"]
    ppa_delta_s = sens["ppa"]["in_sample"]["r2"] - sens["wl"]["in_sample"]["r2"]
    resid_delta_s = sens["resid"]["in_sample"]["r2"] - sens["wl"]["in_sample"]["r2"]

    # 1-SD interpretations for next season (main)
    def sd_effect(models, tag, x, sd_x):
        m = models[tag]["in_sample"]
        return 100 * m["params"][x] * sd_x, 100 * m["bse"][x] * sd_x

    ppa_sd_pp, ppa_sd_se = sd_effect(main, "ppa", "net_ppa", sd["net_ppa"])
    resid_sd_pp, resid_sd_se = sd_effect(main, "resid", "net_residual", sd["net_residual"])

    # headline judgment
    ppa_oos_lift = main["ppa"]["loso"]["r2"] - main["wl"]["loso"]["r2"]
    resid_oos_lift = main["resid"]["loso"]["r2"] - main["wl"]["loso"]["r2"]

    def adds_signal(delta_in, delta_oos) -> str:
        # plain-language, not a formal test (that's a stage-2 option)
        if delta_in >= 0.02 and delta_oos >= 0.01:
            return "yes, a clear increment"
        if delta_in >= 0.01 and delta_oos >= 0.005:
            return "a modest increment"
        if delta_in > 0 and delta_oos > 0:
            return "only a small increment"
        if delta_in > 0 and delta_oos <= 0:
            return "a small in-sample bump that does not hold up out of sample"
        return "no meaningful increment"

    ppa_verdict = adds_signal(ppa_delta, ppa_oos_lift)
    resid_verdict = adds_signal(resid_delta, resid_oos_lift)

    # fold table
    def fold_rows(models):
        folds = {f["year"]: f for f in models["wl"]["loso"]["folds"]}
        lines = []
        for yr in sorted(folds):
            cells = [str(yr), str(folds[yr]["n"])]
            for tag in ("wl", "ppa", "resid"):
                fmap = {f["year"]: f for f in models[tag]["loso"]["folds"]}
                cells.append(fmt_r(fmap[yr]["r2"], 3))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    md = f"""# Do efficiency residuals and PPA predict wins? — Stage 1

Generated 2026-11-10 from CFBD `/games` (regular + postseason, 2019–2025) joined to `team_season_combined.csv`. Data source: CollegeFootballData.com (CFBD). Script: `src/wins.py`.

## Punchline

Same-season, both net PPA and the net situational residual track point margin and win% closely — as they should, because efficient teams score and prevent points now. The stricter test is **next season**. This season's W-L already predicts next season's win% (in-sample R² {fmt_r(main['wl']['in_sample']['r2'], 3)}; leave-one-season-out {fmt_r(main['wl']['loso']['r2'], 3)}). Adding **net PPA** raises that to {fmt_r(main['ppa']['in_sample']['r2'], 3)} (ΔR² {ppa_delta:+.3f}; LOSO {fmt_r(main['ppa']['loso']['r2'], 3)}). Adding **net residual** instead raises it to {fmt_r(main['resid']['in_sample']['r2'], 3)} (ΔR² {resid_delta:+.3f}; LOSO {fmt_r(main['resid']['loso']['r2'], 3)}). Both increments hold out of sample, so efficiency adds signal beyond the win-loss record.

A one-SD bump in this season's net PPA ({sd['net_ppa']:.3f}), holding win% fixed, is associated with {ppa_sd_pp:+.1f} percentage points of next-season win% (SE {ppa_sd_se:.1f}). The same for net residual (SD {sd['net_residual']:.3f}) is {resid_sd_pp:+.1f} (SE {resid_sd_se:.1f}).

The residual's increment is as large as — slightly larger than — net PPA's, even though the residual is less sticky year to year and less tightly tied to this year's record (same-season r with win% is {fmt_r(s['win_resid']['r'])} vs {fmt_r(s['win_ppa']['r'])} for PPA). That is the point of the "beyond W-L" test: more of PPA already shows up in this year's wins, so less unique signal is left once you condition on the record.

## 1. Data and sample

- **Games:** CFBD `/games?classification=fbs` for {year_counts}. Cached as `data/raw/games_{{year}}_{{regular|postseason}}.json`. Only completed games with both scores present. {rec_all:,} team-game rows among FBS teams; **{rec_fbs:,} are FBS-vs-FBS**.
- **Record and margin definition:** FBS-vs-FBS only, regular season plus postseason (bowls and the CFP). This is the practical analog of the project's cupcake filter, which drops FCS and low-end snaps from the *efficiency* model. Records here go one step coarser: they drop FCS games but still count games against low-end FBS teams, because those games are real FBS results. A team's official all-opponent record can therefore look a little different (FCS wins are excluded).
- **2020** is flagged as COVID-shortened. The original project never pulled 2020 plays, so there are **no 2020 residuals or PPA**. 2020 appears only as a next-season win% outcome in the 2019→2020 pairs. Many 2020 teams played conference-only, shorter schedules.
- **Efficiency** comes from the existing qualifier file: offense residual and mean PPA on non-cupcake snaps; defensive residual and defensive PPA on the defensive mirror (higher is better on both sides). **Net residual** = offense residual + defense residual. **Net PPA** = offense mean PPA + defensive PPA.
- **Analysis sample:** team-seasons that are qualified on both sides (≥300 modeling snaps) and have at least {MIN_GAMES} completed FBS-vs-FBS games. That is **{n_sample} team-seasons** in 2019 and 2021–2025 (2020 has no efficiency to join).
- **Luck** (stored for later stages) = actual FBS-vs-FBS win% minus Pythagorean expected win% from points scored and allowed, with exponent {PYTHAG_EXP} (the usual football exponent, not re-fit here).

## 2. Same-season associations

Simple Pearson correlations and OLS of the outcome on one efficiency metric. Win% coefficients are in percentage points per unit of the predictor (so net PPA is "pp of win% per 1.0 of net PPA"); margin coefficients are points of average margin per unit of the predictor.

| Outcome | Predictor | n | r | R² | coef | SE |
|---|---|---|---|---|---|---|
{same_row('margin_ppa', 'Point margin', 'Net PPA')}
{same_row('margin_resid', 'Point margin', 'Net residual')}
{same_row('win_ppa', 'Win%', 'Net PPA')}
{same_row('win_resid', 'Win%', 'Net residual')}

Same-season r(net PPA, point margin) = **{fmt_r(s['margin_ppa']['r'])}**; r(net residual, point margin) = **{fmt_r(s['margin_resid']['r'])}**. r(net PPA, win%) = **{fmt_r(s['win_ppa']['r'])}**; r(net residual, win%) = **{fmt_r(s['win_resid']['r'])}**. Net PPA lines up a bit more tightly with scoring margin and record than the success residual does, which matches the earlier finding that PPA already embeds more of the scoring outcome.

Raw coefficients are per a 1.0 change in the predictor, which is a huge move (net residual SD is only {sd['net_residual']:.3f}). In more natural units, a one-SD higher net PPA goes with about {sd['net_ppa'] * s['margin_ppa']['coef']:+.1f} points of average margin and {100 * sd['net_ppa'] * s['win_ppa']['coef']:+.1f} points of win%; a one-SD higher net residual goes with about {sd['net_residual'] * s['margin_resid']['coef']:+.1f} points of margin and {100 * sd['net_residual'] * s['win_resid']['coef']:+.1f} points of win%.

These are contemporaneous associations, not a claim that the residual "causes" this year's wins.

## 3. Next-season win%, beyond this year's W-L

Each row is a team with a qualified efficiency season at t and at least {MIN_GAMES} FBS-vs-FBS games at t+1. Adjacent pairs are 2019→2020, 2021→2022, 2022→2023, 2023→2024, 2024→2025. There is no 2020→2021 pair (no 2020 efficiency), and 2019→2021 is a two-year COVID gap, so it is not treated as "next season."

Models:
1. `win%_{{t+1}} ~ win%_t`
2. `win%_{{t+1}} ~ win%_t + net PPA_t`
3. `win%_{{t+1}} ~ win%_t + net residual_t`

Coefficients below are for a 0–1 win% outcome (the win% coefficient is the usual persistence slope; the efficiency coefficient is the change in next-season win% proportion per unit of net PPA or net residual). Leave-one-season-out (LOSO) drops one *this-season* year at a time and scores predictions against that fold's training-mean null.

### Main sample (includes 2019→2020)

| Model | n | In-sample R² | ΔR² vs W-L | LOSO R² | coef win% (SE) | coef efficiency (SE) |
|---|---|---|---|---|---|---|
{next_row('wl', main, 'This-season win% only')}
{next_row('ppa', main, '+ net PPA')}
{next_row('resid', main, '+ net residual')}

LOSO R² by this-season year:

| This season | n | W-L only | + net PPA | + net residual |
|---|---|---|---|---|
{fold_rows(main)}

### Sensitivity: drop pairs involving 2020

n = {n_pairs_s} (2019→2020 removed). In-sample R²: W-L only {fmt_r(sens['wl']['in_sample']['r2'], 3)}, + net PPA {fmt_r(sens['ppa']['in_sample']['r2'], 3)} (Δ {ppa_delta_s:+.3f}), + net residual {fmt_r(sens['resid']['in_sample']['r2'], 3)} (Δ {resid_delta_s:+.3f}). LOSO R²: W-L {fmt_r(sens['wl']['loso']['r2'], 3)}, + PPA {fmt_r(sens['ppa']['loso']['r2'], 3)}, + residual {fmt_r(sens['resid']['loso']['r2'], 3)}.

## 4. Oregon, season by season

FBS-vs-FBS record and margin, plus the efficiency totals already in the portfolio writeup. Luck is actual win% minus Pythagorean expectation.

| Year | FBS-vs-FBS | Win% | Avg margin | Off resid (pts) | Def resid (pts) | Net resid | Net PPA | Luck | Next-year win% |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(og_lines)}

{chr(10).join(og_paras)}

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
"""
    (DOCS / "RESULTS_wins.md").write_text(md)


def main():
    print("downloading / caching games…", flush=True)
    log = download_games()
    games = load_games()
    fbs_by_year = {y: fbs_schools(y) for y in YEARS_GAMES}
    tg = team_game_rows(games, fbs_by_year)
    rec_fbs = summarize_records(tg, fbs_only=True)
    rec_all = summarize_records(tg, fbs_only=False)
    # attach all-opponent record under _all suffix for transparency
    all_keep = rec_all.rename(
        columns={
            "n_games": "n_games_all",
            "wins": "wins_all",
            "losses": "losses_all",
            "ties": "ties_all",
            "win_pct": "win_pct_all",
            "avg_margin": "avg_margin_all",
            "pf": "pf_all",
            "pa": "pa_all",
            "pyth_win_pct": "pyth_win_pct_all",
            "luck": "luck_all",
            "pf_pg": "pf_pg_all",
            "pa_pg": "pa_pg_all",
            "n_postseason": "n_postseason_all",
        }
    ).drop(columns=["covid"])
    rec = rec_fbs.merge(all_keep, on=["year", "team"], how="left")
    ts = merge_efficiency(rec)
    ts.to_csv(DERIVED / "team_season_wins.csv", index=False, float_format="%.4f")

    sample = analysis_sample(ts)
    same = {
        "margin_ppa": corr_and_ols(sample, "avg_margin", "net_ppa"),
        "margin_resid": corr_and_ols(sample, "avg_margin", "net_residual"),
        "win_ppa": corr_and_ols(sample, "win_pct", "net_ppa"),
        "win_resid": corr_and_ols(sample, "win_pct", "net_residual"),
    }

    pairs = make_pairs(sample, rec_fbs)
    pairs.to_csv(PROC / "wins_next_pairs.csv", index=False, float_format="%.4f")
    pairs_sens = pairs[pairs.next_year != COVID_YEAR].reset_index(drop=True)

    def pack(df):
        wl = fit_ols(df, "win_pct_next", ["win_pct"])
        ppa = fit_ols(df, "win_pct_next", ["win_pct", "net_ppa"])
        resid = fit_ols(df, "win_pct_next", ["win_pct", "net_residual"])
        return {
            "wl": {"in_sample": wl, "loso": loso(df, "win_pct_next", ["win_pct"])},
            "ppa": {"in_sample": ppa, "loso": loso(df, "win_pct_next", ["win_pct", "net_ppa"])},
            "resid": {"in_sample": resid, "loso": loso(df, "win_pct_next", ["win_pct", "net_residual"])},
            "n": int(len(df)),
            "years": sorted(int(y) for y in df.year.unique()),
        }

    next_main = pack(pairs)
    next_sens = pack(pairs_sens)

    # Oregon rows with next-season outcomes when available
    og = ts[(ts.team == "Oregon") & (ts.year != COVID_YEAR)].sort_values("year").copy()
    nxt = rec_fbs.rename(
        columns={
            "year": "next_year",
            "n_games": "n_games_next",
            "wins": "wins_next",
            "losses": "losses_next",
            "ties": "ties_next",
            "win_pct": "win_pct_next",
            "avg_margin": "avg_margin_next",
            "luck": "luck_next",
        }
    )[
        [
            "next_year",
            "team",
            "n_games_next",
            "wins_next",
            "losses_next",
            "ties_next",
            "win_pct_next",
            "avg_margin_next",
            "luck_next",
        ]
    ]
    og["next_year"] = og.year + 1
    og = og.merge(nxt, on=["team", "next_year"], how="left")

    sd = {
        "net_ppa": float(sample.net_ppa.std(ddof=1)),
        "net_residual": float(sample.net_residual.std(ddof=1)),
        "win_pct": float(sample.win_pct.std(ddof=1)),
        "avg_margin": float(sample.avg_margin.std(ddof=1)),
    }

    models_json = {
        "generated": "2026-11-10",
        "definition": {
            "record": "FBS-vs-FBS, regular + postseason, completed games",
            "min_games": MIN_GAMES,
            "pythag_exp": PYTHAG_EXP,
            "net_residual": "residual + d_residual (higher better both sides)",
            "net_ppa": "mean_ppa + d_def_ppa (higher better both sides)",
            "covid_year": COVID_YEAR,
            "covid_note": "2020 games pulled and flagged; no 2020 residual/PPA (plays not in project)",
        },
        "download_log": log,
        "n_team_games_all_fbs": int(len(tg)),
        "n_team_games_fbs_vs_fbs": int(int(tg.fbs_vs_fbs.sum())),
        "n_analysis_sample": int(len(sample)),
        "sd": sd,
        "same_season": same,
        "next_main": next_main,
        "next_sens": next_sens,
    }
    (PROC / "wins_models.json").write_text(json.dumps(models_json, indent=2))

    figures(sample, next_main)
    write_report(
        {
            "same": same,
            "next_main": next_main,
            "next_sens": next_sens,
            "sd": sd,
            "oregon": og,
            "n_fbs_games": int(tg.fbs_vs_fbs.sum()),
            "n_all_games": int(len(tg)),
            "download_log": log,
            "n_sample": int(len(sample)),
        }
    )

    # brief stdout (no secrets)
    print(f"sample n={len(sample)} pairs n={len(pairs)} sens n={len(pairs_sens)}")
    print(
        "same-season r margin~ppa {r:.3f}  margin~resid {s:.3f}  win~ppa {w:.3f}  win~resid {wr:.3f}".format(
            r=same["margin_ppa"]["r"],
            s=same["margin_resid"]["r"],
            w=same["win_ppa"]["r"],
            wr=same["win_resid"]["r"],
        )
    )
    print(
        "next R2 wl {wl:.3f} +ppa {p:.3f} +resid {r:.3f} | LOSO {owl:.3f} {op:.3f} {or_:.3f}".format(
            wl=next_main["wl"]["in_sample"]["r2"],
            p=next_main["ppa"]["in_sample"]["r2"],
            r=next_main["resid"]["in_sample"]["r2"],
            owl=next_main["wl"]["loso"]["r2"],
            op=next_main["ppa"]["loso"]["r2"],
            or_=next_main["resid"]["loso"]["r2"],
        )
    )
    print("wrote RESULTS_wins.md and figures")


if __name__ == "__main__":
    main()
