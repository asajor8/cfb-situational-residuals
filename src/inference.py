"""Stage 3: formal inference and a horse race.

Nested F-tests, team-clustered bootstrap CIs for incremental R^2 (in-sample and
LOSO), a joint next-season model with cluster-robust SEs, VIF, offense/defense
splits, a 2019->2020 sensitivity, and bootstrap CIs on the original persistence
correlations. Uses only stage-1/2 files and team_season_combined.csv. No API calls.
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
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DERIVED, DOCS, FIGURES, PROC, ensure_dirs  # noqa: E402
from wins import COVID_YEAR, fmt_r  # noqa: E402

ensure_dirs()
FIG = FIGURES
PAIRS = PROC / "wins_next_pairs.csv"
COMBINED = DERIVED / "team_season_combined.csv"
N_BOOT = 2000
SEED = 20261110
Y = "win_pct_next"

# Nested comparisons: name, restricted cols, full cols, added column, label
COMPARISONS = [
    ("ppa_vs_wl", ["win_pct"], ["win_pct", "net_ppa"], "net_ppa", "+ net PPA | this-season win%"),
    ("resid_vs_wl", ["win_pct"], ["win_pct", "net_residual"], "net_residual", "+ net residual | this-season win%"),
    ("luck_vs_ppa", ["win_pct", "net_ppa"], ["win_pct", "net_ppa", "luck"], "luck", "+ luck | win% + net PPA"),
    ("resid_vs_ppa", ["win_pct", "net_ppa"], ["win_pct", "net_ppa", "net_residual"], "net_residual", "+ net residual | win% + net PPA"),
    ("resid_vs_ppa_luck", ["win_pct", "net_ppa", "luck"], ["win_pct", "net_ppa", "net_residual", "luck"], "net_residual", "+ net residual | win% + net PPA + luck"),
]

JOINT_COLS = ["win_pct", "net_ppa", "net_residual", "luck"]
SPLIT_COLS = ["win_pct", "mean_ppa", "d_def_ppa", "residual", "d_residual", "luck"]


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def fmt_p_eq(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "p = —"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def fmt_ci(lo: float, hi: float, digits: int = 3) -> str:
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def add_const(arr: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(arr)), arr])


def ols_beta_r2(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2


def loso_r2(y: np.ndarray, X: np.ndarray, years: np.ndarray) -> float:
    n = len(y)
    yhat = np.zeros(n)
    ynull = np.zeros(n)
    used = np.zeros(n, dtype=bool)
    for yr in np.unique(years):
        test = years == yr
        train = ~test
        if int(train.sum()) < 10 or int(test.sum()) < 5:
            continue
        beta, _ = ols_beta_r2(y[train], X[train])
        yhat[test] = X[test] @ beta
        ynull[test] = y[train].mean()
        used[test] = True
    if not used.any():
        return float("nan")
    sse = float(np.sum((y[used] - yhat[used]) ** 2))
    sst = float(np.sum((y[used] - ynull[used]) ** 2))
    return 1.0 - sse / sst if sst > 0 else float("nan")


def percentile_ci(draws: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    d = draws[np.isfinite(draws)]
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def boot_p_two_sided(draws: np.ndarray, null: float = 0.0) -> float:
    """Two-sided bootstrap p-value from percentile draws about `null`."""
    d = draws[np.isfinite(draws)]
    if len(d) == 0:
        return float("nan")
    # Davison/Hinkley-style: twice the tail probability on the side of the null
    p = 2 * min((d <= null).mean(), (d >= null).mean())
    return float(min(1.0, p))


def team_index(teams: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    uniq, inv = np.unique(teams, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(uniq))]
    return uniq, groups


def draw_cluster(groups: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    pick = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[i] for i in pick])


def sm_ols(df: pd.DataFrame, cols: list[str], cluster: bool = False):
    X = sm.add_constant(df[cols], has_constant="add")
    model = sm.OLS(df[Y], X)
    if cluster:
        return model.fit(cov_type="cluster", cov_kwds={"groups": df["team"]})
    return model.fit()


def nested_tests(df: pd.DataFrame, rest_cols: list[str], full_cols: list[str], added: str) -> dict:
    rest = sm_ols(df, rest_cols, cluster=False)
    full = sm_ols(df, full_cols, cluster=False)
    full_cl = sm_ols(df, full_cols, cluster=True)
    an = anova_lm(rest, full)
    fstat = float(an.iloc[1]["F"])
    fp = float(an.iloc[1]["Pr(>F)"])
    # cluster-robust Wald on the added coefficient
    wald_z = float(full_cl.params[added] / full_cl.bse[added])
    # two-sided normal p
    from math import erfc, sqrt

    wald_p = float(erfc(abs(wald_z) / sqrt(2)))
    y = df[Y].to_numpy()
    years = df.year.to_numpy()
    Xr = add_const(df[rest_cols].to_numpy())
    Xf = add_const(df[full_cols].to_numpy())
    return {
        "n": int(len(df)),
        "r2_rest": float(rest.rsquared),
        "r2_full": float(full.rsquared),
        "delta_r2": float(full.rsquared - rest.rsquared),
        "loso_rest": loso_r2(y, Xr, years),
        "loso_full": loso_r2(y, Xf, years),
        "loso_delta": loso_r2(y, Xf, years) - loso_r2(y, Xr, years),
        "f_stat": fstat,
        "f_p": fp,
        "added_coef": float(full_cl.params[added]),
        "added_se_cluster": float(full_cl.bse[added]),
        "added_ci95_cluster": [float(full_cl.conf_int().loc[added, 0]), float(full_cl.conf_int().loc[added, 1])],
        "wald_z": wald_z,
        "wald_p": wald_p,
    }


def bootstrap_comparisons(df: pd.DataFrame, comps: list, n_boot: int, seed: int) -> dict:
    """Cluster-by-team bootstrap of in-sample and LOSO ΔR², plus added coef."""
    y = df[Y].to_numpy()
    years = df.year.to_numpy()
    _, groups = team_index(df.team.to_numpy())
    mats = {}
    added_col_idx = {}  # index in full matrix (including intercept)
    for name, rest, full, added, _ in comps:
        mats[name] = (
            add_const(df[rest].to_numpy()),
            add_const(df[full].to_numpy()),
        )
        added_col_idx[name] = 1 + full.index(added)
    rng = np.random.default_rng(seed)
    out = {name: {"delta": np.empty(n_boot), "loso_delta": np.empty(n_boot), "coef": np.empty(n_boot)} for name, *_ in comps}
    for b in range(n_boot):
        idx = draw_cluster(groups, rng)
        yb, yrb = y[idx], years[idx]
        for name, *_ in comps:
            Xr, Xf = mats[name]
            _, r_rest = ols_beta_r2(yb, Xr[idx])
            beta_f, r_full = ols_beta_r2(yb, Xf[idx])
            out[name]["delta"][b] = r_full - r_rest
            out[name]["loso_delta"][b] = loso_r2(yb, Xf[idx], yrb) - loso_r2(yb, Xr[idx], yrb)
            out[name]["coef"][b] = beta_f[added_col_idx[name]]
    summary = {}
    for name, *_ in comps:
        d, ld, c = out[name]["delta"], out[name]["loso_delta"], out[name]["coef"]
        d_lo, d_hi = percentile_ci(d)
        ld_lo, ld_hi = percentile_ci(ld)
        c_lo, c_hi = percentile_ci(c)
        summary[name] = {
            "delta_ci95": [d_lo, d_hi],
            "delta_p_boot": boot_p_two_sided(d, 0.0),
            "loso_delta_ci95": [ld_lo, ld_hi],
            "loso_delta_p_boot": boot_p_two_sided(ld, 0.0),
            "coef_ci95_boot": [c_lo, c_hi],
            "coef_p_boot": boot_p_two_sided(c, 0.0),
            "n_boot": n_boot,
            "n_valid_loso": int(np.isfinite(ld).sum()),
        }
    return summary


def joint_model(df: pd.DataFrame, cols: list[str]) -> dict:
    fit = sm_ols(df, cols, cluster=True)
    X = sm.add_constant(df[cols], has_constant="add")
    vifs = {col: float(variance_inflation_factor(X.to_numpy(), i)) for i, col in enumerate(X.columns)}
    sdy = float(df[Y].std(ddof=1))
    rows = {}
    for c in cols:
        sdx = float(df[c].std(ddof=1))
        coef = float(fit.params[c])
        se = float(fit.bse[c])
        ci = fit.conf_int().loc[c]
        rows[c] = {
            "coef": coef,
            "se_cluster": se,
            "ci95_cluster": [float(ci[0]), float(ci[1])],
            "p_cluster": float(fit.pvalues[c]),
            "std_beta": coef * sdx / sdy,
            "pp_per_sd": 100 * coef * sdx,
            "pp_per_sd_ci95": [100 * float(ci[0]) * sdx, 100 * float(ci[1]) * sdx],
            "sd_x": sdx,
        }
    return {
        "n": int(len(df)),
        "r2": float(fit.rsquared),
        "r2_adj": float(fit.rsquared_adj),
        "vif": vifs,
        "intercept": float(fit.params["const"]),
        "intercept_se": float(fit.bse["const"]),
        "terms": rows,
        "corr": df[cols].corr().round(4).to_dict(),
    }


def block_f_test(df: pd.DataFrame, rest_cols: list[str], full_cols: list[str]) -> dict:
    rest = sm_ols(df, rest_cols, cluster=False)
    full = sm_ols(df, full_cols, cluster=False)
    an = anova_lm(rest, full)
    return {
        "r2_rest": float(rest.rsquared),
        "r2_full": float(full.rsquared),
        "delta_r2": float(full.rsquared - rest.rsquared),
        "f_stat": float(an.iloc[1]["F"]),
        "f_p": float(an.iloc[1]["Pr(>F)"]),
        "df_num": float(an.iloc[1]["df_diff"]),
        "n": int(len(df)),
    }


def build_persist_pairs(ts: pd.DataFrame, flag: str, cols: list[str], adjacent_only: bool = True) -> pd.DataFrame:
    years = [2019, 2021, 2022, 2023, 2024, 2025]
    out = []
    for y0, y1 in zip(years[:-1], years[1:]):
        gap = y1 - y0 > 1
        if adjacent_only and gap:
            continue
        a = ts[(ts.year == y0) & (ts[flag] == True)][["team"] + cols]  # noqa: E712
        b = ts[(ts.year == y1) & (ts[flag] == True)][["team"] + cols].rename(
            columns={c: c + "_next" for c in cols}
        )
        m = a.merge(b, on="team")
        m["year"] = y0
        m["next_year"] = y1
        m["gap"] = gap
        out.append(m)
    return pd.concat(out, ignore_index=True)


def persist_bootstrap(pairs: pd.DataFrame, specs: list[tuple[str, str, str]], n_boot: int, seed: int) -> dict:
    """Cluster-by-team bootstrap of Pearson r and selected differences."""
    _, groups = team_index(pairs.team.to_numpy())
    arrays = {name: (pairs[a].to_numpy(), pairs[b].to_numpy()) for name, a, b in specs}
    names = [s[0] for s in specs]
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(n_boot) for name in names}
    for b in range(n_boot):
        idx = draw_cluster(groups, rng)
        for name in names:
            x, y = arrays[name]
            draws[name][b] = float(np.corrcoef(x[idx], y[idx])[0, 1])
    point = {name: float(np.corrcoef(*arrays[name])[0, 1]) for name in names}
    out = {}
    for name in names:
        lo, hi = percentile_ci(draws[name])
        out[name] = {"r": point[name], "ci95": [lo, hi], "n": int(len(pairs))}
    # differences of interest
    diffs = []
    if "off_raw" in point and "off_resid" in point:
        diffs.append(("raw_minus_resid", "off_raw", "off_resid"))
    if "off_ppa" in point and "off_resid" in point:
        diffs.append(("ppa_minus_resid", "off_ppa", "off_resid"))
    if "def_ppa" in point and "def_resid" in point:
        diffs.append(("def_ppa_minus_resid", "def_ppa", "def_resid"))
    for dname, a, b in diffs:
        delta = draws[a] - draws[b]
        lo, hi = percentile_ci(delta)
        out[dname] = {
            "diff": point[a] - point[b],
            "ci95": [lo, hi],
            "p_boot": boot_p_two_sided(delta, 0.0),
        }
    return out


def run_suite(df: pd.DataFrame, seed: int) -> dict:
    comps = {}
    for name, rest, full, added, label in COMPARISONS:
        d = nested_tests(df, rest, full, added)
        d["label"] = label
        d["added"] = added
        comps[name] = d
    boot = bootstrap_comparisons(df, COMPARISONS, N_BOOT, seed)
    for name in comps:
        comps[name]["boot"] = boot[name]
    joint = joint_model(df, JOINT_COLS)
    split = joint_model(df, SPLIT_COLS)
    blocks = {
        "resid_block_given_ppa": block_f_test(
            df, ["win_pct", "mean_ppa", "d_def_ppa"], ["win_pct", "mean_ppa", "d_def_ppa", "residual", "d_residual"]
        ),
        "resid_block_given_ppa_luck": block_f_test(
            df,
            ["win_pct", "mean_ppa", "d_def_ppa", "luck"],
            ["win_pct", "mean_ppa", "d_def_ppa", "residual", "d_residual", "luck"],
        ),
        "off_resid_given_off_ppa": block_f_test(df, ["win_pct", "mean_ppa"], ["win_pct", "mean_ppa", "residual"]),
        "def_resid_given_def_ppa": block_f_test(df, ["win_pct", "d_def_ppa"], ["win_pct", "d_def_ppa", "d_residual"]),
        "ppa_block_given_resid": block_f_test(
            df, ["win_pct", "residual", "d_residual"], ["win_pct", "mean_ppa", "d_def_ppa", "residual", "d_residual"]
        ),
    }
    return {"n": int(len(df)), "n_teams": int(df.team.nunique()), "comparisons": comps, "joint": joint, "split": split, "blocks": blocks}


LABELS_X = {
    "win_pct": "This-season win%",
    "net_ppa": "Net PPA",
    "net_residual": "Net residual",
    "luck": "Luck",
}


def figures(main: dict) -> None:
    plt.rcParams["text.usetex"] = False
    plt.rcParams["text.parse_math"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4))

    # Left: LOSO ΔR² with bootstrap CIs; in-sample ΔR² as open markers
    order = ["ppa_vs_wl", "resid_vs_wl", "luck_vs_ppa", "resid_vs_ppa"]
    short = {
        "ppa_vs_wl": "+ net PPA\n| win%",
        "resid_vs_wl": "+ net residual\n| win%",
        "luck_vs_ppa": "+ luck\n| win% + PPA",
        "resid_vs_ppa": "+ net residual\n| win% + PPA",
    }
    ax = axes[0]
    ys = np.arange(len(order))[::-1]
    for i, name in enumerate(order):
        y = ys[i]
        c = main["comparisons"][name]
        lo, hi = c["boot"]["loso_delta_ci95"]
        mid = c["loso_delta"]
        ax.plot([lo, hi], [y, y], color="#154733", lw=2, zorder=2)
        ax.scatter([mid], [y], s=55, c="#154733", zorder=3, label="LOSO ΔR²" if i == 0 else None)
        ax.scatter(
            [c["delta_r2"]],
            [y + 0.18],
            s=32,
            facecolors="none",
            edgecolors="#555",
            linewidths=1.1,
            zorder=3,
            label="In-sample ΔR²" if i == 0 else None,
        )
    ax.axvline(0, c="k", lw=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([short[n] for n in order], fontsize=9)
    ax.set_xlabel("Incremental R²  (bars: 95% team-clustered bootstrap CI for LOSO ΔR²)")
    ax.set_title("Do the increments hold up?")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_xlim(-0.01, 0.055)

    # Right: joint-model standardized effects (pp of next win% per 1 SD)
    ax = axes[1]
    terms = ["win_pct", "net_ppa", "net_residual", "luck"]
    ys = np.arange(len(terms))[::-1]
    for i, t in enumerate(terms):
        y = ys[i]
        row = main["joint"]["terms"][t]
        lo, hi = row["pp_per_sd_ci95"]
        mid = row["pp_per_sd"]
        color = "#154733" if (lo > 0 or hi < 0) else "#888888"
        ax.plot([lo, hi], [y, y], color=color, lw=2, zorder=2)
        ax.scatter([mid], [y], s=55, c=color, zorder=3)
        ax.annotate(
            f"{mid:+.1f}",
            (mid, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=color,
        )
    ax.axvline(0, c="k", lw=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABELS_X[t] for t in terms], fontsize=9)
    ax.set_xlabel("pp of next-season win% per 1 SD  (95% cluster-robust CI)")
    ax.set_title("Joint model horse race")

    fig.suptitle(
        f"Formal inference on next-season win%\n"
        f"Team-clustered SEs and {N_BOOT:,}-rep team-clustered bootstrap  (n={main['n']} pairs, {main['n_teams']} teams)\n"
        "Data: CollegeFootballData.com",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG / "inference_ci.png", dpi=150)
    plt.close(fig)


def comp_row(c: dict, label: str) -> str:
    b = c["boot"]
    return (
        f"| {label} | {c['n']} | {fmt_r(c['r2_rest'], 3)} → {fmt_r(c['r2_full'], 3)} | "
        f"{c['delta_r2']:+.3f} | {fmt_ci(*b['delta_ci95'])} | {c['f_stat']:.2f} | {fmt_p(c['f_p'])} | "
        f"{c['loso_delta']:+.3f} | {fmt_ci(*b['loso_delta_ci95'])} | {fmt_p(b['loso_delta_p_boot'])} |"
    )


def joint_row(name: str, row: dict) -> str:
    return (
        f"| {name} | {row['coef']:.3f} | {row['se_cluster']:.3f} | "
        f"{fmt_ci(*row['ci95_cluster'])} | {fmt_p(row['p_cluster'])} | "
        f"{row['std_beta']:+.3f} | {row['pp_per_sd']:+.1f} | {fmt_ci(*row['pp_per_sd_ci95'], 1)} |"
    )


def write_report(ctx: dict) -> None:
    main, sens, persist = ctx["main"], ctx["sens"], ctx["persist"]
    c = main["comparisons"]
    j = main["joint"]
    sp = main["split"]
    bl = main["blocks"]
    cs = sens["comparisons"]
    js = sens["joint"]

    # verdict pieces
    resid_given = c["resid_vs_ppa"]
    resid_full = c["resid_vs_ppa_luck"]
    ppa_term = j["terms"]["net_ppa"]
    resid_term = j["terms"]["net_residual"]
    luck_term = j["terms"]["luck"]
    win_term = j["terms"]["win_pct"]

    resid_holds = resid_given["f_p"] < 0.05 and resid_term["p_cluster"] < 0.05
    ppa_joint_sig = ppa_term["p_cluster"] < 0.05
    luck_sig = luck_term["p_cluster"] < 0.05

    if resid_holds and not ppa_joint_sig:
        horse = (
            f"Yes — the net residual still adds something once net PPA is in the model "
            f"(ΔR² {resid_given['delta_r2']:+.3f}, nested F {fmt_p_eq(resid_given['f_p'])}; "
            f"in the joint model the residual is {resid_term['pp_per_sd']:+.1f} pp of next-season win% per SD, "
            f"cluster {fmt_p_eq(resid_term['p_cluster'])}). Net PPA itself is no longer distinguishable from zero "
            f"in that joint model (cluster {fmt_p_eq(ppa_term['p_cluster'])}), but that is largely collinearity "
            f"with this year's win% (VIF ≈ {j['vif']['net_ppa']:.1f}), not a claim that PPA is useless: it clearly "
            f"adds when it is the one extra term on top of W-L ({fmt_p_eq(c['ppa_vs_wl']['f_p'])})."
        )
    elif resid_holds and ppa_joint_sig:
        horse = (
            f"Both unique increments survive in the joint model. The residual still adds after PPA "
            f"(ΔR² {resid_given['delta_r2']:+.3f}, F {fmt_p_eq(resid_given['f_p'])}), and net PPA remains "
            f"detectable as well (cluster {fmt_p_eq(ppa_term['p_cluster'])})."
        )
    else:
        horse = (
            f"Once net PPA is already in the model, the residual's extra increment is "
            f"ΔR² {resid_given['delta_r2']:+.3f} (F {fmt_p_eq(resid_given['f_p'])}; "
            f"joint-model cluster {fmt_p_eq(resid_term['p_cluster'])})."
        )

    loso_notes = []
    for key, lab in [
        ("ppa_vs_wl", "PPA over W-L"),
        ("resid_vs_wl", "residual over W-L"),
        ("luck_vs_ppa", "luck over W-L + PPA"),
        ("resid_vs_ppa", "residual over W-L + PPA"),
    ]:
        lo, hi = c[key]["boot"]["loso_delta_ci95"]
        pboot = c[key]["boot"]["loso_delta_p_boot"]
        excl = "excludes 0" if (hi < 0 or lo > 0) else "includes 0"
        loso_notes.append(f"{lab} LOSO ΔR² {c[key]['loso_delta']:+.3f}, 95% CI {fmt_ci(lo, hi)}, {excl} (bootstrap {fmt_p_eq(pboot)})")

    vif_win, vif_ppa, vif_res, vif_luck = (
        j["vif"]["win_pct"],
        j["vif"]["net_ppa"],
        j["vif"]["net_residual"],
        j["vif"]["luck"],
    )

    # persist lines
    def pr(name):
        d = persist[name]
        return f"{d['r']:.2f} {fmt_ci(*d['ci95'], 2)}"

    off_gap = persist["ppa_minus_resid"]
    raw_gap = persist["raw_minus_resid"]
    def_gap = persist["def_ppa_minus_resid"]

    split_rows = [
        joint_row("This-season win%", sp["terms"]["win_pct"]),
        joint_row("Offense PPA", sp["terms"]["mean_ppa"]),
        joint_row("Defense PPA", sp["terms"]["d_def_ppa"]),
        joint_row("Offense residual", sp["terms"]["residual"]),
        joint_row("Defense residual", sp["terms"]["d_residual"]),
        joint_row("Luck", sp["terms"]["luck"]),
    ]

    md = f"""# Formal inference and a horse race — Stage 3

Generated 2026-11-10 from `data/processed/wins_next_pairs.csv` and `team_season_combined.csv`. Next-season sample is the same {main['n']} adjacent team-pairs as stages 1–2 (2019→2020 and 2021–2025, {main['n_teams']} teams). Inference uses classical nested F-tests, cluster-robust (by team) standard errors, and a {N_BOOT:,}-replication bootstrap that resamples teams with replacement and keeps each team's seasons together (seed {SEED}). Data source: CollegeFootballData.com (CFBD). Script: `src/inference.py`.

## Punchline

The stage-1 and stage-2 increments are real, not just in-sample noise. Adding net PPA to this season's win% raises in-sample R² by **{c['ppa_vs_wl']['delta_r2']:+.3f}** (F = {c['ppa_vs_wl']['f_stat']:.1f}, {fmt_p_eq(c['ppa_vs_wl']['f_p'])}); adding the net residual instead raises it by **{c['resid_vs_wl']['delta_r2']:+.3f}** (F = {c['resid_vs_wl']['f_stat']:.1f}, {fmt_p_eq(c['resid_vs_wl']['f_p'])}). Adding luck on top of win% + net PPA raises it by **{c['luck_vs_ppa']['delta_r2']:+.3f}** (F = {c['luck_vs_ppa']['f_stat']:.1f}, {fmt_p_eq(c['luck_vs_ppa']['f_p'])}). The two efficiency-over-W-L increments have leave-one-season-out 95% CIs that exclude zero. Luck's LOSO CI just includes zero ({fmt_ci(*c['luck_vs_ppa']['boot']['loso_delta_ci95'])}, bootstrap {fmt_p_eq(c['luck_vs_ppa']['boot']['loso_delta_p_boot'])}), so the F-test is stronger than the out-of-sample interval.

**Does the residual still add anything once PPA is in?** {horse} The unique increment is small (LOSO ΔR² {c['resid_vs_ppa']['loso_delta']:+.3f}), and its LOSO bootstrap CI includes zero {fmt_ci(*c['resid_vs_ppa']['boot']['loso_delta_ci95'])}. So the residual's *unique* next-season contribution is statistically detectable in the in-sample tests and the joint-model coefficient, but it is not a large or out-of-sample-decisive increment.

Luck stays negative and significant in the joint model ({luck_term['pp_per_sd']:+.1f} pp per SD, cluster {fmt_p_eq(luck_term['p_cluster'])}): teams that outperformed Pythagorean this year win less next year, holding record and efficiency fixed. Dropping 2019→2020 leaves the F-tests and the joint model in the same place (the PPA-over-W-L LOSO CI becomes a little wider and just includes zero). Persistence correlations, with a team-clustered bootstrap on the same adjacent pairs as the original note, confirm residual < raw success and residual < PPA: the paired differences exclude zero even though the separate r intervals overlap.

## 1. (a) Nested tests and bootstrap CIs for ΔR²

In-sample R² always rises when a regressor is added, so a bootstrap confidence interval for in-sample ΔR² is a statement about the *size* of the increment, not a test that it is zero. The tests of "does this add signal?" are the nested F, the cluster-robust Wald test on the added coefficient, and the LOSO ΔR² (which can be negative) with its bootstrap CI. Bootstrap p-values below are two-sided, from the LOSO ΔR² draws (a fairer null than in-sample ΔR²).

| Increment | n | R² restricted → full | In-sample ΔR² | Boot 95% CI (in-sample) | F | F p | LOSO ΔR² | Boot 95% CI (LOSO) | LOSO boot p |
|---|---|---|---|---|---|---|---|---|---|
{comp_row(c['ppa_vs_wl'], '+ net PPA | win%')}
{comp_row(c['resid_vs_wl'], '+ net residual | win%')}
{comp_row(c['luck_vs_ppa'], '+ luck | win% + net PPA')}
{comp_row(c['resid_vs_ppa'], '+ net residual | win% + net PPA')}
{comp_row(c['resid_vs_ppa_luck'], '+ net residual | win% + net PPA + luck')}

Cluster-robust Wald on the added coefficient (same models): net PPA given win% z = {c['ppa_vs_wl']['wald_z']:.2f}, {fmt_p_eq(c['ppa_vs_wl']['wald_p'])}; net residual given win% z = {c['resid_vs_wl']['wald_z']:.2f}, {fmt_p_eq(c['resid_vs_wl']['wald_p'])}; luck given win% + PPA z = {c['luck_vs_ppa']['wald_z']:.2f}, {fmt_p_eq(c['luck_vs_ppa']['wald_p'])}; residual given win% + PPA z = {c['resid_vs_ppa']['wald_z']:.2f}, {fmt_p_eq(c['resid_vs_ppa']['wald_p'])}.

In plain language: {'; '.join(loso_notes)}. The residual's *unique* increment on top of PPA is smaller (in-sample ΔR² {c['resid_vs_ppa']['delta_r2']:+.3f}) but the F-test rejects zero ({fmt_p_eq(c['resid_vs_ppa']['f_p'])}), and it still rejects after luck is included too (ΔR² {c['resid_vs_ppa_luck']['delta_r2']:+.3f}, {fmt_p_eq(c['resid_vs_ppa_luck']['f_p'])}).

## 2. (b) Joint model and the horse race

`win%_{{t+1}} ~ win%_t + net PPA_t + net residual_t + luck_t`, with standard errors clustered by team. Standardized columns are ordinary betas (SD of next win% per SD of the predictor) and a more readable scale: percentage points of next-season win% per one-SD bump in the predictor, holding the others fixed. Cluster-robust 95% CIs.

| Predictor | Coef | Cluster SE | 95% CI | Cluster p | Std beta | pp / 1 SD | 95% CI (pp / SD) |
|---|---|---|---|---|---|---|---|
{joint_row('This-season win%', win_term)}
{joint_row('Net PPA', ppa_term)}
{joint_row('Net residual', resid_term)}
{joint_row('Luck', luck_term)}

Joint-model R² = {fmt_r(j['r2'], 3)} (n = {j['n']}). A one-SD better residual is associated with {resid_term['pp_per_sd']:+.1f} percentage points of next-season win% after you already know the record, net PPA, and luck (95% CI {fmt_ci(*resid_term['pp_per_sd_ci95'], 1)}). A one-SD luck surplus is associated with {luck_term['pp_per_sd']:+.1f} pp (95% CI {fmt_ci(*luck_term['pp_per_sd_ci95'], 1)}). Net PPA's unique piece in this specification is {ppa_term['pp_per_sd']:+.1f} pp and the CI crosses zero.

**Collinearity.** VIFs: this-season win% {vif_win:.2f}, net PPA {vif_ppa:.2f}, net residual {vif_res:.2f}, luck {vif_luck:.2f}. Win% and net PPA share a lot of variance (they correlated at r = 0.84 in stage 1), so their *separate* joint-model coefficients should be read carefully — VIF near 6 is elevated, not catastrophic. The residual (VIF {vif_res:.1f}) and luck (VIF {vif_luck:.1f}) are in a healthier range. That is why PPA can look "insignificant" in the kitchen-sink model even though it clearly improves a win%-only regression: most of what PPA knows is already in the record (and, once luck is in, in the point totals). The residual is less tied to this year's W-L, so more of its next-season signal is unique.

### Offense / defense split

Same joint idea, but net PPA and net residual are split into offense and defense pieces.

| Predictor | Coef | Cluster SE | 95% CI | Cluster p | Std beta | pp / 1 SD | 95% CI (pp / SD) |
|---|---|---|---|---|---|---|---|
{chr(10).join(split_rows)}

Split-model R² = {fmt_r(sp['r2'], 3)}. In this fuller specification the unique residual signal is clearer on **defense** (cluster {fmt_p_eq(sp['terms']['d_residual']['p_cluster'])}, {sp['terms']['d_residual']['pp_per_sd']:+.1f} pp per SD) than on offense (cluster {fmt_p_eq(sp['terms']['residual']['p_cluster'])}), where offense residual and offense PPA already correlate at r = 0.82. That split should not be over-read: the simpler one-at-a-time horse races do not reject (offense residual given offense PPA + win%, {fmt_p_eq(bl['off_resid_given_off_ppa']['f_p'])}; defense residual given defense PPA + win%, {fmt_p_eq(bl['def_resid_given_def_ppa']['f_p'])}). A joint F-test that *both* residuals are zero given both PPAs and win% does reject (ΔR² {bl['resid_block_given_ppa']['delta_r2']:+.3f}, F = {bl['resid_block_given_ppa']['f_stat']:.2f}, {fmt_p_eq(bl['resid_block_given_ppa']['f_p'])}), but it is only marginal once luck is included ({fmt_p_eq(bl['resid_block_given_ppa_luck']['f_p'])}). The two PPA terms as a block, given the two residuals and win%, are also only marginal ({fmt_p_eq(bl['ppa_block_given_resid']['f_p'])}).

VIFs in the split model stay in the same band (win% {sp['vif']['win_pct']:.2f}, offense PPA {sp['vif']['mean_ppa']:.2f}, defense PPA {sp['vif']['d_def_ppa']:.2f}, offense residual {sp['vif']['residual']:.2f}, defense residual {sp['vif']['d_residual']:.2f}, luck {sp['vif']['luck']:.2f}).

## 3. (c) Sensitivity: drop 2019→2020

n = {sens['n']} pairs, {sens['n_teams']} teams. F-tests and the joint-model residual / luck coefficients are essentially unchanged. The PPA-over-W-L LOSO interval is a little weaker and just includes zero.

| Increment | In-sample ΔR² | F p | LOSO ΔR² | LOSO boot 95% CI | LOSO boot p |
|---|---|---|---|---|---|
| + net PPA | win% | {cs['ppa_vs_wl']['delta_r2']:+.3f} | {fmt_p(cs['ppa_vs_wl']['f_p'])} | {cs['ppa_vs_wl']['loso_delta']:+.3f} | {fmt_ci(*cs['ppa_vs_wl']['boot']['loso_delta_ci95'])} | {fmt_p(cs['ppa_vs_wl']['boot']['loso_delta_p_boot'])} |
| + net residual | win% | {cs['resid_vs_wl']['delta_r2']:+.3f} | {fmt_p(cs['resid_vs_wl']['f_p'])} | {cs['resid_vs_wl']['loso_delta']:+.3f} | {fmt_ci(*cs['resid_vs_wl']['boot']['loso_delta_ci95'])} | {fmt_p(cs['resid_vs_wl']['boot']['loso_delta_p_boot'])} |
| + luck | win% + PPA | {cs['luck_vs_ppa']['delta_r2']:+.3f} | {fmt_p(cs['luck_vs_ppa']['f_p'])} | {cs['luck_vs_ppa']['loso_delta']:+.3f} | {fmt_ci(*cs['luck_vs_ppa']['boot']['loso_delta_ci95'])} | {fmt_p(cs['luck_vs_ppa']['boot']['loso_delta_p_boot'])} |
| + net residual | win% + PPA | {cs['resid_vs_ppa']['delta_r2']:+.3f} | {fmt_p(cs['resid_vs_ppa']['f_p'])} | {cs['resid_vs_ppa']['loso_delta']:+.3f} | {fmt_ci(*cs['resid_vs_ppa']['boot']['loso_delta_ci95'])} | {fmt_p(cs['resid_vs_ppa']['boot']['loso_delta_p_boot'])} |

Joint-model cluster p: win% {fmt_p(js['terms']['win_pct']['p_cluster'])}, net PPA {fmt_p(js['terms']['net_ppa']['p_cluster'])}, net residual {fmt_p(js['terms']['net_residual']['p_cluster'])}, luck {fmt_p(js['terms']['luck']['p_cluster'])}. Residual pp / SD {js['terms']['net_residual']['pp_per_sd']:+.1f} (95% CI {fmt_ci(*js['terms']['net_residual']['pp_per_sd_ci95'], 1)}).

## 4. (d) Persistence correlations, with uncertainty

Rebuilt from `team_season_combined.csv` on the same rule as the original persistence note: adjacent pairs only (2021→2022 through 2024→2025), ≥300 modeling snaps in both seasons of a pair. That recovers n = {persist['off_resid']['n']} offense pairs and n = {persist['def_resid']['n']} defense pairs, and the same point estimates (offense residual 0.39, raw success 0.47, PPA 0.46; defense residual 0.32, defensive PPA 0.44). 95% CIs are from a team-clustered bootstrap ({N_BOOT:,} reps). Differences use the same draws, so they respect the dependence of two correlations measured on the same teams.

| Series | r | 95% CI |
|---|---|---|
| Offense residual | {pr('off_resid')} |
| Offense raw success | {pr('off_raw')} |
| Offense mean PPA | {pr('off_ppa')} |
| Defense residual | {pr('def_resid')} |
| Defensive PPA | {pr('def_ppa')} |

Raw success minus residual: {raw_gap['diff']:+.2f}, 95% CI {fmt_ci(*raw_gap['ci95'], 2)}, bootstrap {fmt_p_eq(raw_gap['p_boot'])}. Offense PPA minus residual: {off_gap['diff']:+.2f}, 95% CI {fmt_ci(*off_gap['ci95'], 2)}, bootstrap {fmt_p_eq(off_gap['p_boot'])}. Defensive PPA minus defense residual: {def_gap['diff']:+.2f}, 95% CI {fmt_ci(*def_gap['ci95'], 2)}, bootstrap {fmt_p_eq(def_gap['p_boot'])}.

The original ranking is supported. Residuals persist, but less than raw success or PPA. The separate r intervals overlap — they usually will — but the paired differences, which are the right comparison, exclude zero: raw success minus residual +0.08 ({fmt_p_eq(raw_gap['p_boot'])}), offense PPA minus residual +0.07 ({fmt_p_eq(off_gap['p_boot'])}), defensive PPA minus defense residual +0.12 ({fmt_p_eq(def_gap['p_boot'])}). The defense gap is the clearest of the three.

## 5. Caveats

- In-sample ΔR² cannot be negative, so its bootstrap CI is not a hypothesis test of zero. Use the F / Wald tests and the LOSO ΔR² CI for that.
- Team-clustered SEs and a team-clustered bootstrap allow any one program's seasons to be dependent. They do not fix unmodeled coaching changes, portal turnover, or schedule.
- VIF near 6 for win% and net PPA means the joint-model horse race between those two should be read as "little unique PPA leftover after the record (and luck)," not "PPA does not predict."
- The offense/defense split gives the residual more moving parts and less precision, especially on offense. The defense-residual finding is interesting but is one split of one sample.
- Persistence CIs use four adjacent year-pairs and a COVID-skipped 2019→2021 gap that is still left out of the pooled figure, matching the earlier writeup.
- {N_BOOT:,} bootstrap replications, seed {SEED}. Percentile intervals; no BCa adjustment.

## 6. Options for what to do next

1. **Fold stages 1–3 into the writeup and the GitHub repo.** The canvas and `asajor8/cfb-situational-residuals` still describe the original residual project. This is the natural packaging step: add the wins / luck / inference results, figures, and a short methods note, without expanding the question further.
2. **Point-margin target.** The unused stage-2 option: does this year's efficiency beat this year's margin for predicting next year's *margin* (not just win%)? Formal tests can follow the same F / bootstrap pattern.
3. **Close-game luck vs turnover.** Split one-score games out of Pythagorean luck, and split team-pairs by coaching or roster continuity, so famous drops like TCU 2022–23 are not counted as pure luck regression.

## Files

- `src/inference.py`
- `data/processed/inference_models.json`
- `figures/inference_ci.png`
- `docs/RESULTS_inference.md`
"""
    (DOCS / "RESULTS_inference.md").write_text(md)


def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    pairs = pd.read_csv(PAIRS)
    ts = pd.read_csv(COMBINED)
    print(f"running main suite n={len(pairs)} ({N_BOOT} boots)…", flush=True)
    main_s = run_suite(pairs, SEED)
    sens_df = pairs[pairs.next_year != COVID_YEAR].reset_index(drop=True)
    print(f"running sensitivity n={len(sens_df)}…", flush=True)
    sens_s = run_suite(sens_df, SEED + 1)

    off = build_persist_pairs(ts, "qualified", ["residual", "raw_success", "mean_ppa"], adjacent_only=True)
    deff = build_persist_pairs(ts, "d_qualified", ["d_residual", "d_def_ppa"], adjacent_only=True)
    # merge offense+defense persistence on team-year for a shared team cluster when possible;
    # bootstrap each side on its own pair file so n matches the original note.
    print("bootstrapping persistence correlations…", flush=True)
    persist_off = persist_bootstrap(
        off,
        [("off_resid", "residual", "residual_next"), ("off_raw", "raw_success", "raw_success_next"), ("off_ppa", "mean_ppa", "mean_ppa_next")],
        N_BOOT,
        SEED + 2,
    )
    persist_def = persist_bootstrap(
        deff,
        [("def_resid", "d_residual", "d_residual_next"), ("def_ppa", "d_def_ppa", "d_def_ppa_next")],
        N_BOOT,
        SEED + 3,
    )
    persist = {**persist_off, **persist_def}

    payload = {
        "generated": "2026-11-10",
        "n_boot": N_BOOT,
        "seed": SEED,
        "main": main_s,
        "sens": sens_s,
        "persist": persist,
    }
    (PROC / "inference_models.json").write_text(json.dumps(_jsonify(payload), indent=2))
    figures(main_s)
    write_report({"main": main_s, "sens": sens_s, "persist": persist})

    def brief(name, s):
        d = s["comparisons"][name]
        lo, hi = d["boot"]["loso_delta_ci95"]
        return f"{name} dR2 {d['delta_r2']:+.3f} F p={d['f_p']:.4g} LOSO {d['loso_delta']:+.3f} CI [{lo:.3f},{hi:.3f}]"

    print(" | ".join(brief(k, main_s) for k in ["ppa_vs_wl", "resid_vs_wl", "luck_vs_ppa", "resid_vs_ppa"]))
    jt = main_s["joint"]["terms"]
    print(
        "joint p: win {w:.4g} ppa {p:.4g} resid {r:.4g} luck {l:.4g}".format(
            w=jt["win_pct"]["p_cluster"], p=jt["net_ppa"]["p_cluster"], r=jt["net_residual"]["p_cluster"], l=jt["luck"]["p_cluster"]
        )
    )
    print(
        "persist r resid {a:.3f} {b}  ppa {c:.3f} {d}".format(
            a=persist["off_resid"]["r"],
            b=persist["off_resid"]["ci95"],
            c=persist["off_ppa"]["r"],
            d=persist["off_ppa"]["ci95"],
        )
    )
    print("wrote RESULTS_inference.md and figures/inference_ci.png")


if __name__ == "__main__":
    main()
