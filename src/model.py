"""Fit league-wide expected-success logistic model on modeling snaps; compute residuals and team-season summary.
Pooled, in-sample, descriptive. Data source: CollegeFootballData.com (CFBD)."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROC, DERIVED, ensure_dirs
ensure_dirs()
MIN_SNAPS = 300

def brier(y, p): return float(np.mean((y - p) ** 2))
def logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def situation_bucket(r):
    d, x = r.down, r.distance
    if d == 1: return "1st & 10" if x == 10 else ("1st & <10" if x < 10 else "1st & 11+")
    if d == 2: return "2nd & 1-3" if x <= 3 else ("2nd & 4-6" if x <= 6 else "2nd & 7+")
    lab = "3rd" if d == 3 else "4th"
    return f"{lab} & 1" if x == 1 else (f"{lab} & 2-3" if x <= 3 else (f"{lab} & 4-6" if x <= 6 else f"{lab} & 7+"))

def main():
    snaps = pd.read_parquet(PROC / "offense_snaps.parquet")
    m = snaps[snaps.use_in_model].dropna(subset=["down", "distance", "yardsToGoal", "score_diff"]).copy()
    dropped_na = int(snaps.use_in_model.sum() - len(m))
    m["down"] = m.down.astype(int)
    fits = {}
    for name, f in {"additive": "success ~ C(down) + distance + yardsToGoal + score_diff",
                    "interaction": "success ~ C(down) * distance + yardsToGoal + score_diff"}.items():
        res = smf.logit(f, data=m).fit(disp=0)
        p = res.predict(m)
        m["p_" + name] = p
        m["b_" + name] = m.apply(situation_bucket, axis=1) if "bucket" not in m else m["bucket"]
        m["bucket"] = m["b_" + name]
        cal = m.groupby("bucket").agg(n=("success", "size"), actual=("success", "mean"), pred=("p_" + name, "mean"))
        fits[name] = dict(res=res, aic=float(res.aic), bic=float(res.bic), brier=brier(m.success, p),
                          logloss=logloss(m.success, p), max_bucket_err=float((cal.actual - cal.pred).abs().max()),
                          wmae_bucket=float(((cal.actual - cal.pred).abs() * cal.n).sum() / cal.n.sum()))
    m = m.drop(columns=[c for c in m.columns if c.startswith("b_")])
    a, i = fits["additive"], fits["interaction"]
    # choose interaction only if it clearly improves bucket calibration (>=1pt weighted error drop)
    chosen = "interaction" if (a["wmae_bucket"] - i["wmae_bucket"]) >= 0.01 else "additive"
    res = fits[chosen]["res"]
    m["expected_success"] = m["p_" + chosen]; m["residual"] = m.success - m.expected_success

    # calibration tables for chosen model
    cal_down = m.groupby("down").agg(n=("success", "size"), actual=("success", "mean"), expected=("expected_success", "mean"))
    cal_sit = m.groupby("bucket").agg(n=("success", "size"), actual=("success", "mean"), expected=("expected_success", "mean"))
    m["ytg_bucket"] = pd.cut(m.yardsToGoal, [0, 10, 20, 40, 60, 80, 100], labels=["1-10", "11-20", "21-40", "41-60", "61-80", "81-99"])
    cal_ytg = m.groupby("ytg_bucket", observed=True).agg(n=("success", "size"), actual=("success", "mean"), expected=("expected_success", "mean"))
    m["sd_bucket"] = pd.cut(m.score_diff, [-200, -17, -9, -1, 0, 8, 16, 200], labels=["<=-17", "-16..-9", "-8..-1", "tied", "+1..+8", "+9..+16", ">=+17"])
    cal_sd = m.groupby("sd_bucket", observed=True).agg(n=("success", "size"), actual=("success", "mean"), expected=("expected_success", "mean"))
    m["p_decile"] = pd.qcut(m.expected_success, 10, labels=False, duplicates="drop")
    cal_dec = m.groupby("p_decile").agg(n=("success", "size"), actual=("success", "mean"), expected=("expected_success", "mean"))

    # attach predictions to all snaps (expected for cupcake snaps too, out-of-model application)
    snaps["down"] = snaps.down.astype(int)
    snaps["expected_success"] = res.predict(snaps)
    snaps["residual"] = snaps.success - snaps.expected_success
    snaps.loc[snaps.use_in_model, "expected_success"] = m["expected_success"]
    snaps.loc[snaps.use_in_model, "residual"] = m["residual"]
    snaps.to_parquet(PROC / "modeling_snaps.parquet", index=False)

    fbs = snaps[snaps.is_fbs_offense]
    bucket_all = fbs.groupby("opp_bucket").agg(n=("success", "size"), success=("success", "mean"),
        expected=("expected_success", "mean"), residual=("residual", "mean"), ppa=("ppa", "mean"))
    bucket_year = fbs.pivot_table(index="year", columns="opp_bucket", values="success", aggfunc="mean")

    raw = fbs.groupby(["year", "off_school"]).agg(n_all=("success", "size"), raw_success=("success", "mean"),
        raw_ppa=("ppa", "mean"), share_cupcake=("use_in_model", lambda s: 1 - s.mean()))
    mod = m.groupby(["year", "off_school"]).agg(n=("success", "size"), model_success=("success", "mean"),
        expected_success=("expected_success", "mean"), residual=("residual", "mean"), mean_ppa=("ppa", "mean"),
        share_power_snaps=("is_power_defense", "mean"), opp_prior_sp=("def_prior_sp", "mean"), conf=("offenseConference", lambda s: s.mode().iat[0]))
    ts = raw.join(mod, how="left").reset_index().rename(columns={"off_school": "team"})
    ts["qualified"] = ts.n >= MIN_SNAPS
    for col, name in [("raw_success", "rank_raw"), ("model_success", "rank_model"), ("residual", "rank_residual"), ("mean_ppa", "rank_ppa")]:
        ts[name] = ts[ts.qualified].groupby("year")[col].rank(ascending=False, method="min")
    ts["rank_change_raw_to_resid"] = ts.rank_raw - ts.rank_residual
    ts.to_csv(DERIVED / "team_season_summary.csv", index=False, float_format="%.4f")
    q = ts[ts.qualified]
    corr_by_year = q.groupby("year").apply(lambda g: g.residual.corr(g.mean_ppa)).to_dict()
    out = dict(chosen=chosen, dropped_na=dropped_na, n_model=len(m), params=res.params.to_dict(), bse=res.bse.to_dict(),
        pseudo_r2=float(res.prsquared),
        fits={k: {kk: vv for kk, vv in v.items() if kk != "res"} for k, v in fits.items()},
        corr_resid_ppa_all=float(q.residual.corr(q.mean_ppa)), corr_raw_ppa_all=float(q.raw_success.corr(q.raw_ppa)),
        corr_resid_ppa_by_year={int(k): float(v) for k, v in corr_by_year.items()},
        corr_raw_vs_resid=float(q.raw_success.corr(q.residual)),
        n_qualified_by_year=q.groupby("year").size().to_dict(),
        n_teamseasons_by_year=ts.groupby("year").size().to_dict(),
        model_overall=dict(n=len(m), success=float(m.success.mean()), ppa=float(m.ppa.mean())))
    (PROC / "model_summary.json").write_text(json.dumps(out, indent=1, default=float))
    for nm, t in [("cal_down", cal_down), ("cal_situation", cal_sit), ("cal_ytg", cal_ytg), ("cal_score", cal_sd),
                  ("cal_decile", cal_dec), ("opp_bucket", bucket_all), ("opp_bucket_by_year", bucket_year)]:
        t.to_csv(PROC / f"{nm}.csv", float_format="%.4f")
    print(res.summary()); print(json.dumps({k: v for k, v in out.items() if k not in ("params", "bse")}, indent=1, default=float))
    for t in [cal_down, cal_sit, cal_ytg, cal_sd, cal_dec, bucket_all, bucket_year]: print(t.round(4).to_string(), "\n")

if __name__ == "__main__":
    main()
