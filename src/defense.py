"""Defensive ratings from the same scrimmage snaps: defensive PPA (= -CFBD ppa) and success residual.
No new model: reuses CFBD ppa and the existing expected_success. Mirrors the offense cupcake filter,
but drops snaps against cupcake OFFENSES. Pooled / in-sample / descriptive. Data: CollegeFootballData.com."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean import team_maps, sp_prior, POWER_24, POWER_PRE24
from paths import PROC, RAW, DERIVED, ensure_dirs

ensure_dirs()
P = PROC
MIN_SNAPS = 300

def add_offense_flags(s):
    parts = []
    for y, g in s.groupby("year"):
        g = g.copy(); _, conf = team_maps(y); ratings, p25, _ = sp_prior(y)
        power = POWER_24 if y >= 2024 else POWER_PRE24
        g["off_conf_season"] = g.off_school.map(conf)
        g["is_power_offense"] = g.is_fbs_offense & (g.off_conf_season.isin(power) | (g.off_school == "Notre Dame"))
        g["off_prior_sp"] = g.off_school.map(ratings)
        g["is_low_end_offense"] = g.is_fbs_offense & ~g.is_power_offense & (g.off_prior_sp.isna() | (g.off_prior_sp < p25))
        parts.append(g)
    s = pd.concat(parts).sort_index()
    s["use_in_def_model"] = s.is_fbs_defense & s.is_fbs_offense & ~s.is_low_end_offense
    s["off_bucket"] = np.select([~s.is_fbs_offense, s.is_low_end_offense, s.is_power_offense],
                                ["FCS/non-FBS", "Low-end non-power", "Power"], default="Other FBS")
    s["defensive_ppa"] = -s.ppa
    s["defensive_success"] = 1 - s.success
    s["defensive_residual"] = s.expected_success - s.success
    return s

def advanced_check():
    """Optional sanity check vs CFBD season advanced defensive PPA (cached)."""
    from download import get, valid
    out = []
    for y in [2019, 2021, 2022, 2023, 2024, 2025]:
        p = RAW / f"adv_season_{y}.json"
        if valid(p):
            d = json.loads(p.read_text())
        else:
            d = get("/stats/season/advanced", {"year": y}) or []
            if d: p.write_text(json.dumps(d))
        for r in d:
            dp = (r.get("defense") or {}).get("ppa")
            if dp is not None:
                out.append({"year": y, "team": r["team"], "cfbd_def_ppa_allowed": dp})
    return pd.DataFrame(out)

def main():
    s = pd.read_parquet(P / "modeling_snaps.parquet")
    need = "is_low_end_offense" not in s.columns
    s = add_offense_flags(s)
    s.to_parquet(P / "modeling_snaps.parquet", index=False)
    fd = s[s.is_fbs_defense]; dm = s[s.use_in_def_model]
    bucket = fd.groupby("off_bucket").agg(n=("success", "size"), success_allowed=("success", "mean"),
        expected=("expected_success", "mean"), def_residual=("defensive_residual", "mean"), def_ppa=("defensive_ppa", "mean"))
    bucket_year = fd.pivot_table(index="year", columns="off_bucket", values="defensive_ppa", aggfunc="mean")
    bucket_year_sr = fd.pivot_table(index="year", columns="off_bucket", values="success", aggfunc="mean")
    raw = fd.groupby(["year", "def_school"]).agg(d_n_all=("success", "size"), d_raw_success_allowed=("success", "mean"),
        d_raw_def_ppa=("defensive_ppa", "mean"), d_share_cupcake=("use_in_def_model", lambda x: 1 - x.mean()))
    mod = dm.groupby(["year", "def_school"]).agg(d_n=("success", "size"), d_success_allowed=("success", "mean"),
        d_expected_faced=("expected_success", "mean"), d_residual=("defensive_residual", "mean"),
        d_def_ppa=("defensive_ppa", "mean"), d_share_power_opp=("is_power_offense", "mean"), d_opp_prior_sp=("off_prior_sp", "mean"))
    d = raw.join(mod, how="left").reset_index().rename(columns={"def_school": "team"})
    d["d_qualified"] = d.d_n >= MIN_SNAPS
    for col, name, asc in [("d_raw_success_allowed", "d_rank_raw_sr", True), ("d_success_allowed", "d_rank_sr", True),
                           ("d_residual", "d_rank_residual", False), ("d_def_ppa", "d_rank_ppa", False), ("d_raw_def_ppa", "d_rank_raw_ppa", False)]:
        d[name] = d[d.d_qualified].groupby("year")[col].rank(ascending=asc, method="min")
    d.to_csv(DERIVED / "team_season_defense.csv", index=False, float_format="%.4f")
    off = pd.read_csv(DERIVED / "team_season_summary.csv")
    comb = off.merge(d, on=["year", "team"], how="outer")
    comb.to_csv(DERIVED / "team_season_combined.csv", index=False, float_format="%.4f")
    q = d[d.d_qualified]
    summ = dict(added_flags=need, n_def_model=len(dm), n_off_model=int(s.use_in_model.sum()), n_fbs_def=len(fd),
        n_qualified_by_year=q.groupby("year").size().to_dict(),
        corr_resid_ppa=float(q.d_residual.corr(q.d_def_ppa)),
        corr_resid_ppa_by_year=q.groupby("year").apply(lambda g: g.d_residual.corr(g.d_def_ppa)).to_dict(),
        spearman_rawppa_vs_ppa=q.groupby("year").apply(lambda g: g.d_rank_raw_ppa.corr(g.d_rank_ppa, method="spearman")).to_dict(),
        spearman_rawsr_vs_resid=q.groupby("year").apply(lambda g: g.d_rank_raw_sr.corr(g.d_rank_residual, method="spearman")).to_dict(),
        def_ppa_null=int(dm.ppa.isna().sum()))
    try:
        adv = advanced_check()
        if len(adv):
            m = q.merge(adv, on=["year", "team"])
            summ["adv_check"] = dict(n=len(m), corr_mean_ppa_allowed_vs_cfbd=float((-m.d_raw_def_ppa).corr(m.cfbd_def_ppa_allowed)),
                                     corr_by_year=m.groupby("year").apply(lambda g: (-g.d_raw_def_ppa).corr(g.cfbd_def_ppa_allowed)).to_dict())
    except Exception as e:
        summ["adv_check"] = f"skipped: {type(e).__name__}"
    (P / "defense_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    bucket.to_csv(P / "def_opp_bucket.csv", float_format="%.4f")
    bucket_year.to_csv(P / "def_ppa_by_offbucket_year.csv", float_format="%.4f")
    bucket_year_sr.to_csv(P / "def_sr_allowed_by_offbucket_year.csv", float_format="%.4f")
    print(json.dumps(summ, indent=1, default=float)); print(bucket.round(4)); print(bucket_year.round(4)); print(bucket_year_sr.round(4))

if __name__ == "__main__":
    main()
