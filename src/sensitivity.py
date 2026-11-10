"""Spec sensitivity: does the choice of expected-success formula change team-season residuals?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.formula.api as smf
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROC
s = pd.read_parquet(PROC / "offense_snaps.parquet")
m = s[s.use_in_model].copy(); m["down"] = m.down.astype(int)
m["ld"] = np.log1p(m.distance.clip(lower=0))
specs = {"additive": "success ~ C(down) + distance + yardsToGoal + score_diff",
         "interaction": "success ~ C(down) * distance + yardsToGoal + score_diff",
         "flex_logdist": "success ~ C(down) * (distance + ld) + yardsToGoal + score_diff"}
out = {}
for k, f in specs.items():
    r = smf.logit(f, data=m).fit(disp=0); m["e_" + k] = r.predict(m)
    sit = m.assign(b=np.where(m.down >= 3, np.where(m.distance == 1, "3/4&1", np.where(m.distance <= 3, "3/4&2-3", np.where(m.distance <= 6, "3/4&4-6", "3/4&7+"))), "1/2"))
    c = sit.groupby(["down", "b"]).apply(lambda g: pd.Series({"n": len(g), "err": g.success.mean() - g["e_" + k].mean()}))
    print(k, "AIC", round(r.aic), "3rd&1 actual-pred", round(c.loc[(3, "3/4&1"), "err"], 4))
    out[k] = m.groupby(["year", "off_school"]).apply(lambda g: pd.Series({"n": len(g), "resid": (g.success - g["e_" + k]).mean()}))
t = pd.concat({k: v.resid for k, v in out.items()}, axis=1); t["n"] = out["additive"].n
q = t[t.n >= 300]
print(q[list(specs)].corr().round(4))
print("max abs team residual diff add vs interaction", (q.additive - q.interaction).abs().max().round(4),
      " add vs flex", (q.additive - q.flex_logdist).abs().max().round(4))
o = q.xs("Oregon", level=1) if "Oregon" in q.index.get_level_values(1) else None
ranks = q.groupby(level=0).rank(ascending=False)
print(pd.concat([o, ranks.xs("Oregon", level=1).add_suffix("_rank")], axis=1).round(4))
