"""Build quote-ready tables (markdown) and the 2025 chart from processed outputs."""
import json, sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = False
plt.rcParams["text.parse_math"] = False
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROC, DERIVED, DOCS, FIGURES, ensure_dirs
ensure_dirs()
P = PROC
ts = pd.read_csv(DERIVED / "team_season_summary.csv"); q = ts[ts.qualified]
pct = lambda x: f"{100*x:.1f}%"; pts = lambda x: f"{100*x:+.1f}"
def row(r):
    return (f"| {r.year} | {r.team} | {pct(r.raw_success)} (#{int(r.rank_raw)}) | {pct(r.share_cupcake)} | {pct(r.model_success)} | "
            f"{pct(r.expected_success)} | {pts(r.residual)} (#{int(r.rank_residual)}) | {r.mean_ppa:.3f} (#{int(r.rank_ppa)}) | {r.opp_prior_sp:.1f} | {int(r.n)} |")
H = ("| Year | Team | Raw success, all FBS-offense snaps (rank) | Cupcake share | Modeling-sample success | Expected success | "
     "Residual, pts (rank) | Mean PPA (rank) | Opp prior SP+ | n model |\n|---|---|---|---|---|---|---|---|---|---|")
out = []
f = pd.read_csv(P / "funnel.csv")
out.append("## Funnel\n" + f.to_markdown(index=False))
b = pd.read_csv(P / "opp_bucket.csv"); out.append("## Opponent buckets (FBS offenses, all years)\n" + b.round(4).to_markdown(index=False))
out.append("## Opponent bucket success by year\n" + pd.read_csv(P / "opp_bucket_by_year.csv").round(4).to_markdown(index=False))
for nm in ["cal_down", "cal_situation", "cal_ytg", "cal_score", "cal_decile"]:
    out.append(f"## {nm}\n" + pd.read_csv(P / f"{nm}.csv").round(4).to_markdown(index=False))
out.append("## Oregon\n" + H + "\n" + "\n".join(row(r) for r in ts[ts.team == "Oregon"].itertuples()))
y = q[q.year == 2025]
out.append(f"## 2025 top 6 residual (qualifiers={len(y)})\n" + H + "\n" + "\n".join(row(r) for r in y.nlargest(6, "residual").itertuples()))
out.append("## 2025 bottom 5 residual\n" + H + "\n" + "\n".join(row(r) for r in y.nsmallest(5, "residual").itertuples()))
ex = [(2024, "Army"), (2025, "Navy"), (2023, "Penn State"), (2025, "Ole Miss"), (2025, "Texas Tech"),
      (2024, "App State"), (2025, "Arkansas"), (2025, "Georgia Tech"), (2019, "UCLA"), (2023, "Washington"), (2023, "Oregon"), (2025, "Indiana"), (2025, "Oregon")]
out.append("## Illustrative rank changes\n" + H + "\n" + "\n".join(row(q[(q.year == a) & (q.team == t)].iloc[0]) for a, t in ex))
(DOCS / "RESULTS_tables.md").write_text("\n\n".join(out) + "\n")

fig, ax = plt.subplots(figsize=(9, 6.5))
ax.scatter(100 * y.raw_success, 100 * y.residual, s=18, c="#999", alpha=.7)
for t in ["Oregon", "Arkansas", "Georgia Tech", "Indiana", "Ohio State", "Navy", "Ole Miss", "North Texas", "Vanderbilt", "Texas Tech"]:
    r = y[y.team == t]
    if len(r):
        r = r.iloc[0]; hl = t == "Oregon"
        ax.scatter(100 * r.raw_success, 100 * r.residual, s=70 if hl else 30, c="#154733" if hl else "#c0392b", zorder=3)
        ax.annotate(t, (100 * r.raw_success, 100 * r.residual), xytext=(4, 3), textcoords="offset points", fontsize=9, weight="bold" if hl else None)
ax.axhline(0, c="k", lw=.6)
ax.set_xlabel("Raw success rate, all FBS-offense snaps (%)"); ax.set_ylabel("Success residual vs expected, non-cupcake snaps (pts)")
ax.set_title(f"2025 FBS offenses (n={len(y)}, >=300 modeling snaps)\nData: CollegeFootballData.com")
fig.tight_layout(); fig.savefig(FIGURES / "raw_vs_residual_2025.png", dpi=150)
print("ok")
