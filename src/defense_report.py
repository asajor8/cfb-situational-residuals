"""Tables + 2025 defensive scatter. Data: CollegeFootballData.com."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = False; plt.rcParams["text.parse_math"] = False
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import PROC, DERIVED, DOCS, FIGURES, ensure_dirs
ensure_dirs()
P = PROC
d = pd.read_csv(DERIVED / "team_season_defense.csv"); q = d[d.d_qualified]
pct = lambda x: f"{100*x:.1f}%"
H = ("| Year | Team | Raw SR allowed (rank) | Cupcake-offense share | Non-cupcake SR allowed (rank) | Expected faced | Def residual, pts (rank) | Def PPA (rank) | Raw def PPA (rank) | n |\n"
     "|---|---|---|---|---|---|---|---|---|---|")
def row(r):
    return (f"| {r.year} | {r.team} | {pct(r.d_raw_success_allowed)} (#{int(r.d_rank_raw_sr)}) | {pct(r.d_share_cupcake)} | {pct(r.d_success_allowed)} (#{int(r.d_rank_sr)}) | "
            f"{pct(r.d_expected_faced)} | {100*r.d_residual:+.1f} (#{int(r.d_rank_residual)}) | {r.d_def_ppa:+.3f} (#{int(r.d_rank_ppa)}) | {r.d_raw_def_ppa:+.3f} (#{int(r.d_rank_raw_ppa)}) | {int(r.d_n)} |")
out = ["## Defense by opponent-offense bucket\n" + pd.read_csv(P / "def_opp_bucket.csv").round(4).to_markdown(index=False),
       "## Def PPA by offense bucket, by year\n" + pd.read_csv(P / "def_ppa_by_offbucket_year.csv").round(4).to_markdown(index=False),
       "## SR allowed by offense bucket, by year\n" + pd.read_csv(P / "def_sr_allowed_by_offbucket_year.csv").round(4).to_markdown(index=False),
       "## Oregon defense\n" + H + "\n" + "\n".join(row(r) for r in q[q.team == "Oregon"].itertuples())]
y = q[q.year == 2025]
for lab, col, asc in [("top by def PPA", "d_def_ppa", False), ("bottom by def PPA", "d_def_ppa", True),
                      ("top by def residual", "d_residual", False), ("bottom by def residual", "d_residual", True)]:
    out.append(f"## 2025 {lab} (qualifiers={len(y)})\n" + H + "\n" + "\n".join(row(r) for r in y.sort_values(col, ascending=asc).head(8).itertuples()))
ex = [(2024, "Louisiana Tech"), (2022, "Toledo"), (2024, "Tulane"), (2019, "UAB"), (2021, "Alabama"),
      (2019, "Air Force"), (2023, "Oklahoma State"), (2024, "Colorado"), (2025, "Buffalo"), (2024, "Memphis"),
      (2023, "Washington"), (2023, "Oregon"), (2022, "Georgia"), (2022, "Oregon")]
out.append("## Illustrative defense rank changes\n" + H + "\n" + "\n".join(row(q[(q.year == a) & (q.team == t)].iloc[0]) for a, t in ex))
(DOCS / "RESULTS_defense_tables.md").write_text("\n\n".join(out) + "\n")
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.scatter(100 * y.d_residual, y.d_def_ppa, s=18, c="#999", alpha=.7)
for t in ["Oregon", "Texas Tech", "James Madison", "Ohio State", "Toledo", "Oklahoma", "Notre Dame", "Air Force", "UAB", "Rutgers", "Tennessee", "Arkansas"]:
    r = y[y.team == t]
    if len(r):
        r = r.iloc[0]; hl = t == "Oregon"
        ax.scatter(100 * r.d_residual, r.d_def_ppa, s=70 if hl else 30, c="#154733" if hl else "#c0392b", zorder=3)
        ax.annotate(t, (100 * r.d_residual, r.d_def_ppa), xytext=(4, 3), textcoords="offset points", fontsize=9, weight="bold" if hl else None)
ax.axhline(y.d_def_ppa.mean(), c="k", lw=.5, ls="--"); ax.axvline(0, c="k", lw=.5)
ax.set_xlabel("Defensive success residual (pts; + = stops beyond situation)"); ax.set_ylabel("Mean defensive PPA per snap (= -PPA allowed)")
ax.set_title(f"2025 FBS defenses vs non-cupcake offenses (n={len(y)}, >=300 snaps)\nData: CollegeFootballData.com; dashed = qualifier mean")
fig.tight_layout(); fig.savefig(FIGURES / "defense_ppa_vs_residual_2025.png", dpi=150); print("ok")
